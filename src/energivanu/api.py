# SPDX-License-Identifier: AGPL-3.0-or-later
"""Energivanu REST API — power prediction and battery optimization."""

import logging
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("energivanu.api")

app = FastAPI(title="Energivanu API", version="0.1.0")

_model = None
_mpc_controller = None


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool


class PredictRequest(BaseModel):
    power_trace: List[float] = Field(..., description="Historical power values (MW)",
                                      min_length=1, max_length=10000)
    seq_len: int = Field(30, description="Sequence length for model input", ge=1, le=1000)


class PredictResponse(BaseModel):
    power_forecast: List[float]
    signal: str
    signal_probabilities: dict


class BatteryRequest(BaseModel):
    current_power_mw: float = Field(..., ge=-1000, le=1000)
    target_power_mw: float = Field(200.0, ge=0, le=1000)
    soc: float = Field(0.5, ge=0.0, le=1.0)


class BatteryResponse(BaseModel):
    battery_action_mw: float
    grid_power_mw: float
    soc: float
    strategy: str
    freq_deviation_hz: float


class PeakShaveRequest(BaseModel):
    hourly_power: List[float] = Field(..., description="Hourly power trace (MW)",
                                       min_length=1, max_length=8760)


class PeakShaveResponse(BaseModel):
    peak_before_mw: float
    peak_after_mw: float
    peak_reduction_pct: float
    monthly_savings_usd: float
    annual_savings_usd: float


@app.on_event("startup")
def startup_event():
    """Load model and initialize controllers on startup."""
    global _model, _mpc_controller
    import os
    from pathlib import Path

    # Try to load model checkpoint
    checkpoint_path = os.getenv(
        "ENERGIVANU_MODEL_PATH",
        str(Path(__file__).resolve().parent.parent.parent
            / "models" / "checkpoints" / "best_model_demo.pt")
    )
    try:
        if os.path.exists(checkpoint_path):
            from .model import load_model
            _model = load_model(checkpoint_path)
            logger.info(f"Model loaded from {checkpoint_path}")
        else:
            logger.warning(f"No checkpoint at {checkpoint_path}, using fallback predictions")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

    # Initialize persistent MPC controller
    try:
        from .mpc import MPCController
        _mpc_controller = MPCController()
        logger.info("MPC controller initialized")
    except Exception as e:
        logger.error(f"Failed to init MPC: {e}")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        version="0.1.0",
        model_loaded=_model is not None,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        trace = np.array(req.power_trace[-req.seq_len:], dtype=np.float32)
        if len(trace) < req.seq_len:
            trace = np.pad(trace, (req.seq_len - len(trace), 0), mode="edge")

        # Check for NaN/Inf
        if np.any(np.isnan(trace)) or np.any(np.isinf(trace)):
            raise HTTPException(status_code=400, detail="power_trace contains NaN or Inf values")

        if _model is not None:
            import torch
            n_features = _model.n_features
            # Build proper feature tensor: repeat power trace across all feature dims
            x = torch.tensor(trace).unsqueeze(0)  # (1, seq_len)
            x = x.unsqueeze(-1).expand(-1, -1, n_features)  # (1, seq_len, n_features)
            with torch.no_grad():
                power_pred, signal_logits = _model(x)
            probs = torch.softmax(signal_logits, dim=-1).squeeze().numpy()
            signal_idx = int(np.argmax(probs))
            power_forecast = power_pred.squeeze().tolist()
        else:
            # Fallback: simple mean-based prediction
            mean_power = float(np.mean(trace))
            power_forecast = [mean_power] * 10
            probs = np.array([0.4, 0.35, 0.25])
            signal_idx = 0

        signal_names = ["hold", "discharge", "charge"]
        return PredictResponse(
            power_forecast=power_forecast,
            signal=signal_names[signal_idx],
            signal_probabilities={
                signal_names[i]: round(float(probs[i]), 4) for i in range(3)
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/optimize/battery", response_model=BatteryResponse)
def optimize_battery(req: BatteryRequest):
    try:
        global _mpc_controller
        if _mpc_controller is None:
            from .mpc import MPCController
            _mpc_controller = MPCController()

        _mpc_controller.reset(req.soc)
        history = [req.current_power_mw]
        _, info = _mpc_controller.optimize(req.current_power_mw, history, req.target_power_mw)
        return BatteryResponse(
            battery_action_mw=info["battery_action_mw"],
            grid_power_mw=info["grid_power_mw"],
            soc=info["soc"],
            strategy="active" if abs(info["battery_action_mw"]) > 0.1 else "hold",
            freq_deviation_hz=info["freq_deviation_hz"],
        )
    except Exception as e:
        logger.error(f"Battery optimization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Battery optimization failed: {str(e)}")


@app.post("/optimize/peak-shave", response_model=PeakShaveResponse)
def optimize_peak_shave(req: PeakShaveRequest):
    try:
        from .optimizer import PeakShavingOptimizer
        opt = PeakShavingOptimizer()
        monthly = np.tile(req.hourly_power, 30)
        result = opt.simulate_month(monthly)
        annual = opt.estimate_annual_savings(result["total_monthly_demand_savings_usd"])
        return PeakShaveResponse(
            peak_before_mw=result["peak_before_mw"],
            peak_after_mw=result["peak_after_mw"],
            peak_reduction_pct=result["peak_reduction_pct"],
            monthly_savings_usd=result["total_monthly_demand_savings_usd"],
            annual_savings_usd=annual["total_annual_savings"],
        )
    except Exception as e:
        logger.error(f"Peak shaving failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Peak shaving failed: {str(e)}")
