# SPDX-License-Identifier: AGPL-3.0-or-later
"""
CodeCarbon Energy Tracker
=========================
Wraps the CodeCarbon ``EmissionsTracker`` to measure GPU energy consumption
during model training and estimate electricity costs.

Usage::

    from energivanu.telemetry.codecarbon_tracker import EnergivanuCarbonTracker

    tracker = EnergivanuCarbonTracker(electricity_rate_usd_per_kwh=0.12)
    tracker.start_training_tracking()

    for epoch in range(num_epochs):
        train_one_epoch(model, loader)
        epoch_stats = tracker.stop_epoch()
        print(f"Epoch energy: {epoch_stats['energy_kwh']:.4f} kWh, "
              f"cost: ${epoch_stats['cost_usd']:.4f}")
        if epoch < num_epochs - 1:
            tracker.start_epoch()

    final = tracker.stop_training_tracking()
    tracker.export_csv("carbon_report.csv")

Dependencies:
    pip install codecarbon

If CodeCarbon is not installed, the tracker falls back to a lightweight
estimation based on nvidia-smi power readings.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..logging_config import get_logger, timed

logger = get_logger("telemetry")

# ---------------------------------------------------------------------------
# Try importing CodeCarbon
# ---------------------------------------------------------------------------

_CODECARBON_AVAILABLE = False
_EmissionsTracker = None  # type: ignore[type-arg]

try:
    from codecarbon import EmissionsTracker as _EmissionsTracker
    _CODECARBON_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EpochStats:
    """Energy and cost statistics for a single training epoch."""
    epoch: int
    start_time: str
    end_time: str
    duration_s: float
    energy_kwh: float
    emissions_kg: float
    cost_usd: float
    gpu_power_w: float
    cpu_power_w: float


@dataclass
class TrainingStats:
    """Cumulative energy and cost statistics for an entire training run."""
    start_time: str
    end_time: str
    total_duration_s: float
    total_energy_kwh: float
    total_emissions_kg: float
    total_cost_usd: float
    num_epochs: int
    avg_energy_per_epoch_kwh: float
    avg_cost_per_epoch_usd: float
    electricity_rate_usd_per_kwh: float


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class EnergivanuCarbonTracker:
    """
    GPU energy tracker with cost estimation for ML training.

    Integrates with CodeCarbon when available, otherwise uses a lightweight
    nvidia-smi-based estimator.

    Args:
        electricity_rate_usd_per_kwh: Cost per kWh of electricity.
        project_name: Project name for CodeCarbon logs.
        output_dir: Directory for CodeCarbon output and CSV exports.
        country_iso_code: ISO country code for emission factor lookup.
        region: Region/state for more accurate emission factors.
        gpu_ids: List of GPU IDs to track.  ``None`` = all.
    """

    def __init__(
        self,
        electricity_rate_usd_per_kwh: float = 0.12,
        project_name: str = "energivanu",
        output_dir: str = "data/carbon",
        country_iso_code: str = "USA",
        region: str = "",
        gpu_ids: Optional[List[int]] = None,
    ):
        self.electricity_rate = electricity_rate_usd_per_kwh
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.country_iso_code = country_iso_code
        self.region = region
        self.gpu_ids = gpu_ids

        self._cc_tracker: Any = None
        self._epoch_trackers: List[EpochStats] = []
        self._current_epoch: int = 0
        self._training_start: float = 0.0
        self._epoch_start: float = 0.0
        self._training_started: bool = False
        self._fallback_power_samples: List[float] = []

        if _CODECARBON_AVAILABLE:
            logger.info("CodeCarbon available — using native tracking")
        else:
            logger.warning(
                "CodeCarbon not installed — using nvidia-smi fallback estimator. "
                "Install with: pip install codecarbon"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_training_tracking(self) -> None:
        """
        Begin tracking energy for an entire training run.

        Creates a CodeCarbon tracker (if available) and records the start time.
        """
        if self._training_started:
            logger.warning("training tracking already started — resetting")
            self.stop_training_tracking()

        self._epoch_trackers = []
        self._current_epoch = 0
        self._training_start = time.time()
        self._training_started = True

        if _CODECARBON_AVAILABLE and _EmissionsTracker is not None:
            self._cc_tracker = _EmissionsTracker(
                project_name=self.project_name,
                output_dir=str(self.output_dir),
                country_iso_code=self.country_iso_code,
                region=self.region if self.region else None,
                log_level="warning",
            )
            self._cc_tracker.start()
            logger.info("CodeCarbon training tracker started")
        else:
            self._fallback_power_samples = []
            logger.info("fallback training tracker started")

    def start_epoch(self, epoch: Optional[int] = None) -> None:
        """
        Mark the beginning of a training epoch.

        Args:
            epoch: Epoch number.  Auto-increments if ``None``.
        """
        if not self._training_started:
            self.start_training_tracking()

        if epoch is not None:
            self._current_epoch = epoch
        self._epoch_start = time.time()

        if _CODECARBON_AVAILABLE and self._cc_tracker is not None:
            # CodeCarbon tracks continuously; we just mark boundaries
            pass
        else:
            self._fallback_power_samples = []

        logger.debug("epoch %d started", self._current_epoch)

    @timed("carbon.epoch")
    def stop_epoch(self) -> Dict[str, Any]:
        """
        End the current epoch and compute energy/cost statistics.

        Returns:
            Dictionary with epoch energy statistics.
        """
        now = time.time()
        duration_s = now - self._epoch_start if self._epoch_start > 0 else 0.0

        if _CODECARBON_AVAILABLE and self._cc_tracker is not None:
            # CodeCarbon tracks cumulatively; compute delta
            emissions_data = self._cc_tracker._prepare_emissions_data()
            energy_kwh = (
                float(emissions_data.energy.kWh)
                if hasattr(emissions_data, 'energy') else 0.0
            )
            emissions_kg = (
                float(emissions_data.emissions)
                if hasattr(emissions_data, 'emissions') else 0.0
            )
            # Use a rough GPU power estimate for cost
            gpu_power_w = self._estimate_gpu_power()
            cost_usd = energy_kwh * self.electricity_rate
        else:
            energy_kwh, emissions_kg, gpu_power_w = self._estimate_energy_fallback(duration_s)
            cost_usd = energy_kwh * self.electricity_rate

        stats = EpochStats(
            epoch=self._current_epoch,
            start_time=datetime.fromtimestamp(self._epoch_start, tz=timezone.utc).isoformat(),
            end_time=datetime.now(timezone.utc).isoformat(),
            duration_s=round(duration_s, 2),
            energy_kwh=round(energy_kwh, 6),
            emissions_kg=round(emissions_kg, 6),
            cost_usd=round(cost_usd, 6),
            gpu_power_w=round(gpu_power_w, 1),
            cpu_power_w=0.0,
        )

        self._epoch_trackers.append(stats)
        self._current_epoch += 1

        logger.info(
            "epoch %d energy tracked",
            stats.epoch,
            extra={
                "energy_kwh": stats.energy_kwh,
                "cost_usd": stats.cost_usd,
                "duration_s": stats.duration_s,
                "gpu_power_w": stats.gpu_power_w,
            },
        )

        return {
            "epoch": stats.epoch,
            "energy_kwh": stats.energy_kwh,
            "emissions_kg": stats.emissions_kg,
            "cost_usd": stats.cost_usd,
            "duration_s": stats.duration_s,
            "gpu_power_w": stats.gpu_power_w,
        }

    @timed("carbon.training")
    def stop_training_tracking(self) -> Dict[str, Any]:
        """
        End training tracking and compute cumulative statistics.

        Returns:
            Dictionary with training-wide energy statistics.
        """
        now = time.time()
        total_duration = now - self._training_start if self._training_start > 0 else 0.0

        if _CODECARBON_AVAILABLE and self._cc_tracker is not None:
            try:
                self._cc_tracker.stop()
            except Exception as exc:
                logger.error("CodeCarbon stop error", extra={"error": str(exc)})
            self._cc_tracker = None

        total_energy = sum(e.energy_kwh for e in self._epoch_trackers)
        total_emissions = sum(e.emissions_kg for e in self._epoch_trackers)
        total_cost = sum(e.cost_usd for e in self._epoch_trackers)
        n_epochs = len(self._epoch_trackers)

        stats = TrainingStats(
            start_time=datetime.fromtimestamp(self._training_start, tz=timezone.utc).isoformat(),
            end_time=datetime.now(timezone.utc).isoformat(),
            total_duration_s=round(total_duration, 2),
            total_energy_kwh=round(total_energy, 6),
            total_emissions_kg=round(total_emissions, 6),
            total_cost_usd=round(total_cost, 6),
            num_epochs=n_epochs,
            avg_energy_per_epoch_kwh=round(total_energy / max(1, n_epochs), 6),
            avg_cost_per_epoch_usd=round(total_cost / max(1, n_epochs), 6),
            electricity_rate_usd_per_kwh=self.electricity_rate,
        )

        self._training_started = False

        logger.info(
            "training carbon tracking complete",
            extra={
                "total_energy_kwh": stats.total_energy_kwh,
                "total_cost_usd": stats.total_cost_usd,
                "num_epochs": stats.num_epochs,
                "avg_energy_per_epoch_kwh": stats.avg_energy_per_epoch_kwh,
            },
        )

        return {
            "total_energy_kwh": stats.total_energy_kwh,
            "total_emissions_kg": stats.total_emissions_kg,
            "total_cost_usd": stats.total_cost_usd,
            "num_epochs": stats.num_epochs,
            "total_duration_s": stats.total_duration_s,
            "avg_energy_per_epoch_kwh": stats.avg_energy_per_epoch_kwh,
            "avg_cost_per_epoch_usd": stats.avg_cost_per_epoch_usd,
            "electricity_rate_usd_per_kwh": stats.electricity_rate_usd_per_kwh,
        }

    def get_epoch_history(self) -> List[Dict[str, Any]]:
        """Return per-epoch energy statistics as a list of dicts."""
        return [
            {
                "epoch": e.epoch,
                "energy_kwh": e.energy_kwh,
                "emissions_kg": e.emissions_kg,
                "cost_usd": e.cost_usd,
                "duration_s": e.duration_s,
                "gpu_power_w": e.gpu_power_w,
            }
            for e in self._epoch_trackers
        ]

    @timed("carbon.export_csv")
    def export_csv(self, filepath: Optional[str] = None) -> str:
        """
        Export per-epoch energy data to CSV.

        Args:
            filepath: Output path.  Defaults to ``<output_dir>/carbon_report.csv``.

        Returns:
            Path to the exported CSV file.
        """
        if filepath is None:
            filepath = str(self.output_dir / "carbon_report.csv")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "epoch", "start_time", "end_time", "duration_s",
                "energy_kwh", "emissions_kg", "cost_usd",
                "gpu_power_w", "cpu_power_w",
            ])
            for e in self._epoch_trackers:
                writer.writerow([
                    e.epoch, e.start_time, e.end_time, e.duration_s,
                    e.energy_kwh, e.emissions_kg, e.cost_usd,
                    e.gpu_power_w, e.cpu_power_w,
                ])

            # Summary row
            writer.writerow([])
            total_energy = sum(e.energy_kwh for e in self._epoch_trackers)
            total_cost = sum(e.cost_usd for e in self._epoch_trackers)
            writer.writerow(["TOTAL", "", "", "", total_energy, "", total_cost, "", ""])
            writer.writerow(["RATE", "", "", "", "", "", self.electricity_rate, "", ""])

        logger.info("carbon report exported", extra={"path": str(path)})
        return str(path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_gpu_power(self) -> float:
        """Estimate current GPU power draw via nvidia-smi (W)."""
        try:
            import subprocess
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                powers = [
                    float(x.strip())
                    for x in result.stdout.strip().split("\n")
                    if x.strip()
                ]
                if powers:
                    return sum(powers)
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, OSError):
            pass

        # Fallback: assume 400 W per GPU (H100 typical training load)
        num_gpus = len(self.gpu_ids) if self.gpu_ids else 8
        return 400.0 * num_gpus

    def _estimate_energy_fallback(self, duration_s: float) -> tuple:
        """
        Estimate energy consumption without CodeCarbon.

        Returns:
            (energy_kwh, emissions_kg, gpu_power_w)
        """
        gpu_power_w = self._estimate_gpu_power()
        # Add ~15% for CPU/overhead
        total_power_w = gpu_power_w * 1.15
        energy_kwh = total_power_w * duration_s / 3_600_000  # W*s -> kWh

        # US average: ~0.42 kg CO2/kWh
        emissions_kg = energy_kwh * 0.42

        return energy_kwh, emissions_kg, gpu_power_w

    def estimate_cost_for_workload(
        self,
        duration_hours: float,
        num_gpus: int = 8,
        gpu_tdp_w: float = 700.0,
        utilization_pct: float = 80.0,
    ) -> Dict[str, float]:
        """
        Estimate energy cost for a hypothetical workload.

        Args:
            duration_hours: Expected training duration in hours.
            num_gpus: Number of GPUs.
            gpu_tdp_w: GPU TDP in watts.
            utilization_pct: Expected GPU utilization (0-100).

        Returns:
            Dictionary with energy and cost estimates.
        """
        gpu_power = num_gpus * gpu_tdp_w * (utilization_pct / 100.0)
        overhead_power = gpu_power * 0.15  # CPU, cooling, networking
        total_power = gpu_power + overhead_power

        energy_kwh = total_power * duration_hours / 1000.0
        cost_usd = energy_kwh * self.electricity_rate
        emissions_kg = energy_kwh * 0.42

        return {
            "gpu_power_kw": round(gpu_power / 1000, 2),
            "total_power_kw": round(total_power / 1000, 2),
            "energy_kwh": round(energy_kwh, 2),
            "cost_usd": round(cost_usd, 2),
            "emissions_kg": round(emissions_kg, 2),
            "duration_hours": duration_hours,
            "electricity_rate": self.electricity_rate,
        }
