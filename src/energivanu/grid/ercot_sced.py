# SPDX-License-Identifier: AGPL-3.0-or-later
"""
ERCOT SCED Signal Parser
=========================
Parse ERCOT Security-Constrained Economic Dispatch (SCED) signals for
data center load flexibility (PCLR — Passive Controllable Load Resource).

ERCOT PCLR Framework (approved Jun 18, 2026):
- Data centers can get faster interconnection by agreeing to be dispatchable
- Must respond to SCED base points within 10 minutes
- Must maintain telemetry (ICCP) with ERCOT

This module parses SCED telemetry messages and translates them into
Energivanu MPC + scheduler commands.

Usage::

    from energivanu.grid import ERCOTSCEDClient, SCEDSignal

    client = ERCOTSCEDClient(
        qse_id="QSE001",
        resource_id="DC_LOAD_001",
    )

    # Parse a SCED message
    signal = client.parse_sced_message({
        "basePoint": 150.0,        # Target power in MW
        "lowEmergencyLimit": 100.0,
        "highEmergencyLimit": 200.0,
    })

    # Generate MPC command
    command = client.generate_command(signal, current_power_mw=180.0)
    print(command)  # {"action": "reduce", "target_mw": 150.0, "reduction_mw": 30.0}

References:
    - ERCOT PCLR Nodal Protocol: https://www.ercot.com/
    - PGRR144: PCLR registration requirements
    - NOGRR282: Converter validation
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from ..logging_config import get_logger, timed

logger = get_logger("grid")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class SCEDResponseType(Enum):
    """How the load should respond to SCED signal."""
    NORMAL = "normal"              # No action needed
    REDUCE = "reduce"              # Reduce power to base point
    INCREASE = "increase"          # Increase power to base point
    EMERGENCY_REDUCE = "emergency_reduce"  # Emergency curtailment
    SHED_LOAD = "shed_load"        # Full load shedding


# Response time requirements (seconds)
PCLR_RESPONSE_TIME_S = 600  # 10 minutes for PCLR
NORMAL_RESPONSE_TIME_S = 1800  # 30 minutes for normal loads


@dataclass
class SCEDSignal:
    """A parsed ERCOT SCED signal."""
    base_point_mw: float          # Target power level (MW)
    low_emergency_mw: float       # Low emergency limit (MW)
    high_emergency_mw: float      # High emergency limit (MW)
    timestamp: datetime           # Signal timestamp
    resource_id: str              # Resource identifier
    qse_id: str                   # QSE identifier
    response_type: SCEDResponseType = SCEDResponseType.NORMAL
    response_deadline_s: float = PCLR_RESPONSE_TIME_S

    @property
    def is_emergency(self) -> bool:
        """Check if this is an emergency signal."""
        return self.response_type in (
            SCEDResponseType.EMERGENCY_REDUCE,
            SCEDResponseType.SHED_LOAD,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_point_mw": round(self.base_point_mw, 2),
            "low_emergency_mw": round(self.low_emergency_mw, 2),
            "high_emergency_mw": round(self.high_emergency_mw, 2),
            "timestamp": self.timestamp.isoformat(),
            "resource_id": self.resource_id,
            "qse_id": self.qse_id,
            "response_type": self.response_type.value,
            "response_deadline_s": self.response_deadline_s,
            "is_emergency": self.is_emergency,
        }


@dataclass
class PCLRConfig:
    """PCLR (Passive Controllable Load Resource) configuration."""
    resource_id: str = "DC_LOAD_001"
    qse_id: str = "QSE001"
    max_power_mw: float = 200.0        # Maximum power draw (MW)
    min_power_mw: float = 50.0         # Minimum power (MW)
    ramp_rate_mw_per_min: float = 5.0  # Maximum ramp rate (MW/min)
    response_time_s: float = 600.0     # Required response time (10 min)
    telemetry_interval_s: float = 4.0  # SCED telemetry interval (4s)
    deadband_mw: float = 5.0           # Deadband around base point (MW)


# ---------------------------------------------------------------------------
# ERCOT SCED Client
# ---------------------------------------------------------------------------

class ERCOTSCEDClient:
    """
    ERCOT SCED signal parser and command generator.

    Parses SCED telemetry messages and generates MPC + scheduler
    commands for PCLR compliance.

    Args:
        qse_id: Qualified Scheduling Entity identifier.
        resource_id: Load resource identifier.
        max_power_mw: Maximum facility power draw in MW.
        min_power_mw: Minimum facility power in MW.
        ramp_rate_mw_per_min: Maximum power ramp rate in MW/min.
        response_time_s: Required response time in seconds.
        mpc_controller: MPCController instance for battery dispatch.
        scheduler: PhaseStaggeringScheduler for phase control.
    """

    def __init__(
        self,
        qse_id: str = "QSE001",
        resource_id: str = "DC_LOAD_001",
        max_power_mw: float = 200.0,
        min_power_mw: float = 50.0,
        ramp_rate_mw_per_min: float = 5.0,
        response_time_s: float = 600.0,
        mpc_controller: Optional[Any] = None,
        scheduler: Optional[Any] = None,
    ):
        self.config = PCLRConfig(
            resource_id=resource_id,
            qse_id=qse_id,
            max_power_mw=max_power_mw,
            min_power_mw=min_power_mw,
            ramp_rate_mw_per_min=ramp_rate_mw_per_min,
            response_time_s=response_time_s,
        )
        self.mpc = mpc_controller
        self.scheduler = scheduler

        self._signal_history: List[SCEDSignal] = []
        self._current_signal: Optional[SCEDSignal] = None
        self._compliance_log: List[Dict[str, Any]] = []

        logger.info(
            "ERCOT SCED client initialized",
            extra={
                "qse_id": qse_id,
                "resource_id": resource_id,
                "max_power_mw": max_power_mw,
            },
        )

    # ------------------------------------------------------------------
    # Signal Parsing
    # ------------------------------------------------------------------

    @timed("grid.ercot.parse")
    def parse_sced_message(self, message: Dict[str, Any]) -> SCEDSignal:
        """
        Parse a SCED telemetry message.

        Expected format:
            {
                "basePoint": 150.0,           # Target power (MW)
                "lowEmergencyLimit": 100.0,   # Low emergency limit (MW)
                "highEmergencyLimit": 200.0,  # High emergency limit (MW)
                "timestamp": "2026-06-29T12:00:00Z"  # Optional
            }

        Args:
            message: SCED message dictionary.

        Returns:
            Parsed SCEDSignal.
        """
        base_point = float(message.get("basePoint", self.config.max_power_mw))
        low_emergency = float(message.get("lowEmergencyLimit", self.config.min_power_mw))
        high_emergency = float(message.get("highEmergencyLimit", self.config.max_power_mw))

        # Parse timestamp
        ts_str = message.get("timestamp")
        if ts_str:
            try:
                timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except ValueError:
                timestamp = datetime.now(timezone.utc)
        else:
            timestamp = datetime.now(timezone.utc)

        # Determine response type
        response_type = self._classify_signal(base_point, low_emergency, high_emergency)

        signal = SCEDSignal(
            base_point_mw=base_point,
            low_emergency_mw=low_emergency,
            high_emergency_mw=high_emergency,
            timestamp=timestamp,
            resource_id=self.config.resource_id,
            qse_id=self.config.qse_id,
            response_type=response_type,
            response_deadline_s=self.config.response_time_s,
        )

        self._current_signal = signal
        self._signal_history.append(signal)

        logger.info(
            "SCED signal parsed",
            extra={
                "base_point_mw": base_point,
                "response_type": response_type.value,
                "is_emergency": signal.is_emergency,
            },
        )

        return signal

    def _classify_signal(
        self,
        base_point: float,
        low_emergency: float,
        high_emergency: float,
    ) -> SCEDResponseType:
        """Classify the SCED signal type."""
        deadband = self.config.deadband_mw

        # Emergency conditions
        if base_point <= self.config.min_power_mw + deadband:
            return SCEDResponseType.SHED_LOAD
        if base_point <= low_emergency:
            return SCEDResponseType.EMERGENCY_REDUCE

        # Normal conditions
        if base_point < self.config.max_power_mw - deadband:
            return SCEDResponseType.REDUCE
        if base_point > self.config.max_power_mw + deadband:
            return SCEDResponseType.INCREASE

        return SCEDResponseType.NORMAL

    # ------------------------------------------------------------------
    # Command Generation
    # ------------------------------------------------------------------

    @timed("grid.ercot.command")
    def generate_command(
        self,
        signal: Optional[SCEDSignal] = None,
        current_power_mw: float = 200.0,
        current_soc: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Generate MPC + scheduler commands from SCED signal.

        Args:
            signal: SCED signal. Uses current signal if None.
            current_power_mw: Current facility power draw (MW).
            current_soc: Current BESS state of charge (0-1).

        Returns:
            Dictionary with MPC and scheduler commands.
        """
        if signal is None:
            signal = self._current_signal
        if signal is None:
            return {"action": "none", "reason": "no_signal"}

        # Calculate required power change
        target_mw = signal.base_point_mw
        delta_mw = target_mw - current_power_mw

        # Check if within deadband
        if abs(delta_mw) <= self.config.deadband_mw:
            return {
                "action": "hold",
                "reason": "within_deadband",
                "current_mw": round(current_power_mw, 2),
                "target_mw": round(target_mw, 2),
                "delta_mw": round(delta_mw, 2),
            }

        # Calculate ramp-limited power change
        max_change = self.config.ramp_rate_mw_per_min * (signal.response_deadline_s / 60)
        actual_change = max(-max_change, min(max_change, delta_mw))

        command = {
            "action": "reduce" if actual_change < 0 else "increase",
            "signal": signal.to_dict(),
            "current_mw": round(current_power_mw, 2),
            "target_mw": round(target_mw, 2),
            "delta_mw": round(delta_mw, 2),
            "actual_change_mw": round(actual_change, 2),
            "ramp_limited": abs(actual_change) < abs(delta_mw),
            "response_deadline_s": signal.response_deadline_s,
            "mpc_command": None,
            "scheduler_command": None,
            "compliance": {
                "compliant": True,
                "can_meet_deadline": True,
                "estimated_response_s": self._estimate_response_time(actual_change),
            },
        }

        # Generate MPC command
        if self.mpc is not None and abs(actual_change) > self.config.deadband_mw:
            bess_power = -actual_change  # BESS discharges to reduce grid power
            command["mpc_command"] = {
                "action": "dispatch_bess",
                "target_bess_power_mw": round(bess_power, 2),
                "target_grid_mw": round(target_mw, 2),
                "emergency": signal.is_emergency,
            }

        # Generate scheduler command
        if self.scheduler is not None and signal.is_emergency:
            command["scheduler_command"] = {
                "action": "emergency_stagger",
                "reduction_pct": abs(actual_change) / current_power_mw * 100,
            }

        # Log compliance
        self._compliance_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal": signal.to_dict(),
            "response": command,
        })

        return command

    def _estimate_response_time(self, change_mw: float) -> float:
        """Estimate time to achieve required power change (seconds)."""
        if abs(change_mw) < 0.1:
            return 0.0
        # BESS responds in ~1-2 seconds
        # Phase staggering responds in ~5-10 seconds
        # Job pausing responds in ~30-60 seconds
        if abs(change_mw) < 10:
            return 2.0  # BESS only
        elif abs(change_mw) < 50:
            return 10.0  # BESS + staggering
        else:
            return 60.0  # BESS + staggering + job control

    # ------------------------------------------------------------------
    # Compliance Checking
    # ------------------------------------------------------------------

    def check_compliance(
        self,
        signal: SCEDSignal,
        actual_power_mw: float,
        response_time_s: float,
    ) -> Dict[str, Any]:
        """
        Check if actual response meets ERCOT PCLR requirements.

        Args:
            signal: The SCED signal that was received.
            actual_power_mw: Actual power after response.
            response_time_s: Time taken to respond.

        Returns:
            Compliance report dictionary.
        """
        target = signal.base_point_mw
        error_mw = abs(actual_power_mw - target)
        within_deadband = error_mw <= self.config.deadband_mw
        within_time = response_time_s <= signal.response_deadline_s

        compliant = within_deadband and within_time

        report = {
            "compliant": compliant,
            "target_mw": round(target, 2),
            "actual_mw": round(actual_power_mw, 2),
            "error_mw": round(error_mw, 2),
            "deadband_mw": self.config.deadband_mw,
            "within_deadband": within_deadband,
            "response_time_s": round(response_time_s, 1),
            "deadline_s": signal.response_deadline_s,
            "within_time": within_time,
            "signal_type": signal.response_type.value,
            "is_emergency": signal.is_emergency,
        }

        if not compliant:
            reasons = []
            if not within_deadband:
                reasons.append(f"power_error_{error_mw:.1f}MW_exceeds_deadband")
            if not within_time:
                reasons.append(f"response_{response_time_s:.0f}s_exceeds_deadline")
            report["violation_reasons"] = reasons
            logger.warning(
                "PCLR compliance violation",
                extra=report,
            )

        return report

    # ------------------------------------------------------------------
    # Simulation / Testing
    # ------------------------------------------------------------------

    def simulate_sced_sequence(
        self,
        duration_minutes: float = 60.0,
        interval_s: float = 4.0,
    ) -> List[SCEDSignal]:
        """
        Simulate a sequence of SCED signals for testing.

        Generates realistic SCED signals with varying base points
        to test MPC response and compliance.

        Args:
            duration_minutes: Simulation duration in minutes.
            interval_s: Signal interval in seconds (default 4s per ERCOT).

        Returns:
            List of simulated SCED signals.
        """
        n_signals = int(duration_minutes * 60 / interval_s)
        signals = []

        # Generate realistic power profile
        t = np.linspace(0, duration_minutes * 60, n_signals)
        base_load = self.config.max_power_mw * 0.8  # 80% base load

        # Add variability (mimics real grid conditions)
        variability = self.config.max_power_mw * 0.15 * np.sin(2 * np.pi * t / 3600)
        noise = np.random.normal(0, self.config.max_power_mw * 0.02, n_signals)
        target_power = base_load + variability + noise

        # Add occasional emergency events
        for i in range(n_signals):
            # Random emergency events (~2% of time)
            if np.random.random() < 0.02:
                target_power[i] = self.config.min_power_mw + 10

        for i in range(n_signals):
            now = datetime.now(timezone.utc)
            low_emer = self.config.min_power_mw
            high_emer = self.config.max_power_mw

            signal = SCEDSignal(
                base_point_mw=float(np.clip(target_power[i], low_emer, high_emer)),
                low_emergency_mw=low_emer,
                high_emergency_mw=high_emer,
                timestamp=now,
                resource_id=self.config.resource_id,
                qse_id=self.config.qse_id,
            )
            signal.response_type = self._classify_signal(
                signal.base_point_mw, low_emer, high_emer
            )
            signals.append(signal)

        logger.info(
            "SCED sequence simulated",
            extra={
                "duration_min": duration_minutes,
                "signals": len(signals),
                "emergencies": sum(1 for s in signals if s.is_emergency),
            },
        )

        return signals

    def get_stats(self) -> Dict[str, Any]:
        """Return SCED client statistics."""
        return {
            "resource_id": self.config.resource_id,
            "qse_id": self.config.qse_id,
            "max_power_mw": self.config.max_power_mw,
            "total_signals": len(self._signal_history),
            "compliance_checks": len(self._compliance_log),
            "current_signal": self._current_signal.to_dict() if self._current_signal else None,
        }


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def create_sced_client_from_config(config: Any = None) -> ERCOTSCEDClient:
    """Create an ERCOTSCEDClient from EnergivanuConfig."""
    if config is None:
        from ..config import get_config
        config = get_config()

    return ERCOTSCEDClient(
        qse_id="QSE001",
        resource_id="DC_LOAD_001",
        max_power_mw=config.grid.facility_current_mw,
        min_power_mw=config.grid.facility_current_mw * 0.25,
        ramp_rate_mw_per_min=config.grid.ramp_rate_limit_mw_per_min,
    )
