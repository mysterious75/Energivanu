# SPDX-License-Identifier: AGPL-3.0-or-later
"""
PyBaMM Physics-Based Battery Simulator
========================================
Realistic battery simulation using PyBaMM (Python Battery Mathematical Modelling).
Models real battery physics: degradation, thermal effects, voltage curves.

Falls back to simplified model if PyBaMM is not installed.

Usage::

    from energivanu.bess import PyBaMMBattery

    battery = PyBaMMBattery(capacity_mwh=655.2, max_power_mw=319.2)
    battery.initialize(soc=0.5)

    # Simulate charge/discharge
    state = battery.step(power_mw=50.0, dt_seconds=5.0)
    print(f"SOC: {state.soc:.2%}, Voltage: {state.voltage_v:.1f}V")

    # Get degradation info
    print(f"Capacity fade: {state.capacity_fade_pct:.3f}%")
    print(f"Cycle count: {state.cycle_count}")

Dependencies::

    pip install pybamm   # Physics-based battery modelling
    # If not installed, uses simplified linear model
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from ..logging_config import get_logger, timed

logger = get_logger("bess")

# ---------------------------------------------------------------------------
# Try importing PyBaMM
# ---------------------------------------------------------------------------

_PYBAMM_AVAILABLE = False
try:
    import pybamm
    _PYBAMM_AVAILABLE = True
    logger.info("PyBaMM available — using physics-based battery model")
except ImportError:
    logger.warning(
        "PyBaMM not installed — using simplified battery model. "
        "Install with: pip install pybamm"
    )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BatteryState:
    """Complete battery state at a point in time."""
    soc: float                    # State of Charge (0.0 - 1.0)
    voltage_v: float              # Terminal voltage (V)
    current_a: float              # Current (A), positive = discharge
    power_mw: float               # Power output (MW), positive = discharge
    temperature_c: float          # Battery temperature (°C)
    capacity_fade_pct: float      # Capacity fade from nominal (%)
    cycle_count: float            # Equivalent full cycles
    internal_resistance_ohm: float # Internal resistance (Ω)
    heat_generation_w: float      # Heat generation rate (W)
    timestamp: float = 0.0        # Unix timestamp

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "soc": round(self.soc, 6),
            "voltage_v": round(self.voltage_v, 2),
            "current_a": round(self.current_a, 1),
            "power_mw": round(self.power_mw, 4),
            "temperature_c": round(self.temperature_c, 1),
            "capacity_fade_pct": round(self.capacity_fade_pct, 4),
            "cycle_count": round(self.cycle_count, 2),
            "internal_resistance_ohm": round(self.internal_resistance_ohm, 4),
            "heat_generation_w": round(self.heat_generation_w, 1),
            "timestamp": self.timestamp,
        }


@dataclass
class BatteryConfig:
    """Battery system configuration."""
    capacity_mwh: float = 655.2       # Total capacity (MWh)
    max_power_mw: float = 319.2       # Max charge/discharge power (MW)
    nominal_voltage_v: float = 1200.0 # Nominal voltage (V)
    num_cells: int = 1000             # Number of cells in pack
    soc_min: float = 0.05             # Minimum SOC
    soc_max: float = 0.95             # Maximum SOC
    efficiency_charge: float = 0.92   # Charging efficiency
    efficiency_discharge: float = 0.92 # Discharging efficiency
    initial_temp_c: float = 25.0      # Initial temperature
    chemistry: str = "LFP"            # Battery chemistry (LFP/NMC)


# ---------------------------------------------------------------------------
# PyBaMM Battery Model
# ---------------------------------------------------------------------------

class PyBaMMBattery:
    """
    Physics-based battery simulation using PyBaMM.

    When PyBaMM is available, uses real electrochemical models (SPMe/DFN).
    When PyBaMM is not available, uses a simplified linear model with
    realistic parameters.

    Args:
        capacity_mwh: Total battery capacity in MWh.
        max_power_mw: Maximum charge/discharge power in MW.
        chemistry: Battery chemistry ("LFP" or "NMC").
        soc_min: Minimum allowed SOC.
        soc_max: Maximum allowed SOC.
        efficiency_charge: Charging round-trip efficiency.
        efficiency_discharge: Discharging round-trip efficiency.
    """

    def __init__(
        self,
        capacity_mwh: float = 655.2,
        max_power_mw: float = 319.2,
        chemistry: str = "LFP",
        soc_min: float = 0.05,
        soc_max: float = 0.95,
        efficiency_charge: float = 0.92,
        efficiency_discharge: float = 0.92,
    ):
        self.config = BatteryConfig(
            capacity_mwh=capacity_mwh,
            max_power_mw=max_power_mw,
            chemistry=chemistry,
            soc_min=soc_min,
            soc_max=soc_max,
            efficiency_charge=efficiency_charge,
            efficiency_discharge=efficiency_discharge,
        )

        self._soc: float = 0.5
        self._temperature_c: float = 25.0
        self._cycle_count: float = 0.0
        self._total_energy_throughput_mwh: float = 0.0
        self._capacity_fade_pct: float = 0.0
        self._step_count: int = 0
        self._history: List[BatteryState] = []

        # PyBaMM model (initialized on first use)
        self._pybamm_model = None
        self._pybamm_solution = None

        # LFP voltage curve parameters (realistic)
        self._voltage_params = self._get_voltage_params(chemistry)

        logger.info(
            "battery initialized",
            extra={
                "capacity_mwh": capacity_mwh,
                "max_power_mw": max_power_mw,
                "chemistry": chemistry,
                "pybamm_available": _PYBAMM_AVAILABLE,
            },
        )

    @staticmethod
    def _get_voltage_params(chemistry: str) -> Dict[str, float]:
        """Get voltage curve parameters for battery chemistry."""
        if chemistry == "LFP":
            return {
                "ocv_full": 3.65,      # Open circuit voltage at 100% SOC
                "ocv_empty": 2.5,      # Open circuit voltage at 0% SOC
                "nominal_v": 3.2,      # Nominal voltage
                "internal_r_base": 0.002,  # Base internal resistance (Ω per cell)
                "r_temp_coeff": 0.001,     # Resistance temperature coefficient
                "degradation_rate": 0.0001, # Capacity fade per cycle (%)
                "thermal_mass": 50000.0,    # Thermal mass (J/K)
                "heat_coeff": 0.01,         # Heat generation coefficient
            }
        else:  # NMC
            return {
                "ocv_full": 4.2,
                "ocv_empty": 2.5,
                "nominal_v": 3.6,
                "internal_r_base": 0.0015,
                "r_temp_coeff": 0.0008,
                "degradation_rate": 0.0002,
                "thermal_mass": 45000.0,
                "heat_coeff": 0.012,
            }

    def initialize(self, soc: float = 0.5, temperature_c: float = 25.0) -> None:
        """Set initial battery state."""
        self._soc = max(self.config.soc_min, min(self.config.soc_max, soc))
        self._temperature_c = temperature_c
        self._step_count = 0
        self._history.clear()
        logger.info("battery initialized", extra={"soc": self._soc, "temp_c": temperature_c})

    @timed("bess.step")
    def step(self, power_mw: float, dt_seconds: float = 5.0) -> BatteryState:
        """
        Simulate one time step of battery operation.

        Args:
            power_mw: Power command in MW. Positive = discharge, negative = charge.
            dt_seconds: Time step duration in seconds.

        Returns:
            BatteryState after the step.
        """
        # Clip power to limits
        power_mw = max(-self.config.max_power_mw, min(self.config.max_power_mw, power_mw))

        # Check SOC limits
        if power_mw > 0 and self._soc <= self.config.soc_min:
            power_mw = 0.0  # Can't discharge below min
        elif power_mw < 0 and self._soc >= self.config.soc_max:
            power_mw = 0.0  # Can't charge above max

        if _PYBAMM_AVAILABLE:
            state = self._step_pybamm(power_mw, dt_seconds)
        else:
            state = self._step_simplified(power_mw, dt_seconds)

        self._history.append(state)
        self._step_count += 1
        return state

    def _step_simplified(self, power_mw: float, dt_seconds: float) -> BatteryState:
        """Simplified battery model with realistic physics."""
        p = self._voltage_params

        # Current calculation
        capacity_ah = self.config.capacity_mwh * 1e6 / self.config.nominal_voltage_v
        capacity_ah  # 1C rate (kept for reference)

        # Internal resistance (increases with temperature deviation and SOC)
        temp_factor = 1.0 + p["r_temp_coeff"] * abs(self._temperature_c - 25.0)
        soc_factor = 1.0 + 0.5 * abs(self._soc - 0.5)  # Higher resistance at extremes
        r_internal = p["internal_r_base"] * temp_factor * soc_factor

        # Open circuit voltage (SOC-dependent)
        ocv = p["ocv_empty"] + (p["ocv_full"] - p["ocv_empty"]) * self._soc

        # Terminal voltage and current
        if abs(power_mw) < 0.001:
            voltage_v = ocv * self.config.num_cells
            current_a = 0.0
            actual_power_mw = 0.0
        else:
            # P = V * I = (OCV - I*R) * I  →  solve quadratic
            power_w = abs(power_mw) * 1e6
            total_ocv = ocv * self.config.num_cells
            total_r = r_internal * self.config.num_cells

            # Quadratic: R*I^2 - OCV*I + P = 0
            discriminant = total_ocv**2 - 4 * total_r * power_w
            if discriminant < 0:
                # Power too high, clip
                max_power_w = total_ocv**2 / (4 * total_r)
                power_w = max_power_w * 0.95
                discriminant = total_ocv**2 - 4 * total_r * power_w

            if power_mw > 0:  # Discharge
                current_a = (total_ocv - math.sqrt(max(0, discriminant))) / (2 * total_r)
                _eff = self.config.efficiency_discharge
            else:  # Charge
                current_a = -(total_ocv - math.sqrt(max(0, discriminant))) / (2 * total_r)
                __eff = self.config.efficiency_charge

            voltage_v = total_ocv - current_a * total_r
            actual_power_mw = voltage_v * current_a / 1e6

        # SOC update
        energy_wh = actual_power_mw * 1e6 * dt_seconds / 3600
        if power_mw > 0:  # Discharge
            energy_wh /= self.config.efficiency_discharge
        else:  # Charge
            energy_wh *= self.config.efficiency_charge

        soc_change = energy_wh / (self.config.capacity_mwh * 1e6)
        self._soc -= soc_change  # Discharge decreases SOC
        self._soc = max(self.config.soc_min, min(self.config.soc_max, self._soc))

        # Temperature model (simplified thermal)
        heat_w = (
            abs(power_mw) * 1e6 * p["heat_coeff"]
            + (current_a**2 * r_internal * self.config.num_cells)
        )
        temp_change = heat_w * dt_seconds / p["thermal_mass"]
        cooling = 0.01 * (self._temperature_c - 25.0) * dt_seconds  # Simple Newton cooling
        self._temperature_c += temp_change - cooling
        self._temperature_c = max(15.0, min(55.0, self._temperature_c))

        # Degradation model (cycle counting)
        self._total_energy_throughput_mwh += abs(actual_power_mw) * dt_seconds / 3600
        equivalent_cycles = self._total_energy_throughput_mwh / (2 * self.config.capacity_mwh)
        self._capacity_fade_pct = equivalent_cycles * p["degradation_rate"] * 100

        return BatteryState(
            soc=self._soc,
            voltage_v=voltage_v,
            current_a=current_a,
            power_mw=actual_power_mw,
            temperature_c=self._temperature_c,
            capacity_fade_pct=self._capacity_fade_pct,
            cycle_count=equivalent_cycles,
            internal_resistance_ohm=r_internal * self.config.num_cells,
            heat_generation_w=heat_w,
        )

    def _step_pybamm(self, power_mw: float, dt_seconds: float) -> BatteryState:
        """PyBaMM physics-based model (when available)."""
        # Initialize PyBaMM model on first use
        if self._pybamm_model is None:
            self._pybamm_model = pybamm.lithium_ion.SPM()

        # Convert power to current
        _capacity_ah = self.config.capacity_mwh * 1e6 / self.config.nominal_voltage_v
        if abs(power_mw) < 0.001:
            current_a = 0.0
        else:
            current_a = power_mw * 1e6 / self.config.nominal_voltage_v

        # Run PyBaMM simulation step
        try:
            sim = pybamm.Simulation(
                self._pybamm_model,
                parameter_values=pybamm.ParameterValues("Chen2020"),
            )
            sim.step(dt_seconds, external_variables={
                "Current function [A]": current_a,
            })

            # Extract results
            sol = sim.solution
            soc = float(sol["State of Charge"].entries[-1])
            voltage = float(sol["Terminal voltage [V]"].entries[-1])
            temp = float(sol["X-averaged cell temperature [K]"].entries[-1]) - 273.15

            self._soc = max(self.config.soc_min, min(self.config.soc_max, soc))
            self._temperature_c = temp

            return BatteryState(
                soc=self._soc,
                voltage_v=voltage,
                current_a=current_a,
                power_mw=power_mw,
                temperature_c=self._temperature_c,
                capacity_fade_pct=self._capacity_fade_pct,
                cycle_count=self._total_energy_throughput_mwh / (2 * self.config.capacity_mwh),
                internal_resistance_ohm=0.002 * self.config.num_cells,
                heat_generation_w=abs(current_a) * 0.1 * self.config.num_cells,
            )
        except Exception as e:
            logger.warning(f"PyBaMM step failed, falling back: {e}")
            return self._step_simplified(power_mw, dt_seconds)

    def get_state(self) -> BatteryState:
        """Get current battery state without stepping."""
        if self._history:
            return self._history[-1]
        return BatteryState(
            soc=self._soc,
            voltage_v=self.config.nominal_voltage_v,
            current_a=0.0,
            power_mw=0.0,
            temperature_c=self._temperature_c,
            capacity_fade_pct=0.0,
            cycle_count=0.0,
            internal_resistance_ohm=0.002 * self.config.num_cells,
            heat_generation_w=0.0,
        )

    def get_history(self) -> List[BatteryState]:
        """Return full state history."""
        return list(self._history)

    def reset(self, soc: float = 0.5) -> None:
        """Reset battery to initial state."""
        self.initialize(soc)

    def get_metrics(self) -> Dict[str, Any]:
        """Return summary metrics."""
        if not self._history:
            return {"steps": 0}

        socs = [s.soc for s in self._history]
        powers = [s.power_mw for s in self._history]
        temps = [s.temperature_c for s in self._history]

        return {
            "steps": len(self._history),
            "current_soc": round(self._soc, 4),
            "min_soc": round(min(socs), 4),
            "max_soc": round(max(socs), 4),
            "avg_power_mw": round(np.mean(np.abs(powers)), 4),
            "max_power_mw": round(max(np.abs(powers)), 4),
            "current_temp_c": round(self._temperature_c, 1),
            "max_temp_c": round(max(temps), 1),
            "cycle_count": round(
                self._total_energy_throughput_mwh / (2 * self.config.capacity_mwh), 2
            ),
            "capacity_fade_pct": round(self._capacity_fade_pct, 4),
            "total_energy_throughput_mwh": round(self._total_energy_throughput_mwh, 2),
            "chemistry": self.config.chemistry,
            "capacity_mwh": self.config.capacity_mwh,
            "max_power_mw_limit": self.config.max_power_mw,
            "pybamm_used": _PYBAMM_AVAILABLE,
        }


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def create_battery_from_config(config: Any = None) -> PyBaMMBattery:
    """Create a PyBaMMBattery from EnergivanuConfig."""
    if config is None:
        from ..config import get_config
        config = get_config()

    return PyBaMMBattery(
        capacity_mwh=config.battery.total_capacity_mwh,
        max_power_mw=config.battery.total_power_mw,
        chemistry=config.battery.chemistry,
        soc_min=config.battery.min_soc,
        soc_max=config.battery.max_soc,
        efficiency_charge=config.battery.round_trip_efficiency,
        efficiency_discharge=config.battery.round_trip_efficiency,
    )
