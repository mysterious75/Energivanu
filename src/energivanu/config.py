# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Energivanu Configuration System
================================
Loads YAML configuration with environment variable overrides and validation.

Usage::

    from energivanu.config import get_config

    cfg = get_config()
    print(cfg.model.n_features)        # 15
    print(cfg.mpc.horizon_steps)        # 12

Environment overrides use the prefix ``ENERGIVANU_`` with double-underscore
separators for nested keys::

    ENERGIVANU_MODEL__N_FEATURES=20 energivanu demo

A singleton pattern ensures all modules share the same configuration object.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

# ---------------------------------------------------------------------------
# Dataclass hierarchy — typed, validated, immutable after creation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """PEB model hyper-parameters."""
    n_features: int = 15
    seq_len: int = 30
    pred_horizon: int = 10
    tcn_channels: Tuple[int, ...] = (32, 64, 128)
    tcn_kernels: Tuple[int, ...] = (5, 3, 3)
    attention_heads: int = 8
    attention_dim: int = 128
    hidden_dims: Tuple[int, ...] = (256, 128)
    n_signal_classes: int = 3
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {self.n_features}")
        if self.seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {self.seq_len}")
        if self.pred_horizon < 1:
            raise ValueError(f"pred_horizon must be >= 1, got {self.pred_horizon}")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyper-parameters."""
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    val_split: float = 0.15
    stride: int = 50
    num_gpus_target: int = 200000
    gpus_per_node: int = 8
    checkpoint_dir: str = "models/checkpoints"
    checkpoint_name: str = "best_model.pt"

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if not 0.0 < self.val_split < 1.0:
            raise ValueError(f"val_split must be in (0, 1), got {self.val_split}")


@dataclass(frozen=True)
class MPCConfig:
    """MPC controller parameters."""
    horizon_steps: int = 12
    step_seconds: int = 5
    soc_min: float = 0.05
    soc_max: float = 0.95
    efficiency: float = 0.92
    Q: float = 100.0
    R: float = 0.01
    S: float = 0.1
    grid_target_mw: float = 200.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.soc_min < self.soc_max <= 1.0:
            raise ValueError(
                f"soc limits must satisfy 0 <= soc_min < soc_max <= 1, "
                f"got [{self.soc_min}, {self.soc_max}]"
            )
        if not 0.0 < self.efficiency <= 1.0:
            raise ValueError(f"efficiency must be in (0, 1], got {self.efficiency}")


@dataclass(frozen=True)
class GridConfig:
    """Grid electrical parameters."""
    nominal_frequency_hz: float = 60.0
    inertia_constant_s: float = 4.5
    ramp_rate_limit_mw_per_min: float = 5.0
    facility_current_mw: float = 200.0
    frequency_deadband_hz: float = 0.02


@dataclass(frozen=True)
class PricingConfig:
    """Electricity tariff / pricing parameters."""
    demand_charge_per_kw_month: float = 15.0
    peak_rate_per_kwh: float = 0.12
    offpeak_rate_per_kwh: float = 0.05
    shoulder_rate_per_kwh: float = 0.08
    peak_hours: Tuple[int, ...] = (14, 15, 16, 17, 18)
    offpeak_hours: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    currency: str = "USD"

    def __post_init__(self) -> None:
        for h in self.peak_hours:
            if not 0 <= h <= 23:
                raise ValueError(f"peak_hours contains invalid hour: {h}")
        for h in self.offpeak_hours:
            if not 0 <= h <= 23:
                raise ValueError(f"offpeak_hours contains invalid hour: {h}")


@dataclass(frozen=True)
class BatteryConfig:
    """Battery energy storage system parameters."""
    total_power_mw: float = 319.2
    total_capacity_mwh: float = 655.2
    chemistry: str = "LFP"
    round_trip_efficiency: float = 0.92
    max_cycles_per_day: int = 2
    min_soc: float = 0.05
    max_soc: float = 0.95

    def __post_init__(self) -> None:
        if self.total_power_mw <= 0:
            raise ValueError(f"total_power_mw must be > 0, got {self.total_power_mw}")
        if self.total_capacity_mwh <= 0:
            raise ValueError(f"total_capacity_mwh must be > 0, got {self.total_capacity_mwh}")


@dataclass(frozen=True)
class ModbusRegistersConfig:
    """Modbus register addresses."""
    soc_register: int = 100
    power_register: int = 102
    status_register: int = 104


@dataclass(frozen=True)
class HardwareConfig:
    """BESS hardware interface parameters."""
    bess_type: str = "Tesla Megapack"
    modbus_host: str = "192.168.1.100"
    modbus_port: int = 502
    modbus_unit_id: int = 1
    modbus_timeout_s: float = 5.0
    modbus_retry_count: int = 3
    modbus_registers: ModbusRegistersConfig = field(default_factory=ModbusRegistersConfig)

    def __post_init__(self) -> None:
        if not 1 <= self.modbus_port <= 65535:
            raise ValueError(f"modbus_port must be in [1, 65535], got {self.modbus_port}")


@dataclass(frozen=True)
class TelemetryConfig:
    """Telemetry collection parameters."""
    source: str = "nvidia-smi"
    collection_interval_s: float = 1.0
    gpu_ids: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
    storage_backend: str = "sqlite"
    sqlite_path: str = "data/telemetry.db"
    csv_path: str = "data/telemetry.csv"
    rolling_window_size: int = 3600
    simulation_mode: bool = False
    simulation_num_gpus: int = 8

    def __post_init__(self) -> None:
        if self.collection_interval_s <= 0:
            raise ValueError(
                f"collection_interval_s must be > 0, got {self.collection_interval_s}"
            )
        if self.storage_backend not in ("sqlite", "csv", "both"):
            raise ValueError(
                f"storage_backend must be sqlite/csv/both, got {self.storage_backend}"
            )


@dataclass(frozen=True)
class MonitoringConfig:
    """Monitoring and observability parameters."""
    prometheus_port: int = 9090
    prometheus_enabled: bool = True
    grafana_url: str = "http://localhost:3000"
    metrics_prefix: str = "energivanu"
    health_check_interval_s: int = 30


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "json"
    log_dir: str = "logs"
    max_bytes: int = 10485760  # 10 MB
    backup_count: int = 5
    console_output: bool = True
    loggers: Dict[str, str] = field(default_factory=lambda: {
        "model": "energivanu.model",
        "mpc": "energivanu.mpc",
        "telemetry": "energivanu.telemetry",
        "api": "energivanu.api",
    })

    def __post_init__(self) -> None:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.level.upper() not in valid_levels:
            raise ValueError(f"level must be one of {valid_levels}, got {self.level}")


@dataclass(frozen=True)
class EnergivanuConfig:
    """Top-level configuration container."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    mpc: MPCConfig = field(default_factory=MPCConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    pricing: PricingConfig = field(default_factory=PricingConfig)
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

# Registry: section name → (dataclass, optional nested registry)
_SECTION_REGISTRY: Dict[str, Any] = {
    "model": ModelConfig,
    "training": TrainingConfig,
    "mpc": MPCConfig,
    "grid": GridConfig,
    "pricing": PricingConfig,
    "battery": BatteryConfig,
    "hardware": HardwareConfig,
    "telemetry": TelemetryConfig,
    "monitoring": MonitoringConfig,
    "logging": LoggingConfig,
}

_NESTED_SECTIONS = {
    "hardware": {"modbus_registers": ModbusRegistersConfig},
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (non-destructive)."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_env_overrides(raw: dict) -> dict:
    """
    Apply environment variable overrides.

    Variables must start with ``ENERGIVANU_`` and use ``__`` (double underscore)
    as a separator for nested keys.  Values are auto-cast to int, float, bool,
    or left as str.

    Examples::

        ENERGIVANU_MODEL__N_FEATURES=20
        ENERGIVANU_MPC__SOC_MIN=0.10
        ENERGIVANU_TELEMETRY__SIMULATION_MODE=true
    """
    prefix = "ENERGIVANU_"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix):].lower().split("__")
        if not parts or not all(parts):
            continue

        # Navigate into nested dict
        d = raw
        for part in parts[:-1]:
            d = d.setdefault(part, {})

        # Auto-cast
        d[parts[-1]] = _auto_cast(value)

    return raw


def _auto_cast(value: str) -> Any:
    """Cast a string value to the most likely Python type."""
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    # Handle YAML-style lists like "[32, 64, 128]"
    if value.startswith("[") and value.endswith("]"):
        try:
            import ast
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
    return value


def _build_section(section_name: str, raw_section: dict) -> Any:
    """Build a frozen dataclass from a dict, handling nested sections."""
    cls = _SECTION_REGISTRY.get(section_name)
    if cls is None:
        return raw_section

    # Handle nested dataclasses
    nested_map = _NESTED_SECTIONS.get(section_name, {})
    for nested_name, nested_cls in nested_map.items():
        if nested_name in raw_section and isinstance(raw_section[nested_name], dict):
            raw_section[nested_name] = nested_cls(**raw_section[nested_name])

    # Filter to only known fields
    import dataclasses
    known = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in raw_section.items() if k in known}
    return cls(**filtered)


def load_config(
    config_path: Optional[str] = None,
    env_overrides: bool = True,
) -> EnergivanuConfig:
    """
    Load configuration from YAML file with optional environment overrides.

    Args:
        config_path: Path to YAML config file. If ``None``, uses
            ``config/default.yaml`` relative to the project root.
        env_overrides: Whether to apply ``ENERGIVANU_*`` environment variables.

    Returns:
        Validated :class:`EnergivanuConfig` instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If a configuration value fails validation.
    """
    if config_path is None:
        # Walk up from this file to find project root (contains config/)
        here = Path(__file__).resolve().parent
        for parent in [here, *here.parents]:
            candidate = parent / "config" / "default.yaml"
            if candidate.exists():
                config_path = str(candidate)
                break
        if config_path is None:
            # Fallback: relative to cwd
            config_path = str(Path.cwd() / "config" / "default.yaml")

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh) or {}

    if env_overrides:
        raw = _apply_env_overrides(raw)

    # Build each section
    sections: Dict[str, Any] = {}
    for section_name, section_cls in _SECTION_REGISTRY.items():
        section_raw = raw.get(section_name, {})
        if not isinstance(section_raw, dict):
            section_raw = {}
        sections[section_name] = _build_section(section_name, section_raw)

    return EnergivanuConfig(**sections)


def config_to_dict(cfg: EnergivanuConfig) -> dict:
    """Convert an EnergivanuConfig back to a plain dict (for serialization)."""
    import dataclasses

    def _convert(obj: Any) -> Any:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {k: _convert(v) for k, v in dataclasses.asdict(obj).items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(item) for item in obj]
        return obj

    return _convert(cfg)


def config_to_mpc_dict(cfg: EnergivanuConfig) -> dict:
    """
    Build the nested dict expected by :class:`energivanu.mpc.MPCController`.

    Returns::

        {
            "mpc": { ... },
            "grid": { ... },
        }
    """
    import dataclasses

    mpc_fields = dataclasses.asdict(cfg.mpc)
    grid_fields = dataclasses.asdict(cfg.grid)
    return {"mpc": mpc_fields, "grid": grid_fields}


def config_to_optimizer_dict(cfg: EnergivanuConfig) -> dict:
    """
    Build the nested dict expected by :class:`energivanu.optimizer.PeakShavingOptimizer`.

    Returns::

        {
            "mpc": { ... },
            "pricing": { ... },
            "battery": { ... },
        }
    """
    import dataclasses

    return {
        "mpc": {"step_seconds": cfg.mpc.step_seconds},
        "pricing": dataclasses.asdict(cfg.pricing),
        "battery": dataclasses.asdict(cfg.battery),
    }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_singleton_lock = threading.Lock()
_singleton_instance: Optional[EnergivanuConfig] = None


def get_config(
    config_path: Optional[str] = None,
    force_reload: bool = False,
) -> EnergivanuConfig:
    """
    Return the global singleton configuration.

    The first call loads and validates the config; subsequent calls return the
    cached instance.  Pass ``force_reload=True`` to re-read from disk.

    Args:
        config_path: Path to YAML config file (only used on first load or
            when ``force_reload`` is ``True``).
        force_reload: If ``True``, discard cached config and reload.

    Returns:
        The global :class:`EnergivanuConfig`.
    """
    global _singleton_instance
    if _singleton_instance is not None and not force_reload:
        return _singleton_instance

    with _singleton_lock:
        if _singleton_instance is not None and not force_reload:
            return _singleton_instance
        _singleton_instance = load_config(config_path)
        return _singleton_instance


def reset_config() -> None:
    """Reset the singleton (useful for testing)."""
    global _singleton_instance
    with _singleton_lock:
        _singleton_instance = None
