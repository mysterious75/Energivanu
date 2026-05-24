"""
ENERGIVANU — Central Configuration
"""

from dataclasses import dataclass, field


@dataclass
class ClusterConfig:
    num_gpus: int = 135_000
    gpu_tdp_watts: float = 700.0
    gpu_idle_watts: float = 75.0
    location: str = "Memphis, TN"


@dataclass
class BatteryConfig:
    capacity_mwh: float = 3000.0
    max_discharge_mw: float = 500.0
    max_charge_mw: float = 400.0
    min_soc: float = 10.0
    max_soc: float = 95.0


@dataclass
class GridConfig:
    max_import_mw: float = 150.0
    nominal_freq_hz: float = 60.0
    max_ramp_mw_min: float = 10.0


@dataclass
class ModelConfig:
    model_type: str = "dlinear"
    num_features: int = 30
    lookback: int = 60
    horizon: int = 60
    patch_size: int = 10
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.35
    n_classes: int = 3
    use_freq: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 128
    lr: float = 1e-4
    weight_decay: float = 3e-4
    epochs: int = 80
    warmup: int = 500
    patience: int = 0
    grad_clip: float = 1.0
    under_w: float = 5.0
    over_w: float = 1.0
    spike_std: float = 1.5
    cls_w: float = 0.5
    dir_w: float = 30.0


@dataclass
class SignalConfig:
    critical_mw: float = 85.0
    warning_mw: float = 70.0


@dataclass
class SimConfig:
    num_days: int = 30
    interval_sec: int = 5
    solar_cap_mw: float = 500.0
    noise: float = 0.05
    pattern_spikes: bool = True


@dataclass
class Config:
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    sim: SimConfig = field(default_factory=SimConfig)
