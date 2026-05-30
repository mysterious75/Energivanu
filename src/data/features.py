"""
ENERGIVANU — Feature Engineering v2
v2: Lag features, cyclical time, more rolling windows, cross-feature stats.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple
from src.config import Config


BASE_COLS = [
    "gpu_power_mw","gpu_load_pct","gpu_temp_c","temp_c","humid_pct",
    "cloud_pct","solar_wm2","solar_mw","wind_ms","freq_hz","volt_pu",
    "soc_pct","batt_mw","grid_mw","hour","dow","pwr_rate","net_load",
]

class FeatureStore:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.scaler = StandardScaler()
        self.t_mean = 0.0; self.t_std = 1.0
        self.cols: list = []

    def _add_rolling(self, df):
        # Rolling stats on gpu_power_mw at 6 window sizes
        for wn, w in [("15s",3),("30s",6),("1m",12),("2m",24),("3m",36),("6m",72),("10m",120)]:
            df[f"pm_{wn}"] = df["gpu_power_mw"].rolling(w,min_periods=1).mean()
            df[f"ps_{wn}"] = df["gpu_power_mw"].rolling(w,min_periods=1).std().fillna(0)
            df[f"sw_{wn}"] = (df["gpu_power_mw"].rolling(w,min_periods=1).max()
                              - df["gpu_power_mw"].rolling(w,min_periods=1).min())
        # Rolling percentiles
        for wn, w in [("3m",36),("6m",72)]:
            df[f"pp10_{wn}"] = df["gpu_power_mw"].rolling(w,min_periods=1).quantile(0.1)
            df[f"pp90_{wn}"] = df["gpu_power_mw"].rolling(w,min_periods=1).quantile(0.9)

        # Acceleration (second derivative)
        df["accel"] = df["pwr_rate"].diff().fillna(0)/5

        # Grid stress
        df["g_stress"] = np.abs(df["freq_hz"]-60)

        # Solar availability
        df["sol_avail"] = df["solar_mw"]/(df["solar_mw"].rolling(72,min_periods=1).max()+1e-6)

        # Battery headroom
        df["batt_head"] = df["soc_pct"]/100*self.cfg.battery.capacity_mwh

        # Cross-feature rolling stats
        df["solar_rm_3m"] = df["solar_mw"].rolling(36,min_periods=1).mean()
        df["solar_rm_6m"] = df["solar_mw"].rolling(72,min_periods=1).mean()
        df["grid_rm_3m"] = df["grid_mw"].rolling(36,min_periods=1).mean()
        df["grid_rm_6m"] = df["grid_mw"].rolling(72,min_periods=1).mean()
        df["batt_rm_3m"] = df["batt_mw"].rolling(36,min_periods=1).mean()

        # Lag features (t-1, t-2, t-3 for key columns)
        for lag in [1, 2, 3]:
            df[f"pwr_lag{lag}"] = df["gpu_power_mw"].shift(lag).fillna(method="bfill")
            df[f"solar_lag{lag}"] = df["solar_mw"].shift(lag).fillna(method="bfill")
            df[f"grid_lag{lag}"] = df["grid_mw"].shift(lag).fillna(method="bfill")

        # Delta features (rate of change for non-power columns)
        df["solar_delta"] = df["solar_mw"].diff().fillna(0)/5
        df["grid_delta"] = df["grid_mw"].diff().fillna(0)/5
        df["batt_delta"] = df["batt_mw"].diff().fillna(0)/5
        df["temp_delta"] = df["temp_c"].diff().fillna(0)/5

        # Cyclical time encoding
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

        # Interaction features
        df["pwr_x_solar"] = df["gpu_power_mw"] * df["solar_mw"]
        df["pwr_x_stress"] = df["gpu_power_mw"] * df["g_stress"]

        return df

    def prepare(self, df, fit=True):
        df = self._add_rolling(df.copy())

        # Collect all feature columns (order doesn't matter, just need the set)
        rolling_cols = [c for c in df.columns if any(c.startswith(p) for p in
                        ["pm_","ps_","sw_","pp10_","pp90_","pwr_lag","solar_lag",
                         "grid_lag","solar_rm","grid_rm","batt_rm"])]
        other_cols = ["accel","g_stress","sol_avail","batt_head",
                      "solar_delta","grid_delta","batt_delta","temp_delta",
                      "hour_sin","hour_cos","dow_sin","dow_cos",
                      "pwr_x_solar","pwr_x_stress"]

        self.cols = BASE_COLS + rolling_cols + other_cols
        self.cols = list(dict.fromkeys(self.cols))

        F = np.nan_to_num(df[self.cols].values.astype(np.float32))
        T = np.nan_to_num(df["gpu_power_mw"].values.astype(np.float32))

        if fit:
            F = self.scaler.fit_transform(F)
            self.t_mean = float(T.mean())
            self.t_std = float(T.std())+1e-8
        else:
            F = self.scaler.transform(F)

        lb = self.cfg.model.lookback
        hz = self.cfg.model.horizon
        X, Y, S, D = [], [], [], []
        for i in range(lb, len(F)-hz):
            X.append(F[i-lb:i])
            future = T[i:i+hz]
            Y.append(future)
            mx = np.max(future)
            S.append(2 if mx >= self.cfg.signal.critical_mw
                     else 1 if mx >= self.cfg.signal.warning_mw else 0)
            D.append(1 if future[-1] > future[0] else 0)

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        S = np.array(S, dtype=np.int64)
        D = np.array(D, dtype=np.int64)
        print(f"  Features: {len(self.cols)} | X: {X.shape} | Y: {Y.shape}")
        print(f"  SAFE:{(S==0).sum()} PREPARE:{(S==1).sum()} CRITICAL:{(S==2).sum()}")
        print(f"  Direction: UP:{(D==1).sum()} DOWN:{(D==0).sum()}")
        return X, Y, S, D
