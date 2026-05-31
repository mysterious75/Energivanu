"""
ENERGIVANU — Feature Engineering v3 (reverted to proven 34-feature config)
stride=1, 34 features: MAE 3.52 MW baseline.
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
        for wn, w in [("30s",6),("1m",12),("3m",36),("6m",72)]:
            df[f"pm_{wn}"] = df["gpu_power_mw"].rolling(w,min_periods=1).mean()
            df[f"ps_{wn}"] = df["gpu_power_mw"].rolling(w,min_periods=1).std().fillna(0)
            df[f"sw_{wn}"] = (df["gpu_power_mw"].rolling(w,min_periods=1).max()
                              - df["gpu_power_mw"].rolling(w,min_periods=1).min())
        df["accel"] = df["pwr_rate"].diff().fillna(0)/5
        df["g_stress"] = np.abs(df["freq_hz"]-60)
        df["sol_avail"] = df["solar_mw"]/(df["solar_mw"].rolling(72,min_periods=1).max()+1e-6)
        df["batt_head"] = df["soc_pct"]/100*self.cfg.battery.capacity_mwh
        return df

    def prepare(self, df, fit=True, stride=1):
        df = self._add_rolling(df.copy())
        self.cols = BASE_COLS + [c for c in df.columns
                                 if c.startswith("pm_") or c.startswith("ps_")
                                 or c.startswith("sw_") or c.startswith("accel")
                                 or c.startswith("g_") or c.startswith("sol_")
                                 or c.startswith("batt_h")]
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
        nf = F.shape[1]
        N = (len(F) - lb - hz) // stride

        X = np.zeros((N, lb, nf), dtype=np.float32)
        Y = np.zeros((N, hz), dtype=np.float32)
        S = np.zeros(N, dtype=np.int64)
        D = np.zeros(N, dtype=np.int64)

        for i in range(N):
            idx = lb + i * stride
            X[i] = F[idx-lb:idx]
            future = T[idx:idx+hz]
            Y[i] = future
            mx = future.max()
            S[i] = 2 if mx >= self.cfg.signal.critical_mw else (1 if mx >= self.cfg.signal.warning_mw else 0)
            D[i] = 1 if future[-1] > future[0] else 0

        print(f"  Features: {len(self.cols)} | X: {X.shape} | Y: {Y.shape} | stride: {stride}")
        print(f"  SAFE:{(S==0).sum()} PREPARE:{(S==1).sum()} CRITICAL:{(S==2).sum()}")
        print(f"  Direction: UP:{(D==1).sum()} DOWN:{(D==0).sum()}")
        return X, Y, S, D
