"""
ENERGIVANU — Synthetic Data Generator
Generates realistic Colossus GPU + Weather + Grid data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict
from src.config import Config


class GPUData:
    def __init__(self, cfg: Config):
        self.c = cfg.cluster
        self.s = cfg.sim
        np.random.seed(42)

    def generate(self, n: int):
        t = np.arange(n)
        hours = t * self.s.interval_sec / 3600
        days = hours / 24

        base = 0.6 + 0.15 * np.sin(2*np.pi*hours/8) + 0.1*np.sin(2*np.pi*hours/24)
        spikes = np.zeros(n)
        idx = np.random.choice(range(100, n-100), size=n//200, replace=False)
        for i in idx:
            dur = np.random.randint(6, 60)
            spikes[i:min(i+dur, n)] = np.random.uniform(0.1, 0.3)

        util = np.clip(base + spikes, 0.2, 1.0)
        idle = self.c.num_gpus * self.c.gpu_idle_watts / 1e6
        dyn = self.c.num_gpus * (self.c.gpu_tdp_watts - self.c.gpu_idle_watts) / 1e6
        power = idle + dyn * util**1.3
        power += np.random.randn(n) * power * self.s.noise
        temp = 30 + 35*util + np.random.randn(n)*2
        return power, util*100, temp


class WeatherData:
    def __init__(self, cfg: Config):
        self.s = cfg.sim

    def generate(self, n: int) -> Dict[str, np.ndarray]:
        t = np.arange(n)
        h = t * self.s.interval_sec / 3600
        d = h / 24

        temp = 22 + 8*np.sin(2*np.pi*(h-6)/24) + 3*np.sin(2*np.pi*d/7) + np.random.randn(n)*1.5
        humid = np.clip(60 - 0.5*(temp-22) + np.random.randn(n)*5, 15, 95)
        cloud = np.clip(30 + 25*np.sin(2*np.pi*d/5) + 15*np.sin(2*np.pi*d/2.3) + np.random.randn(n)*10, 0, 100)
        solar_clear = np.maximum(0, 1000*np.sin(2*np.pi*(h-6)/24))
        solar = np.maximum(0, solar_clear*(1-cloud/100)*0.85 + np.random.randn(n)*10)
        solar_mw = np.maximum(0, solar/1000 * self.s.solar_cap_mw)
        wind = np.clip(4 + 2.5*np.sin(2*np.pi*d/3.7) + np.random.randn(n)*1.5, 0, 20)

        return {"temp": temp, "humid": humid, "cloud": cloud,
                "solar_wm2": solar, "solar_mw": solar_mw, "wind": wind}


class GridData:
    def __init__(self, cfg: Config):
        self.b = cfg.battery
        self.g = cfg.grid
        self.soc = 75.0

    def generate(self, power: np.ndarray, solar: np.ndarray):
        n = len(power)
        freq = np.zeros(n); volt = np.zeros(n); soc = np.zeros(n)
        bp = np.zeros(n); gi = np.zeros(n)
        dt = 5/3600
        self.soc = 75.0

        for i in range(n):
            net = power[i] - solar[i]
            if net > self.g.max_import_mw * 0.8:
                need = net - self.g.max_import_mw*0.7
                md = min(self.b.max_discharge_mw, self.soc/100*self.b.capacity_mwh/dt)
                d = min(need, md)
                bp[i] = -d
                self.soc -= d*dt/self.b.capacity_mwh*100
            elif net < self.g.max_import_mw*0.3 and solar[i] > 50:
                room = self.b.max_soc - self.soc
                c = min(self.b.max_charge_mw, room/100*self.b.capacity_mwh/dt)
                bp[i] = c
                self.soc += c*dt/self.b.capacity_mwh*100
            self.soc = np.clip(self.soc, self.b.min_soc, self.b.max_soc)
            soc[i] = self.soc
            stress = max(0, (net-self.g.max_import_mw*0.7)/(self.g.max_import_mw*0.3))
            freq[i] = 60 + np.random.randn()*0.02*(1+3*stress)
            volt[i] = 1.0 - stress*0.05 + np.random.randn()*0.005
            gi[i] = max(0, net+bp[i])

        return {"freq": freq, "volt": volt, "soc": soc, "bp": bp, "gi": gi}


def generate_dataset(cfg: Config = None) -> pd.DataFrame:
    cfg = cfg or Config()
    n = int(cfg.sim.num_days * 24 * 3600 / cfg.sim.interval_sec)

    print(f"  Generating {cfg.sim.num_days} days ({n:,} points)...")

    gpu = GPUData(cfg)
    pwr, load, temp = gpu.generate(n)

    wth = WeatherData(cfg)
    w = wth.generate(n)

    grd = GridData(cfg)
    g = grd.generate(pwr, w["solar_mw"])

    ts = [datetime(2026,1,1) + timedelta(seconds=i*5) for i in range(n)]

    df = pd.DataFrame({
        "timestamp": ts,
        "gpu_power_mw": pwr, "gpu_load_pct": load, "gpu_temp_c": temp,
        "temp_c": w["temp"], "humid_pct": w["humid"], "cloud_pct": w["cloud"],
        "solar_wm2": w["solar_wm2"], "solar_mw": w["solar_mw"], "wind_ms": w["wind"],
        "freq_hz": g["freq"], "volt_pu": g["volt"],
        "soc_pct": g["soc"], "batt_mw": g["bp"], "grid_mw": g["gi"],
    })
    df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute/60
    df["dow"] = df["timestamp"].dt.dayofweek
    df["pwr_rate"] = df["gpu_power_mw"].diff().fillna(0)/5
    df["net_load"] = df["gpu_power_mw"] - df["solar_mw"]

    print(f"  Done: {len(df)} rows, {len(df.columns)} cols")
    return df
