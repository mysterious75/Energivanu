"""
ENERGIVANU — Production Inference Engine
5-second inference cycle: Load → Predict → Signal → Log
"""

import torch, numpy as np, time, pickle
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from src.config import Config
from src.models.transformer import ColossusTransformer
from src.engine.signal import SignalEngine, Action


@dataclass
class InferenceStep:
    ts: datetime
    peak_mw: float
    signal: Action
    soc: float
    conf: float
    latency_ms: float


class InferenceEngine:
    def __init__(self, cfg: Config, model: Optional[ColossusTransformer] = None,
                 ckpt_path: Optional[str] = None, device: Optional[str] = None):
        self.cfg = cfg
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        if model is None and ckpt_path is not None:
            self.load(ckpt_path)
        elif model is not None:
            self.model = model.to(self.dev)
        self.sig_eng = SignalEngine(cfg)
        self.soc = 80.0
        self.history: List[InferenceStep] = []
        self.scaler = None
        self.t_mean = 0.0
        self.t_std = 1.0

    def load(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location=self.dev, weights_only=False)
        self.model = ColossusTransformer(self.cfg.model).to(self.dev)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        return self

    def load_scaler(self, path: str):
        with open(path, "rb") as f:
            sd = pickle.load(f)
        if "scaler" in sd:
            self.scaler = sd["scaler"]
        self.t_mean = sd.get("t_mean", 0.0)
        self.t_std = sd.get("t_std", 1.0)
        return self

    def step(self, x: np.ndarray) -> InferenceStep:
        t0 = time.perf_counter()
        x_t = torch.FloatTensor(x).unsqueeze(0).to(self.dev)
        with torch.no_grad():
            pp, ps = self.model(x_t)
        conf = float(torch.softmax(ps, dim=-1).max())
        power = pp.cpu().numpy().flatten() * self.t_std + self.t_mean
        action = self.sig_eng.generate(power, self.soc, conf)
        if action.sig == 2:
            self.soc -= 2.0
        elif action.sig == 1:
            self.soc += 0.5
        self.soc = np.clip(self.soc, 5, 100)
        lat = (time.perf_counter() - t0) * 1000
        step = InferenceStep(datetime.now(), float(np.max(power)), action, self.soc, conf, lat)
        self.history.append(step)
        return step

    def run(self, features: np.ndarray, soc_init: float = 80.0, interval: float = 5.0,
            max_steps: Optional[int] = None):
        self.soc = soc_init
        total = max_steps or len(features)
        print(f"\n{'='*70}")
        print(f"  ENERGIVANU Inference — {total} steps @ {interval}s interval")
        print(f"{'='*70}")
        print(f"  {'Step':<6} {'Signal':<12} {'Peak MW':<10} {'SOC%':<8} {'Latency':<10} {'Action'}")
        print(f"  {'─'*6} {'─'*12} {'─'*10} {'─'*8} {'─'*10} {'─'*30}")

        for i in range(total):
            if i >= len(features):
                break
            s = self.step(features[i])
            print(f"  {i+1:<6} {s.signal.name:<12} {s.peak_mw:<10.1f} "
                  f"{s.soc:<8.1f} {s.latency_ms:<10.1f} {s.signal.reason[:30]}")
            if i < total - 1:
                time.sleep(interval)
        return self.history

    def summary(self):
        if not self.history:
            return
        crit = sum(1 for s in self.history if s.signal.sig == 2)
        prep = sum(1 for s in self.history if s.signal.sig == 1)
        safe = sum(1 for s in self.history if s.signal.sig == 0)
        avg_lat = np.mean([s.latency_ms for s in self.history])
        print(f"\n  Inference Summary: {len(self.history)} steps")
        print(f"  SAFE: {safe} | PREPARE: {prep} | CRITICAL: {crit}")
        print(f"  Avg latency: {avg_lat:.2f} ms | Final SOC: {self.soc:.1f}%")
