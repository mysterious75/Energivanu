#!/usr/bin/env python3
"""
ENERGIVANU — One-Click Project Builder
Run on your laptop:  python setup.py
Then upload energivanu_colab.ipynb to Google Colab
"""

from pathlib import Path

FILES = {}

# ── requirements.txt ──
FILES["requirements.txt"] = """torch>=2.1.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
pyarrow>=12.0
tqdm>=4.65.0
matplotlib>=3.7.0
"""

# ── src/__init__.py ──
FILES["src/__init__.py"] = ""

# ── src/config.py ──
FILES["src/config.py"] = '''"""
ENERGIVANU — Central Configuration
"""

from dataclasses import dataclass, field


@dataclass
class ClusterConfig:
    num_gpus: int = 100_000
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
    num_features: int = 30
    lookback: int = 120
    horizon: int = 120
    patch_size: int = 10
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    n_classes: int = 3
    use_freq: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 50
    warmup: int = 500
    patience: int = 10
    grad_clip: float = 1.0
    under_w: float = 5.0
    over_w: float = 1.0
    spike_std: float = 1.5
    cls_w: float = 0.5


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


@dataclass
class Config:
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    sim: SimConfig = field(default_factory=SimConfig)
'''

# ── src/data/__init__.py ──
FILES["src/data/__init__.py"] = ""

# ── src/data/generator.py ──
FILES["src/data/generator.py"] = '''"""
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
'''

# ── src/data/features.py ──
FILES["src/data/features.py"] = '''"""
ENERGIVANU — Feature Engineering
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

    def prepare(self, df, fit=True):
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
        X, Y, S = [], [], []
        for i in range(lb, len(F)-hz):
            X.append(F[i-lb:i])
            future = T[i:i+hz]
            Y.append(future)
            mx = np.max(future)
            S.append(2 if mx >= self.cfg.signal.critical_mw
                     else 1 if mx >= self.cfg.signal.warning_mw else 0)

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        S = np.array(S, dtype=np.int64)
        print(f"  Features: {len(self.cols)} | X: {X.shape} | Y: {Y.shape}")
        print(f"  SAFE:{(S==0).sum()} PREPARE:{(S==1).sum()} CRITICAL:{(S==2).sum()}")
        return X, Y, S
'''

# ── src/models/__init__.py ──
FILES["src/models/__init__.py"] = ""

# ── src/models/transformer.py ──
FILES["src/models/transformer.py"] = '''"""
ENERGIVANU — Time-Series Transformer
Dual branch: Time (Transformer) + Frequency (FFT) → Power + Signal
"""

import torch, torch.nn as nn, math
from typing import Tuple
from src.config import ModelConfig


class PatchEmbed(nn.Module):
    def __init__(self, nf, dm, ps=10):
        super().__init__()
        self.ps = ps
        self.proj = nn.Linear(nf*ps, dm)
        self.norm = nn.LayerNorm(dm)
    def forward(self, x):
        B,T,F = x.shape; np_ = T//self.ps
        x = x[:,:np_*self.ps,:].reshape(B,np_,F*self.ps)
        return self.norm(self.proj(x))


class GRN(nn.Module):
    def __init__(self, di, dh, do, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(di,dh); self.fc2 = nn.Linear(dh,do)
        self.gate = nn.Linear(di,do)
        self.skip = nn.Linear(di,do) if di!=do else nn.Identity()
        self.norm = nn.LayerNorm(do); self.d = nn.Dropout(drop)
    def forward(self, x):
        r = self.skip(x)
        h = self.d(torch.elu(self.fc1(x)))
        h = self.d(self.fc2(h))
        g = torch.sigmoid(self.gate(x))
        return self.norm(r + g*h)


class FreqBranch(nn.Module):
    def __init__(self, sl, dm):
        super().__init__()
        fb = sl//2+1
        self.p = nn.Sequential(nn.Linear(fb,dm), nn.GELU(), nn.Dropout(0.1))
    def forward(self, x):
        return self.p(torch.abs(torch.fft.rfft(x, dim=-1)))


class ColossusTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        np_ = cfg.lookback // cfg.patch_size

        self.patch = PatchEmbed(cfg.num_features, cfg.d_model, cfg.patch_size)
        self.pos = nn.Parameter(torch.randn(1, np_, cfg.d_model)*0.02)
        el = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout,
            batch_first=True, activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(el, cfg.n_layers)

        self.freq = FreqBranch(cfg.lookback, cfg.d_model) if cfg.use_freq else None
        self.fgate = nn.Linear(cfg.d_model*2, cfg.d_model) if cfg.use_freq else None

        self.grn = GRN(cfg.d_model, cfg.d_ff, cfg.d_model, cfg.dropout)
        self.phead = nn.Sequential(
            nn.Linear(cfg.d_model,cfg.d_ff), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff,cfg.d_ff//2), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff//2, cfg.horizon))
        self.shead = nn.Sequential(
            nn.Linear(cfg.d_model,cfg.d_ff//2), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff//2, cfg.n_classes))

        self._init()
        print(f"  Params: {sum(p.numel() for p in self.parameters()):,}")

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.patch(x) + self.pos
        h = self.enc(h).mean(1)
        if self.freq and self.fgate:
            hf = self.freq(x[:,:,0])
            g = torch.sigmoid(self.fgate(torch.cat([h,hf],-1)))
            h = g*h + (1-g)*hf
        h = self.grn(h)
        return self.phead(h), self.shead(h)
'''

# ── src/models/losses.py ──
FILES["src/models/losses.py"] = '''"""
ENERGIVANU — Asymmetric Spike Loss
Under-predicting spikes = 5x penalty (grid damage risk)
"""

import torch, torch.nn as nn, torch.nn.functional as F
from typing import Tuple, Dict


class SpikeLoss(nn.Module):
    def __init__(self, uw=5.0, ow=1.0, ss=1.5, cw=0.5):
        super().__init__()
        self.uw=uw; self.ow=ow; self.ss=ss; self.cw=cw

    def forward(self, pp, tp, ps, ts) -> Tuple[torch.Tensor, Dict]:
        err = tp - pp
        th = tp.mean() + self.ss*tp.std()
        sp = (tp>th).float()
        w = torch.where(err>0, self.uw+self.uw*sp, torch.tensor(self.ow,device=err.device))
        pl = (w*err.pow(2)).mean()

        cw = torch.tensor([1.,2.,5.], device=ps.device)
        sl = F.cross_entropy(ps, ts, weight=cw)

        total = pl + self.cw*sl
        with torch.no_grad():
            mae = err.abs().mean()
            pd = pp[:,1:]-pp[:,:-1]; td = tp[:,1:]-tp[:,:-1]
            da = (pd.sign()==td.sign()).float().mean()
            sa = (ps.argmax(-1)==ts).float().mean()
        return total, {"pl":pl.item(),"sl":sl.item(),"loss":total.item(),
                        "mae":mae.item(),"da":da.item(),"sa":sa.item()}
'''

# ── src/engine/__init__.py ──
FILES["src/engine/__init__.py"] = ""

# ── src/engine/trainer.py ──
FILES["src/engine/trainer.py"] = '''"""
ENERGIVANU — Trainer with warmup + cosine LR + early stopping
"""

import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np, time
from pathlib import Path
from src.config import Config
from src.models.transformer import ColossusTransformer
from src.models.losses import SpikeLoss


class Trainer:
    def __init__(self, model, cfg: Config):
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.dev)
        self.cfg = cfg
        self.loss_fn = SpikeLoss(cfg.train.under_w, cfg.train.over_w,
                                  cfg.train.spike_std, cfg.train.cls_w)
        self.opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                      weight_decay=cfg.train.weight_decay, betas=(0.9,0.98))
        self.step = 0
        self.hist = {"tl":[],"vl":[],"vm":[],"vs":[]}

    def _lr(self, cur, total):
        c = self.cfg.train
        if cur < c.warmup: return c.lr * cur / max(c.warmup,1)
        p = (cur-c.warmup)/max(total-c.warmup,1)
        return c.lr * 0.5 * (1+np.cos(np.pi*p))

    def _epoch(self, dl, train, total=0):
        self.model.train() if train else self.model.eval()
        s = {k:0. for k in ["pl","sl","loss","mae","da","sa"]}; n=0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for x,yp,ys in dl:
                x,yp,ys = x.to(self.dev),yp.to(self.dev),ys.to(self.dev)
                if train:
                    self.step+=1
                    for g in self.opt.param_groups:
                        g["lr"]=self._lr(self.step,total)
                    self.opt.zero_grad()
                pp,ps = self.model(x)
                l,m = self.loss_fn(pp,yp,ps,ys)
                if train:
                    l.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
                    self.opt.step()
                for k in s: s[k]+=m[k]
                n+=1
        return {k:v/max(n,1) for k,v in s.items()}

    def fit(self, Xtr,Ytr,Str,Xvl,Yvl,Svl):
        tc = self.cfg.train
        tds = TensorDataset(torch.FloatTensor(Xtr),torch.FloatTensor(Ytr),torch.LongTensor(Str))
        vds = TensorDataset(torch.FloatTensor(Xvl),torch.FloatTensor(Yvl),torch.LongTensor(Svl))
        tdl = DataLoader(tds, tc.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        vdl = DataLoader(vds, tc.batch_size, shuffle=False, num_workers=0, pin_memory=True)

        total = len(tdl)*tc.epochs; best=float("inf"); wait=0
        Path("checkpoints").mkdir(exist_ok=True)

        print(f"\\n  Device: {self.dev} | Train: {len(tds):,} | Val: {len(vds):,}")
        print(f"  Batch: {tc.batch_size} | Epochs: {tc.epochs}\\n")

        for ep in range(1,tc.epochs+1):
            t0=time.time()
            tm=self._epoch(tdl,True,total)
            vm=self._epoch(vdl,False)
            dt=time.time()-t0

            self.hist["tl"].append(tm["loss"])
            self.hist["vl"].append(vm["loss"])
            self.hist["vm"].append(vm["mae"])
            self.hist["vs"].append(vm["sa"])

            if ep%5==0 or ep==1:
                print(f"  Ep {ep:3d}/{tc.epochs} | TL:{tm['loss']:.4f} VL:{vm['loss']:.4f} "
                      f"| MAE:{vm['mae']:.2f}MW SigAcc:{vm['sa']:.3f} "
                      f"DirAcc:{vm['da']:.3f} LR:{self.opt.param_groups[0]['lr']:.2e} {dt:.1f}s")

            if vm["loss"]<best:
                best=vm["loss"]; wait=0
                torch.save({"ep":ep,"model":self.model.state_dict(),
                            "opt":self.opt.state_dict(),"vl":best},
                           "checkpoints/best.pt")
            else:
                wait+=1
                if wait>=tc.patience:
                    print(f"\\n  Early stop at {ep}"); break

        print(f"\\n  Best val loss: {best:.4f}")
        return self.hist
'''

# ── src/engine/signal.py ──
FILES["src/engine/signal.py"] = '''"""
ENERGIVANU — Battery Signal Engine
SAFE / PREPARE / CRITICAL → Tesla Megapack commands
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from src.config import Config


@dataclass
class Action:
    ts: datetime; sig: int; name: str; peak_mw: float; ttp: int
    soc: float; dis_mw: float; thr: float; conf: float; reason: str


class SignalEngine:
    N = {0:"SAFE",1:"PREPARE",2:"CRITICAL"}
    def __init__(self, cfg: Config):
        self.s = cfg.signal; self.b = cfg.battery; self.hist=[]

    def generate(self, pred, soc, mc=0.8, ts=None):
        ts = ts or datetime.now()
        peak=float(np.max(pred)); avg=float(np.mean(pred))
        ttp=int(np.argmax(pred))*5

        self.hist.append(peak)
        if len(self.hist)>100: self.hist=self.hist[-100:]
        stab = max(0,1-np.std(self.hist[-10:])/20) if len(self.hist)>5 else 0.5
        conf = 0.5*mc + 0.5*stab

        avail = soc/100*self.b.capacity_mwh
        md = min(self.b.max_discharge_mw, avail/0.25)

        if peak>=self.s.critical_mw:
            df=peak-self.s.critical_mw; d=min(df*1.2,md)
            t=min(50,max(0,(df-d)/max(peak,1)*100))
            r=f"CRITICAL: {peak:.1f}MW in {ttp}s. Discharge {d:.1f}MW. SOC:{soc:.0f}%."
            sig=2
        elif peak>=self.s.warning_mw:
            d,t=0,0
            r=f"WARNING: {peak:.1f}MW in {ttp}s. Pre-charge. SOC:{soc:.0f}%."
            sig=1
        else:
            d,t=0,0
            r=f"SAFE: {avg:.1f}MW avg. Peak {peak:.1f}MW. SOC:{soc:.0f}%."
            sig=0

        return Action(ts,sig,self.N[sig],peak,ttp,soc,d,t,conf,r)
'''

# ── src/co_optimize/__init__.py ──
FILES["src/co_optimize/__init__.py"] = ""

# ── src/co_optimize/scheduler.py ──
FILES["src/co_optimize/scheduler.py"] = '''"""
ENERGIVANU — Workload + Battery Co-Optimizer
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict
from enum import Enum
from src.config import Config


class Pri(Enum):
    CRIT=1; HIGH=2; MED=3; LOW=4

@dataclass
class Job:
    id: str; pri: Pri; pw: float; dur: float; dl: datetime
    throttle: bool=True; fmax: float=1800

@dataclass
class Result:
    sched: List[Dict]; batt: List[Dict]; dvfs: Dict
    stress: float; savings: float


class CoOptimizer:
    def __init__(self, cfg: Config):
        self.b = cfg.battery; self.g = cfg.grid

    def optimize(self, jobs, pred, soc, now):
        sc, bp, dv = [],[],[]
        sorted_j = sorted(jobs, key=lambda j: j.pri.value)
        hr = self.g.max_import_mw - float(np.max(pred))
        ab = min(self.b.max_discharge_mw, soc/100*self.b.capacity_mwh/0.25)
        sv = 0.0

        for j in sorted_j:
            if hr>=j.pw:
                sc.append({"job":j.id,"act":"RUN","freq":j.fmax})
                hr-=j.pw
            elif hr+ab>=j.pw:
                d=j.pw-hr; ab-=d
                sc.append({"job":j.id,"act":"RUN+BATT","freq":j.fmax})
                bp.append({"dis":d}); hr=0
            elif j.throttle and j.pri.value>=3:
                r=max(0.5,(hr/max(j.pw,1))**(1/3))
                nf=j.fmax*r; ap=j.pw*r**3
                sv+=j.pw-ap; dv[j.id]=nf
                sc.append({"job":j.id,"act":"THROTTLED","freq":nf,"save":(1-r**3)*100})
                hr-=ap
            elif j.pri.value>=3:
                dl=min((j.dl-now).total_seconds()/3600,24)
                sc.append({"job":j.id,"act":"DELAY","h":dl})
            else:
                sc.append({"job":j.id,"act":"EMERGENCY"})

        st=max(0,min(1,(float(np.max(pred))-self.g.max_import_mw*0.8)/(self.g.max_import_mw*0.2)))
        tp=sum(j.pw for j in jobs)
        return Result(sc,bp,dv,st,sv/max(tp,1)*100)
'''

# ── README.md ──
FILES["README.md"] = '''# ENERGIVANU

**AI-Powered Energy Management for GPU Supercomputers**

Predicts power demand spikes 10 minutes ahead and signals Tesla Megapack batteries before grid instability.

## Quick Start

```bash
# 1. Generate project
python setup.py

# 2. Upload energivanu_colab.ipynb to Google Colab

# 3. Run all cells in Colab (training happens on free T4 GPU)
```

## Architecture

```
GPU Telemetry + Weather + Grid → Features → Transformer → Power + Signal → Battery
```

## Project Name: ENERGIVANU
'''

# ── .gitignore ──
FILES[".gitignore"] = """__pycache__/
*.pyc
data/
checkpoints/
.env
.ipynb_checkpoints/
"""


# ============================================================
# COLAB NOTEBOOK CONTENT (as Python cells)
# ============================================================

FILES["energivanu_colab.py"] = r'''"""
=============================================================
  ENERGIVANU — Google Colab Training Script
  Paste each CELL section into separate Colab cells
=============================================================
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 1: Mount Google Drive + Install Dependencies
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from google.colab import drive
drive.mount('/content/drive')

import os
PROJECT_DIR = "/content/drive/MyDrive/energivanu"
os.makedirs(f"{PROJECT_DIR}/data", exist_ok=True)
os.makedirs(f"{PROJECT_DIR}/checkpoints", exist_ok=True)
os.makedirs(f"{PROJECT_DIR}/results", exist_ok=True)

!pip install -q torch numpy pandas scikit-learn pyarrow tqdm matplotlib

import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB" if torch.cuda.is_available() else "")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 2: Clone or Upload Project Code
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Option A: If you pushed to GitHub already:
# !git clone https://github.com/YOUR_USERNAME/energivanu.git
# %cd energivanu

# Option B: Upload src/ folder manually to Colab
# Just drag-drop the src/ folder into Colab file panel

# Option C: Create everything inline (paste setup.py content and run)

print("Project code ready!")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 3: Generate Data (saves to Google Drive)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import sys, os
sys.path.insert(0, "/content/energivanu")  # adjust path

from src.config import Config
from src.data.generator import generate_dataset

cfg = Config()
cfg.sim.num_days = 30  # 30 days = ~500K data points

df = generate_dataset(cfg)

# Save to Google Drive
df.to_parquet(f"{PROJECT_DIR}/data/colossus_30d.parquet", index=False)
print(f"Saved to Drive: {df.shape}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 4: Feature Engineering
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import pandas as pd
import numpy as np
import pickle

df = pd.read_parquet(f"{PROJECT_DIR}/data/colossus_30d.parquet")

from src.data.features import FeatureStore

fs = FeatureStore(cfg)
X, Y, S = fs.prepare(df, fit=True)

# Save features to Drive (so we don't regenerate every session)
np.save(f"{PROJECT_DIR}/data/X.npy", X)
np.save(f"{PROJECT_DIR}/data/Y.npy", Y)
np.save(f"{PROJECT_DIR}/data/S.npy", S)

# Save scaler
with open(f"{PROJECT_DIR}/data/scaler.pkl", "wb") as f:
    pickle.dump({"scaler": fs.scaler, "t_mean": fs.t_mean,
                 "t_std": fs.t_std, "cols": fs.cols}, f)

print(f"Features saved to Drive: X={X.shape}, Y={Y.shape}, S={S.shape}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 5: Train Transformer (this runs on Colab T4 GPU)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import numpy as np
import pickle

# Load from Drive (fast — no regeneration needed)
X = np.load(f"{PROJECT_DIR}/data/X.npy")
Y = np.load(f"{PROJECT_DIR}/data/Y.npy")
S = np.load(f"{PROJECT_DIR}/data/S.npy")

from src.config import Config
from src.models.transformer import ColossusTransformer
from src.engine.trainer import Trainer

cfg = Config()
cfg.model.num_features = X.shape[2]

split = int(0.8 * len(X))
print(f"Train: {split:,} | Val: {len(X)-split:,}")

model = ColossusTransformer(cfg.model)
trainer = Trainer(model, cfg)

# CHANGE EPOCHS HERE if needed (50 is good for first run)
cfg.train.epochs = 50

history = trainer.fit(
    X[:split], Y[:split], S[:split],
    X[split:], Y[split:], S[split:],
)

# Model is auto-saved to checkpoints/best.pt
# Copy to Drive for persistence
import shutil
shutil.copy("checkpoints/best.pt", f"{PROJECT_DIR}/checkpoints/best.pt")
print(f"Model saved to Drive!")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 6: Plot Training Curves
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(history["tl"], label="Train")
axes[0].plot(history["vl"], label="Val")
axes[0].set_title("Loss"); axes[0].legend()

axes[1].plot(history["vm"])
axes[1].set_title("Val MAE (MW)")

axes[2].plot(history["vs"])
axes[2].set_title("Val Signal Accuracy")

plt.tight_layout()
plt.savefig(f"{PROJECT_DIR}/results/training_curves.png", dpi=150)
plt.show()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 7: Inference Simulation (Battery Signals)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import torch
import numpy as np
import pickle
from src.config import Config
from src.models.transformer import ColossusTransformer
from src.engine.signal import SignalEngine

cfg = Config()
cfg.model.num_features = X.shape[2]

# Load model from Drive
model = ColossusTransformer(cfg.model)
ckpt = torch.load(f"{PROJECT_DIR}/checkpoints/best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()

# Load scaler
with open(f"{PROJECT_DIR}/data/scaler.pkl", "rb") as f:
    sd = pickle.load(f)

sig_eng = SignalEngine(cfg)
battery_soc = 80.0

print(f"\n{'Step':<6} {'Signal':<12} {'Peak MW':<10} {'TTP':<8} {'SOC%':<8} {'Action'}")
print(f"{'─'*6} {'─'*12} {'─'*10} {'─'*8} {'─'*8} {'─'*40}")

for step in range(15):
    idx = split + step * 10
    if idx + cfg.model.lookback >= len(X):
        break

    x = torch.FloatTensor(X[idx:idx+1])
    with torch.no_grad():
        pp, ps = model(x)

    power = pp.numpy().flatten() * sd["t_std"] + sd["t_mean"]
    conf = float(torch.softmax(ps, dim=-1).max())

    action = sig_eng.generate(power, battery_soc, conf)

    if action.sig == 2: battery_soc -= 2.0
    elif action.sig == 1: battery_soc += 0.5
    battery_soc = np.clip(battery_soc, 5, 100)

    print(f"{step+1:<6} {action.name:<12} {action.peak_mw:<10.1f} "
          f"{action.ttp:<8} {battery_soc:<8.1f} {action.reason[:45]}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 8: Co-Optimization Demo
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from datetime import datetime
from src.co_optimize.scheduler import CoOptimizer, Job, Pri

co = CoOptimizer(cfg)
jobs = [
    Job("train_llm", Pri.HIGH, 25.0, 4.0, datetime(2026,1,2)),
    Job("inference", Pri.CRIT, 5.0, 1.0, datetime(2026,1,1,1)),
    Job("batch", Pri.LOW, 15.0, 8.0, datetime(2026,1,3)),
    Job("analysis", Pri.MED, 10.0, 6.0, datetime(2026,1,2)),
]

pred = np.random.uniform(40, 90, 120)
r = co.optimize(jobs, pred, battery_soc, datetime(2026,1,1))

print(f"\nGrid stress: {r.stress:.3f}")
print(f"Savings: {r.savings:.1f}%")
print(f"DVFS: {r.dvfs}")
print(f"\nSchedule:")
for s in r.sched:
    print(f"  {s}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 9: Summary — Everything saved to Drive
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import os

print("=" * 60)
print("  ENERGIVANU — Training Complete!")
print("=" * 60)
print(f"\n  Files in Google Drive ({PROJECT_DIR}):")
for root, dirs, files in os.walk(PROJECT_DIR):
    level = root.replace(PROJECT_DIR, "").count(os.sep)
    indent = "  " * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = "  " * (level + 1)
    for f in files:
        size = os.path.getsize(os.path.join(root, f))
        print(f"{subindent}{f}  ({size/1e6:.1f} MB)")

print("\n  Next steps:")
print("  1. Download best.pt from Drive")
print("  2. Use InferenceEngine for real-time predictions")
print("  3. Deploy on edge server with < 500ms latency")
'''


# ============================================================
# BUILD
# ============================================================

def build():
    root = Path(".")
    for d in ["src","src/data","src/models","src/engine","src/co_optimize","data","checkpoints","tests"]:
        (root/d).mkdir(parents=True, exist_ok=True)

    for fp, content in FILES.items():
        p = root/fp
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content.strip()+"\n", encoding="utf-8")
        print(f"  [OK] {fp}")

    print(f"\n{'='*50}")
    print(f"  ENERGIVANU — Project Created!")
    print(f"  Files: {len(FILES)}")
    print(f"{'='*50}")
    print(f"\n  WHAT TO DO NOW:")
    print(f"  ────────────────")
    print(f"  1. Upload energivanu_colab.py to Google Colab")
    print(f"  2. Split it into cells (each # CELL section = 1 cell)")
    print(f"  3. Upload src/ folder to Colab (drag-drop)")
    print(f"  4. Run all cells — training on free T4 GPU!")
    print(f"  5. Model saves automatically to Google Drive")
    print(f"\n  Google Drive path: /MyDrive/energivanu/")
    print(f"    data/           ← training data")
    print(f"    checkpoints/    ← trained model")
    print(f"    results/        ← plots and logs")


if __name__ == "__main__":
    build()
