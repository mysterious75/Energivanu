# %% [markdown]
# # ⚡ Energivanu — Full Alibaba Training (200 epochs) + CVXPY MPC
# 50 lakh rows → TCN+Attention → MPC Controller

# %% Cell 1: Setup + PyTorch P100 fix
import os, sys, subprocess, time, json, warnings
import numpy as np
import pandas as pd

print("Installing PyTorch cu118 (P100 compatible)...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "torch", "--index-url", "https://download.pytorch.org/whl/cu118"], check=True)
print("Installing CVXPY...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "cvxpy"], check=True)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cvxpy as cp
from datetime import datetime

warnings.filterwarnings("ignore")
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    try:
        _t = torch.zeros(2, 2, device="cuda"); del _t; torch.cuda.synchronize()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except Exception as e:
        print(f"CUDA issue: {e}, CPU fallback")
        DEVICE = "cpu"
print(f"Device: {DEVICE}")

# %% Cell 2: Load Full Alibaba Data
print("="*60)
print("LOADING FULL ALIBABA DATA (50 lakh rows)")
print("="*60)

# Find data — prefer .csv.gz (full 50 lakh) over .csv (subset)
data_path = None
all_csvs = []
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f.endswith(".csv.gz"):
            data_path = os.path.join(root, f)
            break
        if f.endswith(".csv"):
            all_csvs.append(os.path.join(root, f))
    if data_path:
        break
if data_path is None and all_csvs:
    data_path = max(all_csvs, key=os.path.getsize)

if data_path:
    print(f"Loading: {data_path}")
    if data_path.endswith(".gz"):
        df = pd.read_csv(data_path, compression="gzip")
    else:
        df = pd.read_csv(data_path)
    print(f"Shape: {df.shape}")
else:
    raise RuntimeError("No Alibaba data found! Check Kaggle inputs.")

# Check if already has 15 features
FEATURE_COLS = [
    "facility_mw", "power_roc", "power_roc2", "power_roll_mean", "power_roll_std",
    "gpu_avg_power_norm", "gpu_max_power_norm", "gpu_avg_temp_norm", "gpu_max_temp_norm",
    "gpu_avg_util_norm", "gpu_avg_mem_util_norm", "cpu_util_est_norm",
    "hour_sin", "hour_cos", "is_allreduce",
]

if all(c in df.columns for c in FEATURE_COLS):
    print("✅ Already has 15 features")
    features = df[FEATURE_COLS].values.astype(np.float32)
else:
    print("Computing features...")
    col_map = {"cpu_usage":"cpu_util","gpu_wrk_util":"gpu_util","avg_mem":"mem_util",
               "avg_gpu_wrk_mem":"gpu_mem_util","machine_cpu_usr":"cpu_util","machine_gpu":"gpu_util"}
    df = df.rename(columns=col_map)
    if "gpu_util" not in df.columns:
        raise RuntimeError(f"No gpu_util! Cols: {list(df.columns)[:8]}")
    
    gpu_util = df["gpu_util"].fillna(0).clip(0, 100).values.astype(np.float32)
    n = len(gpu_util)
    GPU_TDP = 700.0; FACILITY_GPUS = 200000
    single_w = 70 + (GPU_TDP - 70) * (gpu_util / 100.0)
    mw = (single_w * FACILITY_GPUS / 1e6).astype(np.float32)
    rm = pd.Series(mw).rolling(30, min_periods=1).mean().values.astype(np.float32)
    rs = pd.Series(mw).rolling(30, min_periods=1).std().fillna(0).values.astype(np.float32)
    roc = np.diff(mw, prepend=mw[0]); roc2 = np.diff(roc, prepend=roc[0])
    t_idx = np.arange(n)
    hsin = np.sin(2*np.pi*(t_idx%3600)/3600).astype(np.float32)
    hcos = np.cos(2*np.pi*(t_idx%3600)/3600).astype(np.float32)
    temp = (0.4+0.4*gpu_util/100).clip(0,1).astype(np.float32)
    mem = (gpu_util*0.8).clip(0,100).astype(np.float32)
    cpu_u = np.full(n, 50.0, dtype=np.float32)
    is_ar = ((gpu_util > 80) & (mem < 30)).astype(np.float32)
    features = np.column_stack([mw,roc,roc2,rm,rs,gpu_util/100,gpu_util/100,temp,temp,
                                gpu_util/100,mem/100,cpu_u/100,hsin,hcos,is_ar])

features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
n = len(features)
del df  # Free memory
print(f"Features: {features.shape}")
print(f"Power: {features[:,0].min():.1f} - {features[:,0].max():.1f} MW")

# %% Cell 3: Sequences
SEQ_LEN = 30; PH = 10; STRIDE = 10; BATCH = 512

pw = features[:, 0]
pc = np.diff(pw, prepend=pw[0])
sig = np.zeros(n, dtype=np.int64)
sig[pc > 0.5] = 1; sig[pc < -0.5] = 2

X, Yp, Ys = [], [], []
for i in range(0, n - SEQ_LEN - PH, STRIDE):
    X.append(features[i:i+SEQ_LEN])
    Yp.append(pw[i+SEQ_LEN:i+SEQ_LEN+PH])
    Ys.append(sig[i+SEQ_LEN])

X = np.array(X, dtype=np.float32)
Yp = np.array(Yp, dtype=np.float32)
Ys = np.array(Ys, dtype=np.int64)
del features  # Free memory

ns = len(X); idx = np.random.permutation(ns); sp = int(ns*0.85)
print(f"Sequences: {ns} | Train: {sp} | Val: {ns-sp}")

class DS(Dataset):
    def __init__(s, x, yp, ys):
        s.x = torch.tensor(x); s.yp = torch.tensor(yp); s.ys = torch.tensor(ys)
    def __len__(s): return len(s.x)
    def __getitem__(s, i): return s.x[i], s.yp[i], s.ys[i]

tdl = DataLoader(DS(X[idx[:sp]], Yp[idx[:sp]], Ys[idx[:sp]]), BATCH, shuffle=True, pin_memory=True)
vdl = DataLoader(DS(X[idx[sp:]], Yp[idx[sp:]], Ys[idx[sp:]]), BATCH, pin_memory=True)

# %% Cell 4: Model
class TB(nn.Module):
    def __init__(s, ic, oc, k, d, dr=0.1):
        super().__init__()
        p = (k-1)*d
        s.c1 = nn.Conv1d(ic, oc, k, padding=p, dilation=d)
        s.c2 = nn.Conv1d(oc, oc, k, padding=p, dilation=d)
        s.n1 = nn.LayerNorm(oc); s.n2 = nn.LayerNorm(oc)
        s.d1 = nn.Dropout(dr); s.d2 = nn.Dropout(dr)
        s.r = nn.Conv1d(ic, oc, 1) if ic != oc else nn.Identity()
    def forward(s, x):
        res = s.r(x)
        o = s.d1(torch.relu(s.n1(s.c1(x)[:,:,:x.size(2)].transpose(1,2)).transpose(1,2)))
        o = s.d2(torch.relu(s.n2(s.c2(o)[:,:,:x.size(2)].transpose(1,2)).transpose(1,2)))
        return torch.relu(o + res)

class PEB(nn.Module):
    def __init__(s, nf=15, sl=30, ph=10):
        super().__init__()
        s.pn = nn.LayerNorm(min(7,nf)); s.tn = nn.LayerNorm(min(7,max(0,nf-7)))
        s.qn = nn.LayerNorm(max(0,nf-14))
        s._p = min(7,nf); s._t = min(7,max(0,nf-7)); s._q = max(0,nf-14)
        s.proj = nn.Linear(nf, 128)
        s.tcn = nn.Sequential(TB(128,32,5,1), TB(32,64,3,2), TB(64,128,3,4))
        s.attn = nn.MultiheadAttention(128, 8, dropout=0.1, batch_first=True)
        s.an = nn.LayerNorm(128); s.lw = nn.Linear(128, 1)
        s.ph = nn.Sequential(nn.Linear(128,256), nn.GELU(), nn.Dropout(0.1),
                             nn.Linear(256,128), nn.GELU(), nn.Dropout(0.05), nn.Linear(128,ph))
        s.sh = nn.Sequential(nn.Linear(128+ph,256), nn.GELU(), nn.Dropout(0.1),
                             nn.Linear(256,128), nn.GELU(), nn.Dropout(0.05), nn.Linear(128,3))
    def forward(s, x):
        xn = torch.zeros_like(x)
        if s._p: xn[:,:,:s._p] = s.pn(x[:,:,:s._p])
        if s._t: xn[:,:,s._p:s._p+s._t] = s.tn(x[:,:,s._p:s._p+s._t])
        if s._q: xn[:,:,s._p+s._t:] = s.qn(x[:,:,s._p+s._t:])
        h = s.tcn(s.proj(xn).transpose(1,2)).transpose(1,2)
        h, _ = s.attn(h, h, h); h = s.an(h)
        l, m = h[:,-1,:], h.mean(1)
        a = torch.sigmoid(s.lw(l)); g = a*l + (1-a)*m
        p = s.ph(g)
        return p, s.sh(torch.cat([g, p], 1))

model = PEB(15, SEQ_LEN, PH).to(DEVICE)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: {params:,} params on {DEVICE}")

# %% Cell 5: Training (200 epochs + early stopping)
EPOCHS = 200
PATIENCE = 25  # Early stopping patience
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, eta_min=1e-5)
pl = nn.HuberLoss(); sl_fn = nn.CrossEntropyLoss()
best = float("inf")
patience_counter = 0
train_hist, val_hist, mape_hist = [], [], []

print(f"\n{'='*60}")
print(f"TRAINING — {EPOCHS} epochs, early stopping (patience={PATIENCE})")
print(f"{'='*60}")

t_start = time.time()
for ep in range(EPOCHS):
    # Train
    model.train(); tl = 0
    for xb, yb, sb in tdl:
        xb, yb, sb = xb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)
        pp, ps = model(xb)
        loss = pl(pp, yb) + 0.3 * sl_fn(ps, sb)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tl += loss.item() * len(xb)
    tl /= sp

    # Validate
    model.eval(); vl = vm = 0
    with torch.no_grad():
        for xb, yb, sb in vdl:
            xb, yb, sb = xb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)
            pp, ps = model(xb)
            l = pl(pp, yb) + 0.3 * sl_fn(ps, sb)
            vl += l.item() * len(xb)
            vm += (torch.mean(torch.abs(pp - yb)/(yb.abs()+1e-6))*100).item() * len(xb)
    vl /= (ns-sp); vm /= (ns-sp)
    sched.step()
    
    train_hist.append(tl); val_hist.append(vl); mape_hist.append(vm)

    # Early stopping check
    if vl < best:
        best = vl
        patience_counter = 0
        torch.save({"model_state_dict": model.state_dict(),
                     "config": {"n_features":15,"seq_len":SEQ_LEN,"pred_horizon":PH},
                     "val_loss":vl, "val_mape":vm, "epoch":ep}, "best_model.pt")
    else:
        patience_counter += 1

    if (ep+1) % 20 == 0 or ep == 0 or patience_counter == 0:
        elapsed = time.time() - t_start
        marker = " ★" if patience_counter == 0 else ""
        print(f"Ep {ep+1:3d}/{EPOCHS} | Train {tl:.4f} | Val {vl:.4f} | MAPE {vm:.2f}% | {elapsed:.0f}s{marker}")

    if patience_counter >= PATIENCE:
        print(f"\n⏹ Early stopping at epoch {ep+1} (patience {PATIENCE} exceeded)")
        break

total_time = time.time() - t_start
print(f"\n✅ Training done in {total_time:.0f}s | Best val loss: {best:.4f}")

# Overfitting check
if len(val_hist) > 20:
    recent_train = np.mean(train_hist[-10:])
    recent_val = np.mean(val_hist[-10:])
    gap = (recent_val - recent_train) / recent_train * 100
    print(f"Overfitting check: train={recent_train:.4f}, val={recent_val:.4f}, gap={gap:.1f}%")
    if gap > 50:
        print("⚠️ Possible overfitting detected")
    else:
        print("✅ No significant overfitting")

# %% Cell 6: CVXPY MPC Controller
print(f"\n{'='*60}")
print("CVXPY MPC CONTROLLER")
print(f"{'='*60}")

class CVXPYMPC:
    """Model Predictive Control using CVXPY for battery optimization."""
    
    def __init__(self, horizon=12, dt=5, 
                 soc_min=0.05, soc_max=0.95, efficiency=0.92,
                 battery_power_mw=319.2, battery_capacity_mwh=655.2,
                 grid_target_mw=200.0, Q=100.0, R=0.01, S=0.1):
        self.horizon = horizon
        self.dt = dt / 3600  # Convert seconds to hours
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.efficiency = efficiency
        self.battery_power_mw = battery_power_mw
        self.battery_capacity_mwh = battery_capacity_mwh
        self.grid_target_mw = grid_target_mw
        self.Q = Q  # Tracking weight
        self.R = R  # Control effort weight
        self.S = S  # Terminal weight
        self.soc = 0.5  # Initial SOC
    
    def optimize(self, predicted_power):
        """Solve MPC optimization problem.
        
        Args:
            predicted_power: array of predicted facility power (MW) for horizon steps
            
        Returns:
            battery_power: optimal battery power command (MW, + = discharge)
            soc_trajectory: predicted SOC trajectory
        """
        H = min(self.horizon, len(predicted_power))
        p_load = predicted_power[:H]
        
        # Decision variables
        u = cp.Variable(H)  # Battery power (MW), + = discharge, - = charge
        soc = cp.Variable(H + 1)  # State of charge
        p_grid = cp.Variable(H)  # Grid power
        
        # Constraints
        constraints = [
            soc[0] == self.soc,
            p_grid == p_load - u,
            soc[1:] == soc[:-1] - u * self.dt / self.battery_capacity_mwh * self.efficiency,
            soc >= self.soc_min,
            soc <= self.soc_max,
            u >= -self.battery_power_mw,
            u <= self.battery_power_mw,
            p_grid >= 0,
        ]
        
        # Cost: track grid target + minimize control effort + terminal SOC
        cost = self.Q * cp.sum_squares(p_grid - self.grid_target_mw)
        cost += self.R * cp.sum_squares(u)
        cost += self.S * cp.square(soc[H] - 0.5)
        
        problem = cp.Problem(cp.Minimize(cost), constraints)
        
        try:
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            if problem.status == "optimal":
                self.soc = float(soc.value[H])
                return u.value, soc.value
            else:
                print(f"  MPC status: {problem.status}")
                return np.zeros(H), np.full(H + 1, self.soc)
        except Exception as e:
            print(f"  MPC error: {e}")
            return np.zeros(H), np.full(H + 1, self.soc)

# Test MPC with model predictions
print("Testing MPC with model predictions...")
model.eval()
mpc = CVXPYMPC(horizon=12, dt=5)

# Take a validation sample
test_idx = idx[sp:sp+100]
test_x = torch.tensor(X[test_idx]).to(DEVICE)
with torch.no_grad():
    pred_p, pred_s = model(test_x)
    pred_p = pred_p.cpu().numpy()

total_savings = 0
grid_smooth = 0
for i in range(len(test_idx)):
    battery_cmd, soc_traj = mpc.optimize(pred_p[i])
    grid_power = pred_p[i] - battery_cmd
    peak_reduction = max(0, max(pred_p[i]) - max(grid_power))
    total_savings += peak_reduction
    grid_smooth += np.std(grid_power) / (np.mean(grid_power) + 1e-6)

avg_savings = total_savings / len(test_idx)
avg_smooth = grid_smooth / len(test_idx) * 100

print(f"\n📊 MPC Results ({len(test_idx)} test samples):")
print(f"  Avg peak reduction: {avg_savings:.2f} MW")
print(f"  Grid smoothness: {avg_smooth:.1f}% (lower = smoother)")
print(f"  Battery SOC range: {mpc.soc_min} - {mpc.soc_max}")

# %% Cell 7: Results
results = {
    "timestamp": datetime.now().isoformat(),
    "status": "COMPLETE",
    "device": DEVICE,
    "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU",
    "data_source": "Alibaba GPU Trace 2020 (CC BY 4.0)",
    "data_rows": n,
    "train_samples": sp,
    "val_samples": ns - sp,
    "model_params": params,
    "epochs_run": len(train_hist),
    "epochs_max": EPOCHS,
    "early_stopped": patience_counter >= PATIENCE,
    "best_val_loss": round(float(best), 6),
    "best_val_mape": round(float(min(mape_hist)), 2),
    "final_train_loss": round(float(train_hist[-1]), 6),
    "final_val_loss": round(float(val_hist[-1]), 6),
    "overfitting_gap_pct": round(float(gap), 1) if len(val_hist) > 20 else None,
    "training_time_s": round(total_time),
    "mpc_peak_reduction_mw": round(float(avg_savings), 2),
    "mpc_grid_smoothness_pct": round(float(avg_smooth), 1),
}

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print("📊 FINAL RESULTS")
print(f"{'='*60}")
for k, v in results.items():
    print(f"  {k}: {v}")

print(f"\n✅ DONE! Files: best_model.pt, results.json")
