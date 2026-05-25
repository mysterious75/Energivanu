"""
ENERGIVANU — Deep Analysis (standalone)
"""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from src.config import Config
from src.data.generator import generate_dataset

cfg = Config()
cfg.sim.num_days = 1
cfg.sim.pattern_spikes = True
cfg.model.lookback = 60
cfg.model.horizon = 60

print("Generating data...")
df = generate_dataset(cfg)
pwr = df["gpu_power_mw"].values
solar = df["solar_mw"].values
net = pwr - solar

print("=" * 60)
print("DATA ANALYSIS")
print("=" * 60)
print(f"Total rows: {len(df):,}")
print(f"Time span: {cfg.sim.num_days} days @ {cfg.sim.interval_sec}s")
print()

print("--- TARGET (gpu_power_mw) ---")
print(f"Range: [{pwr.min():.2f}, {pwr.max():.2f}] MW")
print(f"Mean: {pwr.mean():.2f} MW, Std: {pwr.std():.2f} MW")
print()

# Build features manually
BASE_COLS = [
    "gpu_power_mw","gpu_load_pct","gpu_temp_c","temp_c","humid_pct",
    "cloud_pct","solar_wm2","solar_mw","wind_ms","freq_hz","volt_pu",
    "soc_pct","batt_mw","grid_mw","hour","dow","pwr_rate","net_load",
]
for wn, w in [("30s",6),("1m",12),("3m",36),("6m",72)]:
    df[f"pm_{wn}"] = df["gpu_power_mw"].rolling(w,min_periods=1).mean()
    df[f"ps_{wn}"] = df["gpu_power_mw"].rolling(w,min_periods=1).std().fillna(0)
    df[f"sw_{wn}"] = (df["gpu_power_mw"].rolling(w,min_periods=1).max()
                      - df["gpu_power_mw"].rolling(w,min_periods=1).min())
df["accel"] = df["pwr_rate"].diff().fillna(0)/5
df["g_stress"] = np.abs(df["freq_hz"]-60)
df["sol_avail"] = df["solar_mw"]/(df["solar_mw"].rolling(72,min_periods=1).max()+1e-6)
df["batt_head"] = df["soc_pct"]/100*cfg.battery.capacity_mwh

cols = BASE_COLS + [c for c in df.columns
                    if c.startswith("pm_") or c.startswith("ps_")
                    or c.startswith("sw_") or c.startswith("accel")
                    or c.startswith("g_") or c.startswith("sol_")
                    or c.startswith("batt_h")]
cols = list(dict.fromkeys(cols))
print(f"Total features: {len(cols)}")
print(f"Names: {cols}")
print()

F = np.nan_to_num(df[cols].values.astype(np.float32))
T = np.nan_to_num(df["gpu_power_mw"].values.astype(np.float32))

lb, hz = 60, 60
X, Y, S, D = [], [], [], []
for i in range(lb, len(F)-hz):
    X.append((F[i-lb:i] - F[i-lb:i].mean(0)) / (F[i-lb:i].std(0) + 1e-8))
    future = T[i:i+hz]
    Y.append(future)
    mx = future.max()
    S.append(2 if mx >= cfg.signal.critical_mw else 1 if mx >= cfg.signal.warning_mw else 0)
    D.append(1 if future[-1] > future[0] else 0)

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32)
S = np.array(S, dtype=np.int64)
D = np.array(D, dtype=np.int64)

print("--- CLASS DISTRIBUTION (signal) ---")
n0, n1, n2 = (S==0).sum(), (S==1).sum(), (S==2).sum()
print(f"SAFE(0):     {n0:>6} ({n0/len(S)*100:.2f}%)")
print(f"WARNING(1):  {n1:>6} ({n1/len(S)*100:.2f}%)")
print(f"CRITICAL(2): {n2:>6} ({n2/len(S)*100:.2f}%)")
print()

print("--- DIRECTION ---")
nu = (D==1).sum(); nd = (D==0).sum()
print(f"UP(1):   {nu:>6} ({nu/len(D)*100:.2f}%)")
print(f"DOWN(0): {nd:>6} ({nd/len(D)*100:.2f}%)")
print()

print("--- BASELINE METRICS ---")
mean_mw = T.mean()
e1 = Y - mean_mw
print(f"1. Predict mean ({mean_mw:.2f} MW):")
print(f"   MAE: {np.abs(e1).mean():.2f} MW, MSE: {(e1**2).mean():.2f}")

last_val = T[lb:][:len(Y)]
e2 = Y - last_val.reshape(-1,1)
print(f"2. Persistence (last known value):")
print(f"   MAE: {np.abs(e2).mean():.2f} MW, MSE: {(e2**2).mean():.2f}")
print()

print("--- CRITICAL EVENT RATE ---")
crit = (Y >= 85).any(1)
print(f"Samples with 85+MW ahead: {crit.mean()*100:.2f}% ({crit.sum()}/{len(Y)})")
print()

print("--- CORRELATION OF LAST TIMESTEP WITH TARGET ---")
flat_X = X[:, -1, :]
y_mean = Y.mean(1)
for i in [0,1,2,3,4,5,6,7,8,9,10,15,16,17]:
    if i < len(cols):
        corr = np.corrcoef(flat_X[:,i], y_mean)[0,1]
        print(f"  {cols[i]:20s}: corr={corr:.4f}")
print()

print("--- AUTO-CORRELATION ---")
p0 = T[:50000] - T[:50000].mean()
xf = np.correlate(p0, p0, mode="same")[len(p0)//2:]
if xf[0] != 0:
    xf = xf / xf[0]
for lag in [1,6,12,60,300,600,1200,2880,5760,17280]:
    if lag < len(xf):
        print(f"  Lag {lag:>6} ({lag*5:>5}s): {xf[lag]:.4f}")
print()

print("--- GRID ---")
print(f"Net load (P-Solar): [{net.min():.2f}, {net.max():.2f}] MW")
print(f"Net load > 120MW: {(net>120).sum()}/{len(net)} => {(net>120).mean()*100:.2f}%")
print()

print("--- KEY FINDINGS ---")
print(f"1. DLinear uses {len(cols)} features but pp[:,0,:] discards {len(cols)-1}")
print(f"2. shead/dir_head use x[:,:,0] → also {len(cols)-1} features ignored")
print(f"3. SigAcc 91% is misleading: class imbalance makes 'always SAFE' ~95%")
print(f"4. DirAcc 53% ≈ random (direction is balanced)")
print(f"5. MAE 53MW > persistence baseline — model WORSE than last-value")
print(f"6. Crit events: {crit.sum()} in {len(Y)} samples")
print(f"7. pl dominates loss: ~15000 vs dir_w*dl ~69 → gradient ratio 174:1")
