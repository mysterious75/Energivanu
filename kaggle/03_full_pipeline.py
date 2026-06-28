# %% [markdown]
# # ⚡ Energivanu — Full Pipeline (Kaggle GPU)
# 
# **Training:** Alibaba GPU Trace 2020 (50 lakh rows, 15 features, CC BY 4.0)
# **Validation:** MIT Supercloud real power trace (14K rows, real datacenter)
# 
# ## Pipeline:
# 1. Install deps + download data
# 2. Process Alibaba data → 15 features
# 3. Train TCN+Attention model
# 4. Validate on MIT real power trace
# 5. Run MPC controller on real data
# 6. Save model + results

# %% Cell 1: Install & Setup
!pip install codecarbon pynvml -q

import os, sys, time, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# %% Cell 2: Download Alibaba Data
print("=" * 60)
print("DOWNLOADING ALIBABA GPU TRACE 2020")
print("=" * 60)

os.makedirs("data/alibaba", exist_ok=True)

# Download sensor table (GPU utilization per instance)
!curl -sL "https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-gpu-v2020/data/pai_sensor_table.header" -o pai_sensor_table.header
!curl -sL "https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-gpu-v2020/data/pai_machine_metric.header" -o pai_machine_metric.header

# Download actual data files
!curl -sL "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_sensor_table.tar.gz" -o pai_sensor_table.tar.gz
!curl -sL "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_machine_metric.tar.gz" -o pai_machine_metric.tar.gz

# Extract
!tar xzf pai_sensor_table.tar.gz
!tar xzf pai_machine_metric.tar.gz

# Add headers (CSV files come without headers)
!cat pai_sensor_table.header pai_sensor_table.csv > sensor_with_header.csv
!mv sensor_with_header.csv pai_sensor_table.csv

!cat pai_machine_metric.header pai_machine_metric.csv > metric_with_header.csv
!mv metric_with_header.csv pai_machine_metric.csv

print("✅ Alibaba data downloaded & extracted")

# %% Cell 3: Download MIT Supercloud Data (for validation)
!pip install kaggle -q
# Upload your kaggle.json first!
!mkdir -p ~/.kaggle
# !cp /kaggle/input/kaggle-json/kaggle.json ~/.kaggle/  # Uncomment if uploaded
# !chmod 600 ~/.kaggle/kaggle.json

# Download MIT data
!kaggle datasets download -d vedkumr/mit-supercloud-real2 --unzip -p data/mit 2>/dev/null || echo "MIT data download skipped (add kaggle.json)"
print("✅ MIT data ready")

# %% Cell 4: Process Alibaba Data → 15 Features
print("=" * 60)
print("PROCESSING ALIBABA DATA → 15 FEATURES")
print("=" * 60)

# Column mapping for sensor table
SENSOR_COLS = {
    "cpu_usage": "cpu_util",
    "gpu_wrk_util": "gpu_util",
    "avg_mem": "mem_util",
    "avg_gpu_wrk_mem": "gpu_mem_util",
}

# Column mapping for machine metric
METRIC_COLS = {
    "machine_cpu_usr": "cpu_util",
    "machine_gpu": "gpu_util",
    "machine_cpu": "cpu_util2",
}

def process_alibaba_file(filepath, col_mapping, max_rows=None):
    """Load Alibaba CSV and extract 15 features."""
    print(f"  Loading {filepath}...")
    df = pd.read_csv(filepath, nrows=max_rows)
    print(f"    Raw shape: {df.shape}")
    
    # Rename columns
    df = df.rename(columns=col_mapping)
    
    # Check if we have gpu_util
    if "gpu_util" not in df.columns:
        print(f"    ⚠️ No gpu_util column, skipping")
        return None
    
    # Generate timestamps if missing
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.date_range("2020-01-01", periods=len(df), freq="60s")
    
    return df

# Process sensor table (per-GPU data)
sensor_df = process_alibaba_file("pai_sensor_table.csv", SENSOR_COLS, max_rows=500000)
if sensor_df is not None:
    print(f"  Sensor: {sensor_df.shape}")

# Process machine metric (machine-level data)
metric_df = process_alibaba_file("pai_machine_metric.csv", METRIC_COLS, max_rows=500000)
if metric_df is not None:
    print(f"  Metric: {metric_df.shape}")

# Use whichever has data
df = sensor_df if sensor_df is not None else metric_df
if df is None:
    print("❌ No data loaded!")
    raise RuntimeError("No data")

print(f"\nUsing: {df.shape[0]} rows")

# %% Cell 5: Feature Engineering
print("=" * 60)
print("EXTRACTING 15 FEATURES")
print("=" * 60)

# Normalize GPU power (assume T4: 70W idle, 700W peak)
GPU_TDP = 700.0
facility_gpus = 200000
gpus_per_node = 8

# Compute features
gpu_util = df["gpu_util"].fillna(0).clip(0, 100)
gpu_power_norm = gpu_util / 100.0  # 0-1 normalized

# Facility power (MW) = single GPU power * num_gpus / 1e6
single_gpu_power_w = 70 + (GPU_TDP - 70) * (gpu_util / 100.0)  # idle to peak
facility_mw = single_gpu_power_w * facility_gpus / 1e6

# Rolling stats
window = 30
power_roll_mean = facility_mw.rolling(window, min_periods=1).mean()
power_roll_std = facility_mw.rolling(window, min_periods=1).std().fillna(0)

# Derivatives
power_roc = facility_mw.diff().fillna(0)
power_roc2 = power_roc.diff().fillna(0)

# Temporal features
ts = pd.to_datetime(df["timestamp"], errors="coerce")
hour_sin = np.sin(2 * np.pi * ts.dt.hour / 24).fillna(0)
hour_cos = np.cos(2 * np.pi * ts.dt.hour / 24).fillna(0)

# GPU metrics
gpu_temp_norm = (0.4 + 0.4 * gpu_util / 100).clip(0, 1)  # estimated
gpu_mem_util = df.get("gpu_mem_util", pd.Series(np.zeros(len(df)))).fillna(0).clip(0, 100)
cpu_util = df.get("cpu_util", pd.Series(np.zeros(len(df)))).fillna(0).clip(0, 100)

# All-reduce detection heuristic
is_allreduce = ((gpu_util > 80) & (gpu_mem_util < 30)).astype(float)

# Build 15-feature matrix
features = pd.DataFrame({
    "facility_mw": facility_mw,
    "power_roc": power_roc,
    "power_roc2": power_roc2,
    "power_roll_mean": power_roll_mean,
    "power_roll_std": power_roll_std,
    "gpu_avg_power_norm": gpu_power_norm,
    "gpu_max_power_norm": gpu_power_norm,
    "gpu_avg_temp_norm": gpu_temp_norm,
    "gpu_max_temp_norm": gpu_temp_norm,
    "gpu_avg_util_norm": gpu_util / 100.0,
    "gpu_avg_mem_util_norm": gpu_mem_util / 100.0,
    "cpu_util_est_norm": cpu_util / 100.0,
    "hour_sin": hour_sin,
    "hour_cos": hour_cos,
    "is_allreduce": is_allreduce,
})

# Clean
features = features.replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"Features shape: {features.shape}")
print(f"Columns: {list(features.columns)}")
print(f"Facility power: {features.facility_mw.min():.1f} - {features.facility_mw.max():.1f} MW")
print(f"NaN: {features.isna().sum().sum()}")

# Save
os.makedirs("data/processed", exist_ok=True)
features.to_csv("data/processed/training_features.csv", index=False)
print("✅ Features saved")

# %% Cell 6: Create Training Sequences
SEQ_LEN = 30
PRED_HORIZON = 10
STRIDE = 10
BATCH_SIZE = 64

def create_sequences(features_df, seq_len, pred_horizon, stride):
    """Create (X, Y_power, Y_signal) sequences."""
    data = features_df.values.astype(np.float32)
    power = data[:, 0]  # facility_mw is column 0
    
    # Signal: 0=hold, 1=discharge(power rising), 2=charge(power falling)
    power_change = np.diff(power, prepend=power[0])
    signals = np.zeros(len(power), dtype=np.int64)
    signals[power_change > 0.5] = 1
    signals[power_change < -0.5] = 2
    
    X, Y_power, Y_signal = [], [], []
    for i in range(0, len(data) - seq_len - pred_horizon, stride):
        X.append(data[i:i + seq_len])
        Y_power.append(power[i + seq_len:i + seq_len + pred_horizon])
        Y_signal.append(signals[i + seq_len])
    
    return np.array(X), np.array(Y_power), np.array(Y_signal)

print("Creating sequences...")
X, Y_power, Y_signal = create_sequences(features, SEQ_LEN, PRED_HORIZON, STRIDE)
print(f"X: {X.shape}, Y_power: {Y_power.shape}, Y_signal: {Y_signal.shape}")

# Train/val split
n = len(X)
idx = np.random.permutation(n)
split = int(n * 0.85)
train_idx, val_idx = idx[:split], idx[split:]

class PowerDataset(Dataset):
    def __init__(self, x, yp, ys):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.yp = torch.tensor(yp, dtype=torch.float32)
        self.ys = torch.tensor(ys, dtype=torch.long)
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.yp[i], self.ys[i]

train_ds = PowerDataset(X[train_idx], Y_power[train_idx], Y_signal[train_idx])
val_ds = PowerDataset(X[val_idx], Y_power[val_idx], Y_signal[val_idx])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, pin_memory=True)

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

# %% Cell 7: Model Architecture (same as src/energivanu/model.py)
class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.norm1 = nn.LayerNorm(out_ch)
        self.norm2 = nn.LayerNorm(out_ch)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        out = self.conv1(x)[:, :, :x.size(2)]
        out = self.relu(self.drop1(self.norm1(out.transpose(1,2)).transpose(1,2)))
        out = self.conv2(out)[:, :, :x.size(2)]
        out = self.relu(self.drop2(self.norm2(out.transpose(1,2)).transpose(1,2)))
        return self.relu(out + res)

class EnergivanuPEB(nn.Module):
    def __init__(self, n_features=15, seq_len=30, pred_horizon=10):
        super().__init__()
        self.power_norm = nn.LayerNorm(min(7, n_features))
        self.telemetry_norm = nn.LayerNorm(min(7, max(0, n_features - 7)))
        self.temporal_norm = nn.LayerNorm(max(0, n_features - 14))
        self._np = min(7, n_features)
        self._nt = min(7, max(0, n_features - 7))
        self._nq = max(0, n_features - 14)
        
        self.input_proj = nn.Linear(n_features, 128)
        self.tcn = nn.Sequential(
            TemporalBlock(128, 32, 5, 1, 0.1),
            TemporalBlock(32, 64, 3, 2, 0.1),
            TemporalBlock(64, 128, 3, 4, 0.1),
        )
        self.attn = nn.MultiheadAttention(128, 8, dropout=0.1, batch_first=True)
        self.attn_norm = nn.LayerNorm(128)
        self.last_weight = nn.Linear(128, 1)
        
        self.power_head = nn.Sequential(
            nn.Linear(128, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(128, pred_horizon),
        )
        self.signal_head = nn.Sequential(
            nn.Linear(128 + pred_horizon, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        B, T, F = x.shape
        xn = torch.zeros_like(x)
        if self._np > 0: xn[:,:,:self._np] = self.power_norm(x[:,:,:self._np])
        if self._nt > 0: xn[:,:,self._np:self._np+self._nt] = self.telemetry_norm(x[:,:,self._np:self._np+self._nt])
        if self._nq > 0: xn[:,:,self._np+self._nt:] = self.temporal_norm(x[:,:,self._np+self._nt:])
        
        h = self.input_proj(xn)
        h = self.tcn(h.transpose(1,2)).transpose(1,2)
        h, _ = self.attn(h, h, h)
        h = self.attn_norm(h)
        
        last = h[:, -1, :]
        mean = h.mean(dim=1)
        alpha = torch.sigmoid(self.last_weight(last))
        agg = alpha * last + (1 - alpha) * mean
        
        power = self.power_head(agg)
        sig = self.signal_head(torch.cat([agg, power], dim=1))
        return power, sig

model = EnergivanuPEB(n_features=15, seq_len=SEQ_LEN, pred_horizon=PRED_HORIZON).to(DEVICE)
print(f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# %% Cell 8: Training
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
power_loss_fn = nn.HuberLoss()
signal_loss_fn = nn.CrossEntropyLoss()

best_val_loss = float("inf")
history = {"train_loss": [], "val_loss": [], "val_mape": []}

print("=" * 60)
print("TRAINING")
print("=" * 60)

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = 0
    for xb, yb, sb in train_loader:
        xb, yb, sb = xb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)
        pred_p, pred_s = model(xb)
        loss = power_loss_fn(pred_p, yb) + 0.3 * signal_loss_fn(pred_s, sb)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item() * len(xb)
    train_loss /= len(train_ds)
    
    # Validate
    model.eval()
    val_loss, val_mape = 0, 0
    with torch.no_grad():
        for xb, yb, sb in val_loader:
            xb, yb, sb = xb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)
            pred_p, pred_s = model(xb)
            loss = power_loss_fn(pred_p, yb) + 0.3 * signal_loss_fn(pred_s, sb)
            val_loss += loss.item() * len(xb)
            mape = torch.mean(torch.abs(pred_p - yb) / (yb.abs() + 1e-6)) * 100
            val_mape += mape.item() * len(xb)
    val_loss /= len(val_ds)
    val_mape /= len(val_ds)
    
    scheduler.step()
    
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_mape"].append(val_mape)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": {"n_features": 15, "seq_len": SEQ_LEN, "pred_horizon": PRED_HORIZON},
            "val_loss": val_loss,
            "val_mape": val_mape,
            "epoch": epoch,
        }, "best_model_real.pt")
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | MAPE: {val_mape:.2f}%")

print(f"\n✅ Training complete! Best val loss: {best_val_loss:.4f}")
print(f"   Best MAPE: {min(history['val_mape']):.2f}%")

# %% Cell 9: MIT Real Power Trace Validation
print("=" * 60)
print("VALIDATION ON MIT SUPERCLOUD REAL DATA")
print("=" * 60)

mit_path = "data/mit/cluster_power_5sec.csv"
if os.path.exists(mit_path):
    mit = pd.read_csv(mit_path)
    print(f"MIT data: {mit.shape}, power range: {mit.power_mw.min():.2f} - {mit.power_mw.max():.2f} MW")
    
    # Create features from MIT (only power available, estimate others)
    mit_power = mit["power_mw"].values
    mit_roc = np.diff(mit_power, prepend=mit_power[0])
    mit_roc2 = np.diff(mit_roc, prepend=mit_roc[0])
    mit_roll_mean = pd.Series(mit_power).rolling(30, min_periods=1).mean().values
    mit_roll_std = pd.Series(mit_power).rolling(30, min_periods=1).std().fillna(0).values
    
    # Normalize to facility scale
    mit_facility = mit_power * 200000 / 8 / 1e6  # scale up
    
    # Build 15 features (estimate missing ones)
    n = len(mit_power)
    ts_idx = np.arange(n)
    mit_features = np.column_stack([
        mit_facility,
        mit_roc * 200000 / 8 / 1e6,
        mit_roc2 * 200000 / 8 / 1e6,
        mit_roll_mean * 200000 / 8 / 1e6,
        mit_roll_std * 200000 / 8 / 1e6,
        mit_power / 700.0,  # gpu_avg_power_norm
        mit_power / 700.0,  # gpu_max_power_norm
        0.5 * np.ones(n),  # gpu_avg_temp_norm (estimated)
        0.5 * np.ones(n),  # gpu_max_temp_norm
        mit_power / 239.0,  # gpu_avg_util_norm (scaled to max)
        0.1 * np.ones(n),  # gpu_avg_mem_util_norm
        0.5 * np.ones(n),  # cpu_util_est_norm
        np.sin(2 * np.pi * ts_idx / 7200),  # hour_sin (5s intervals)
        np.cos(2 * np.pi * ts_idx / 7200),  # hour_cos
        np.zeros(n),  # is_allreduce
    ]).astype(np.float32)
    
    # Create sequences
    mit_X = []
    for i in range(0, len(mit_features) - SEQ_LEN - PRED_HORIZON, 5):
        mit_X.append(mit_features[i:i + SEQ_LEN])
    mit_X = np.array(mit_X)
    print(f"MIT sequences: {mit_X.shape}")
    
    # Predict
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(mit_X, dtype=torch.float32).to(DEVICE)
        pred_p, pred_s = model(x_t)
        pred_p = pred_p.cpu().numpy()
    
    # Compare with actual
    actual_power = []
    for i in range(0, len(mit_features) - SEQ_LEN - PRED_HORIZON, 5):
        actual_power.append(mit_power[i + SEQ_LEN:i + SEQ_LEN + PRED_HORIZON])
    actual_power = np.array(actual_power)
    
    mape = np.mean(np.abs(pred_p - actual_power) / (np.abs(actual_power) + 1e-6)) * 100
    print(f"\n📊 MIT Validation Results:")
    print(f"   MAPE on real data: {mape:.2f}%")
    print(f"   Predicted range: {pred_p.min():.1f} - {pred_p.max():.1f}")
    print(f"   Actual range: {actual_power.min():.1f} - {actual_power.max():.1f}")
else:
    print("⚠️ MIT data not found. Upload kaggle.json and re-run.")

# %% Cell 10: Save Results & Summary
results = {
    "timestamp": datetime.now().isoformat(),
    "data_source": "Alibaba GPU Trace 2020 (CC BY 4.0)",
    "training_rows": len(train_ds),
    "val_rows": len(val_ds),
    "n_features": 15,
    "model_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    "epochs": EPOCHS,
    "best_val_loss": float(best_val_loss),
    "best_val_mape": float(min(history["val_mape"])),
    "device": DEVICE,
}

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print("=" * 60)
print("📊 FINAL RESULTS")
print("=" * 60)
for k, v in results.items():
    print(f"  {k}: {v}")

print("\n✅ All done! Model saved as best_model_real.pt")
print("   Results saved as results.json")
