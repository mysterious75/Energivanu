"""
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
