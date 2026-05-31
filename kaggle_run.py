"""
ENERGIVANU — Production Training Script v3
Optimized for Kaggle 2x T4 GPU with ALL fixes and multiple model support

MODELS: transformer, dlinear, tsmixer, nlinear, autoformer
FIXES:
  1. Uncertainty Weighting (Kendall et al.) for loss balancing
  2. Direction Classification Head (BCE, not MSE-on-diffs)
  3. Spike detection uses power target (not signal target)
  4. Mixed Precision AMP (FP16 on T4 Tensor Cores)
  5. torch.compile for PyTorch 2.0+
  6. persistent_workers + prefetch_factor=4
  7. Save latest.pt + best.pt for better resume
  8. Heartbeat every 2 min (prevent Kaggle timeout)
  9. Label smoothing for direction classification
  10. Better monitoring (separate loss tracking + uncertainty weights)

USAGE:
  Kaggle Cell 1:
    !rm -rf /kaggle/working/energivanu /kaggle/working/energivanu_prod
    !git clone https://github.com/mysterious75/Energivanu.git /kaggle/working/energivanu
    import sys; sys.path.insert(0, "/kaggle/working/energivanu")
    !pip install -q pyarrow tqdm

  Kaggle Cell 2:
    !cd /kaggle/working/energivanu && python kaggle_run.py
"""

import os, sys, pickle, numpy as np, time, shutil, json, warnings, gc
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DATA_SOURCE = "synthetic"        # "synthetic" | "real"
DAYS = 30

# Model: "transformer" | "dlinear" | "tsmixer" | "nlinear" | "autoformer"
MODEL_TYPE = "transformer"
D_MODEL = 128
N_LAYERS = 3
N_HEADS = 4
D_FF = 512
LOOKBACK = 60
HORIZON = 60
DROPOUT = 0.35

# Training
BATCH_SIZE = 256
EPOCHS = 120
PATIENCE = 0
LR = 1e-4
WARMUP = 500
WEIGHT_DECAY = 3e-4
GRAD_CLIP = 1.0

# Loss
UNDER_W = 5.0
OVER_W = 1.0
SPIKE_STD = 1.5
DIR_WEIGHT = 0.2               # Direction loss downweight (avoids gradient domination by random head)
USE_UNCERTAINTY = False           # Fixed weights for stability
DIR_SMOOTHING = 0.1              # Label smoothing for direction

# Optimization
USE_AMP = True
USE_COMPILE = True
USE_DP = True

# Persistence
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = "vedkumr/energivanu"

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════════

BASE = "/kaggle/working/energivanu_prod"
if not os.path.exists("/kaggle"):
    BASE = "/content/energivanu_prod"
DATA_DIR = f"{BASE}/data"
CKPT_DIR = f"{BASE}/checkpoints"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

print(f"{'='*70}")
print(f"  ENERGIVANU — Production Training v3")
print(f"  Model: {MODEL_TYPE} | D={D_MODEL} L={N_LAYERS} H={N_HEADS}")
print(f"  Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR}")
print(f"  AMP: {USE_AMP} | Uncertainty: {USE_UNCERTAINTY}")
print(f"{'='*70}")

# ═══════════════════════════════════════════════════════════════════════════════
# REPO SETUP
# ═══════════════════════════════════════════════════════════════════════════════

REPO = "/kaggle/working/energivanu"
os.chdir("/kaggle/working")
if not os.path.exists(REPO):
    os.system("git clone https://github.com/mysterious75/Energivanu.git " + REPO)
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch, torch.nn as nn
from src.config import Config
from src.data.generator import generate_dataset
from src.data.features import FeatureStore
from src.models.dlinear import DLinear
from src.models.transformer import ColossusTransformer
from src.models.tsmixer import TSMixer, NLinear
from src.models.autoformer import AutoformerModel
from src.engine.trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} "
              f"({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")

# ═══════════════════════════════════════════════════════════════════════════════
# HEARTBEAT
# ═══════════════════════════════════════════════════════════════════════════════

import threading as _th
t_start = time.time()

def _heartbeat():
    while True:
        time.sleep(120)
        elapsed = time.time() - t_start
        print(f"  [heartbeat] {time.strftime('%H:%M:%S')} | "
              f"elapsed: {elapsed/3600:.1f}h | training active", flush=True)
_th.Thread(target=_heartbeat, daemon=True).start()

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

cfg = Config()
cfg.sim.num_days = DAYS
cfg.sim.pattern_spikes = True
cfg.model.model_type = MODEL_TYPE
cfg.model.d_model = D_MODEL
cfg.model.n_layers = N_LAYERS
cfg.model.n_heads = N_HEADS
cfg.model.d_ff = D_FF
cfg.model.lookback = LOOKBACK
cfg.model.horizon = HORIZON
cfg.model.dropout = DROPOUT
cfg.train.batch_size = BATCH_SIZE
cfg.train.epochs = EPOCHS
cfg.train.patience = PATIENCE
cfg.train.lr = LR
cfg.train.warmup = WARMUP
cfg.train.weight_decay = WEIGHT_DECAY
cfg.train.grad_clip = GRAD_CLIP
cfg.train.under_w = UNDER_W
cfg.train.over_w = OVER_W
cfg.train.spike_std = SPIKE_STD
cfg.train.dir_w = DIR_WEIGHT
cfg.cluster.num_gpus = 150_000

# ═══════════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════════

if not os.path.exists(f"{DATA_DIR}/X.npy"):
    print("\n[1/3] Generating data...")
    df = generate_dataset(cfg)
    df.to_parquet(f"{DATA_DIR}/colossus_{DAYS}d.parquet", index=False)
    fs = FeatureStore(cfg)
    X, Y, S, D = fs.prepare(df, fit=True)
    cfg.model.num_features = X.shape[2]
    np.save(f"{DATA_DIR}/X.npy", X)
    np.save(f"{DATA_DIR}/Y.npy", Y)
    np.save(f"{DATA_DIR}/S.npy", S)
    np.save(f"{DATA_DIR}/D.npy", D)
    pickle.dump(cfg, open(f"{DATA_DIR}/cfg.pkl", "wb"))
    print(f"  Features: {cfg.model.num_features} | X:{X.shape}")
    del df, fs; gc.collect()
else:
    print("\n[1/3] Loading saved data...")
    cfg = pickle.load(open(f"{DATA_DIR}/cfg.pkl", "rb"))
    cfg.model.model_type = MODEL_TYPE
    cfg.model.d_model = D_MODEL
    cfg.model.n_layers = N_LAYERS
    cfg.model.n_heads = N_HEADS
    cfg.model.d_ff = D_FF
    cfg.model.dropout = DROPOUT
    cfg.train.epochs = EPOCHS
    cfg.train.lr = LR
    cfg.train.batch_size = BATCH_SIZE
    cfg.train.dir_w = DIR_WEIGHT
    X = np.load(f"{DATA_DIR}/X.npy")
    Y = np.load(f"{DATA_DIR}/Y.npy")
    S = np.load(f"{DATA_DIR}/S.npy")
    D = np.load(f"{DATA_DIR}/D.npy")
    print(f"  Loaded: X:{X.shape} Y:{Y.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n  Data Distribution:")
print(f"  SAFE: {(S==0).sum():,} ({(S==0).mean()*100:.1f}%)")
print(f"  PREPARE: {(S==1).sum():,} ({(S==1).mean()*100:.1f}%)")
print(f"  CRITICAL: {(S==2).sum():,} ({(S==2).mean()*100:.1f}%)")
print(f"  UP: {(D==1).sum():,} ({(D==1).mean()*100:.1f}%)")
print(f"  DOWN: {(D==0).sum():,} ({(D==0).mean()*100:.1f}%)")

# ═══════════════════════════════════════════════════════════════════════════════
# RESUME
# ═══════════════════════════════════════════════════════════════════════════════

resume_ep = 0
RESUME_CKPT = f"{CKPT_DIR}/latest.pt"
if not os.path.exists(RESUME_CKPT):
    RESUME_CKPT = f"{CKPT_DIR}/best.pt"
model_type_file = f"{CKPT_DIR}/.model_type"
if os.path.exists(RESUME_CKPT) and os.path.exists(model_type_file):
    saved_type = open(model_type_file).read().strip()
    if saved_type == MODEL_TYPE:
        ckpt_info = torch.load(RESUME_CKPT, map_location="cpu", weights_only=False)
        resume_ep = ckpt_info.get("ep", 0)
        print(f"\n  Checkpoint ({saved_type}) found at epoch {resume_ep}")
    else:
        print(f"\n  Model type changed ({saved_type} -> {MODEL_TYPE}), starting fresh")
        resume_ep = 0
elif os.path.exists(RESUME_CKPT):
    print(f"\n  No model type marker, starting fresh")
    resume_ep = 0
else:
    print(f"\n  No checkpoint found, starting fresh")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[2/3] Building model...")
split = int(0.8 * len(X))
y_mean = float(Y[:split].mean())
y_std = float(Y[:split].std()) + 1e-8
Y = (Y - y_mean) / y_std
print(f"  Y standardized: mean={y_mean:.2f}MW, std={y_std:.2f}MW")

# Select model
if MODEL_TYPE == "dlinear":
    model = DLinear(cfg.model)
elif MODEL_TYPE == "tsmixer":
    model = TSMixer(cfg.model)
elif MODEL_TYPE == "nlinear":
    model = NLinear(cfg.model)
elif MODEL_TYPE == "autoformer":
    model = AutoformerModel(cfg.model)
else:
    model = ColossusTransformer(cfg.model)

# NOTE: torch.compile disabled — conflicts with DataParallel
# Can enable later if using single GPU or DDP

with open(f"{CKPT_DIR}/.model_type", "w") as f:
    f.write(MODEL_TYPE)

trainer = Trainer(
    model, cfg, y_mean=y_mean, y_std=y_std,
    use_dp=USE_DP, num_workers=2, use_amp=USE_AMP,
    use_uncertainty=USE_UNCERTAINTY
)

if resume_ep > 0:
    ckpt = torch.load(RESUME_CKPT, map_location=device, weights_only=False)
    model_to_load = trainer.model_raw if hasattr(trainer, 'model_raw') else model
    model_to_load.load_state_dict(ckpt["model"])
    trainer.opt.load_state_dict(ckpt["opt"])
    if "scaler" in ckpt and trainer.scaler is not None:
        trainer.scaler.load_state_dict(ckpt["scaler"])

# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE
# ═══════════════════════════════════════════════════════════════════════════════

Y_val_orig = Y[split:] * y_std + y_mean
persistence_mae = np.abs(Y_val_orig[:, -1:] - Y_val_orig).mean()
print(f"\n  Persistence baseline MAE: {persistence_mae:.2f} MW")
print(f"  Target: MAE < 3.00 MW")

# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n[3/3] Training...")
t_train = time.time()

history = trainer.fit(
    X[:split], Y[:split], S[:split], D[:split],
    X[split:], Y[split:], S[split:], D[split:],
    drive_dir=CKPT_DIR, save_every=1, resume_from=resume_ep
)

t_total = time.time() - t_train

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE & RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

if os.path.exists("checkpoints/best.pt"):
    shutil.copy("checkpoints/best.pt", f"{CKPT_DIR}/best_final.pt")
with open(f"{CKPT_DIR}/history.json", "w") as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f)

print(f"\n{'='*70}")
print(f"  TRAINING COMPLETE — {MODEL_TYPE.upper()}")
print(f"{'='*70}")
print(f"  Time: {t_total/60:.1f} min")
print(f"  Best MAE: {min(history['vm']):.2f} MW (target: <3.00)")
print(f"  Best SigAcc: {max(history['vs'])*100:.1f}% (target: >95%)")
print(f"  Best DirAcc: {max(history['vd'])*100:.1f}% (target: >55%)")
print(f"  Persistence: {persistence_mae:.2f} MW")

if min(history['vm']) < persistence_mae:
    imp = (persistence_mae - min(history['vm'])) / persistence_mae * 100
    print(f"  IMPROVEMENT: {imp:.1f}% better than persistence!")
else:
    print(f"  WARNING: Worse than persistence baseline")

print(f"  Model: {CKPT_DIR}/best_final.pt")
print(f"{'='*70}")

# HF Upload
if HF_TOKEN:
    try:
        from huggingface_hub import HfApi, create_repo
        api = HfApi()
        create_repo(HF_REPO, token=HF_TOKEN, exist_ok=True)
        api.upload_file(path_or_fileobj=f"{CKPT_DIR}/best_final.pt",
                        path_in_repo="best.pt", repo_id=HF_REPO, token=HF_TOKEN)
        print(f"  Uploaded to HF: {HF_REPO}")
    except Exception as e:
        print(f"  HF upload failed: {e}")

# Cleanup
ckpts = sorted([f for f in os.listdir(CKPT_DIR) if f.startswith("checkpoint_ep")])
if len(ckpts) > 5:
    for old in ckpts[:-5]:
        os.remove(f"{CKPT_DIR}/{old}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
print(f"\n  Done! Total: {t_total/60:.1f} min")
