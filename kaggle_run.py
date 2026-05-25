"""
ENERGIVANU — Kaggle/Colab Auto-Train with HF Hub persistence
python kaggle_run.py  → auto-resume, auto-save, HF upload
"""

import os, sys, pickle, numpy as np, glob, re, time, shutil, json, warnings
warnings.filterwarnings("ignore")

# ─── Config ────────────────────────────────────────────────────────
DATA_SOURCE = "synthetic"        # "synthetic" | "real" (MIT Supercloud)
DAYS = 30
MAX_JOBS = 50                    # job files to download when DATA_SOURCE=real
REAL_STRIDE = 50                 # 50×100ms = 5s per step (match synthetic)
MODEL_TYPE = "transformer"
D_MODEL = 128
N_LAYERS = 3
N_HEADS = 4
D_FF = 512
LOOKBACK = 60
HORIZON = 60
BATCH_SIZE = 256
EPOCHS = 120
PATIENCE = 0
LR = 5e-6
WARMUP = 500
WEIGHT_DECAY = 3e-4
DROPOUT = 0.35
DIR_W = 5.0
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = "vedkumr/energivanu"

# ─── Paths ─────────────────────────────────────────────────────────
BASE = "/kaggle/working/energivanu_prod"
if not os.path.exists("/kaggle"):
    BASE = "/content/energivanu_prod"
DATA_DIR = f"{BASE}/data"
CKPT_DIR = f"{BASE}/checkpoints"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

print(f"{'='*60}")
print(f"  ENERGIVANU — Production Training")
print(f"  Data: {DATA_DIR}")
print(f"  Checkpoints: {CKPT_DIR}")
print(f"  Days: {DAYS} | Model: {D_MODEL}d/{N_LAYERS}l | Epochs: {EPOCHS}")
print(f"{'='*60}")

# ─── Repo setup ────────────────────────────────────────────────────
REPO = "/kaggle/working/energivanu"
SAFE_DIR = "/kaggle/working"
os.chdir(SAFE_DIR)
if not os.path.exists(REPO):
    print("Cloning repo...")
    os.system("git clone https://github.com/mysterious75/Energivanu.git " + REPO)
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch
from src.config import Config
from src.data.generator import generate_dataset
from src.data.features import FeatureStore
from src.models.dlinear import DLinear
from src.models.transformer import ColossusTransformer
from src.engine.trainer import Trainer

if DATA_SOURCE == "real":
    from src.data.real_data import prepare as prepare_real_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

import threading as _th
def _heartbeat():
    while True:
        time.sleep(300)
        print(f"  [heartbeat] {time.strftime('%H:%M:%S')} — training running", flush=True)
_th.Thread(target=_heartbeat, daemon=True).start()

# ─── Config ────────────────────────────────────────────────────────
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
cfg.train.dir_w = DIR_W
cfg.cluster.num_gpus = 150_000

# ─── Data ──────────────────────────────────────────────────────────
if DATA_SOURCE == "real":
    print(f"\n[1/3] Loading real MIT Supercloud data ({MAX_JOBS} jobs)...")
    X, Y, S, D = prepare_real_data(cfg, f"{BASE}/real_data", max_jobs=MAX_JOBS,
                                    stride=REAL_STRIDE,
                                    force_download=not os.path.exists(f"{BASE}/real_data/dataset.pkl"))
    np.save(f"{DATA_DIR}/X.npy", X)
    np.save(f"{DATA_DIR}/Y.npy", Y)
    np.save(f"{DATA_DIR}/S.npy", S)
    np.save(f"{DATA_DIR}/D.npy", D)
else:
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
        print(f"  Features: {cfg.model.num_features} | X:{X.shape} Y:{Y.shape} S:{S.shape} D:{D.shape}")
    else:
        print("\n[1/3] Loading saved data...")
        cfg = pickle.load(open(f"{DATA_DIR}/cfg.pkl", "rb"))
        cfg.train.epochs = EPOCHS
        cfg.train.lr = LR
        cfg.train.patience = PATIENCE
        cfg.train.batch_size = BATCH_SIZE
        cfg.train.warmup = WARMUP
        cfg.train.weight_decay = WEIGHT_DECAY
        cfg.train.dir_w = DIR_W
        cfg.model.dropout = DROPOUT
        X = np.load(f"{DATA_DIR}/X.npy")
        Y = np.load(f"{DATA_DIR}/Y.npy")
        S = np.load(f"{DATA_DIR}/S.npy")
        D = np.load(f"{DATA_DIR}/D.npy")
        print(f"  Loaded: X:{X.shape} Y:{Y.shape} S:{S.shape} D:{D.shape}")

# ─── Resume from best checkpoint ──────────────────────────────────
resume_ep = 0
RESUME_CKPT = f"{CKPT_DIR}/best.pt"
if os.path.exists(RESUME_CKPT):
    ckpt_info = torch.load(RESUME_CKPT, map_location="cpu", weights_only=False)
    resume_ep = ckpt_info["ep"]
    print(f"\n  Found best checkpoint at epoch {resume_ep} (LR: {LR:.0e})")

# ─── Model ─────────────────────────────────────────────────────────
print("\n[2/3] Building model...")
split = int(0.8 * len(X))
y_mean = float(Y[:split].mean())
y_std = float(Y[:split].std()) + 1e-8
Y = (Y - y_mean) / y_std
print(f"  Y standardized: mean={y_mean:.2f}MW, std={y_std:.2f}MW → N(0,1)")

if cfg.model.model_type == "dlinear":
    model = DLinear(cfg.model)
    print(f"  Model: DLinear ({sum(p.numel() for p in model.parameters()):,} params)")
else:
    model = ColossusTransformer(cfg.model)
    print(f"  Model: Transformer ({sum(p.numel() for p in model.parameters()):,} params)")
trainer = Trainer(model, cfg, y_mean=y_mean, y_std=y_std, use_dp=True, num_workers=2)

if resume_ep > 0:
    ckpt = torch.load(RESUME_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    trainer.opt.load_state_dict(ckpt["opt"])
    print(f"  Resuming from epoch {resume_ep}")

# ─── Train ─────────────────────────────────────────────────────────
print(f"\n  Baseline: std(y)={y_std:.2f}MW → predicting mean gives MAE≈{y_std:.2f}MW")
print("\n[3/3] Training...")
t_start = time.time()
history = trainer.fit(
    X[:split], Y[:split], S[:split], D[:split],
    X[split:], Y[split:], S[split:], D[split:],
    drive_dir=CKPT_DIR, save_every=1, resume_from=resume_ep
)
t_total = time.time() - t_start

# ─── Save final ────────────────────────────────────────────────────
shutil.copy("checkpoints/best.pt", f"{CKPT_DIR}/best_final.pt")
with open(f"{CKPT_DIR}/history.json", "w") as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f)

print(f"\n{'='*60}")
print(f"  TRAINING COMPLETE")
print(f"  Time: {t_total/3600:.1f} hours")
print(f"  Best MAE: {min(history['vm']):.2f} MW")
print(f"  Best SigAcc: {max(history['vs'])*100:.1f}%")
print(f"  Model saved: {CKPT_DIR}/best_final.pt")
print(f"{'='*60}")

# ─── Hugging Face upload (if token set) ────────────────────────────
if HF_TOKEN:
    try:
        from huggingface_hub import HfApi, create_repo
        api = HfApi()
        create_repo(HF_REPO, token=HF_TOKEN, exist_ok=True)
        api.upload_file(path_or_fileobj=f"{CKPT_DIR}/best_final.pt",
                        path_in_repo="best.pt", repo_id=HF_REPO, token=HF_TOKEN)
        api.upload_file(path_or_fileobj=f"{CKPT_DIR}/history.json",
                        path_in_repo="history.json", repo_id=HF_REPO, token=HF_TOKEN)
        print(f"  Uploaded to HF: {HF_REPO}")
    except Exception as e:
        print(f"  HF upload failed: {e}")
else:
    print("  Set HF_TOKEN env var to auto-upload to Hugging Face")
