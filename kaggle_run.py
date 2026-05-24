"""
ENERGIVANU — Kaggle/Colab Auto-Train with HF Hub persistence
python kaggle_run.py  → auto-resume, auto-save, HF upload
"""

import os, sys, pickle, numpy as np, glob, re, time, shutil, json, warnings
warnings.filterwarnings("ignore")

# ─── Config ────────────────────────────────────────────────────────
DAYS = 30
MODEL_TYPE = "transformer"
D_MODEL = 128
N_LAYERS = 3
N_HEADS = 4
D_FF = 512
LOOKBACK = 60
HORIZON = 60
BATCH_SIZE = 128
EPOCHS = 80
PATIENCE = 0
LR = 1e-4
WEIGHT_DECAY = 3e-4
DROPOUT = 0.35
DIR_W = 100.0
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
cfg.train.weight_decay = WEIGHT_DECAY
cfg.train.dir_w = DIR_W
cfg.cluster.num_gpus = 150_000

# ─── Data (skip if already saved) ──────────────────────────────────
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
    X = np.load(f"{DATA_DIR}/X.npy")
    Y = np.load(f"{DATA_DIR}/Y.npy")
    S = np.load(f"{DATA_DIR}/S.npy")
    D = np.load(f"{DATA_DIR}/D.npy")
    print(f"  Loaded: X:{X.shape} Y:{Y.shape} S:{S.shape} D:{D.shape}")

# ─── Resume check ──────────────────────────────────────────────────
resume_ep = 0
ckpts = sorted(glob.glob(f"{CKPT_DIR}/checkpoint_ep*.pt"))
if not ckpts:
    ckpts = sorted(glob.glob(f"{CKPT_DIR}/*.pt"))
if ckpts:
    last = ckpts[-1]
    m = re.search(r'ep(\d+)', str(last))
    if m:
        resume_ep = int(m.group(1))
        print(f"\n  Found checkpoint at epoch {resume_ep}")

# ─── Model ─────────────────────────────────────────────────────────
print("\n[2/3] Building model...")
split = int(0.8 * len(X))
if cfg.model.model_type == "dlinear":
    model = DLinear(cfg.model)
    print(f"  Model: DLinear ({sum(p.numel() for p in model.parameters()):,} params)")
else:
    model = ColossusTransformer(cfg.model)
    print(f"  Model: Transformer ({sum(p.numel() for p in model.parameters()):,} params)")
trainer = Trainer(model, cfg)

if resume_ep > 0:
    ckpt = torch.load(ckpts[-1], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    trainer.opt.load_state_dict(ckpt["opt"])
    print(f"  Resuming from epoch {resume_ep}")

# ─── Train ─────────────────────────────────────────────────────────
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
