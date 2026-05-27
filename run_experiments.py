"""
ENERGIVANU — Experiment Runner
Runs all solutions one by one on Kaggle 2x T4 GPU.
Each experiment is self-contained with auto-resume.

USAGE (Kaggle):
  Cell 1:
    !rm -rf /kaggle/working/energivanu /kaggle/working/energivanu_prod
    !git clone https://github.com/mysterious75/Energivanu.git /kaggle/working/energivanu
    import sys; sys.path.insert(0, "/kaggle/working/energivanu")
    !pip install -q pyarrow tqdm

  Cell 2:
    !cd /kaggle/working/energivanu && python run_experiments.py
"""

import os, sys, json, time, warnings, gc
warnings.filterwarnings("ignore")

os.chdir("/kaggle/working/energivanu" if os.path.exists("/kaggle/working/energivanu") else ".")
sys.path.insert(0, ".")

import numpy as np
import torch

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

EXPERIMENTS = [
    # === Phase 1: Quick Wins ===
    {
        "name": "exp1_tsmixer_uncertainty",
        "desc": "TSMixer + Uncertainty Weighting (best guess)",
        "model_type": "tsmixer",
        "d_model": 128, "n_layers": 3, "n_heads": 4, "d_ff": 512,
        "batch_size": 256, "epochs": 120, "lr": 5e-6,
        "dropout": 0.35, "use_uncertainty": True,
    },
    {
        "name": "exp2_dlinear_uncertainty",
        "desc": "DLinear + Uncertainty Weighting (simple baseline)",
        "model_type": "dlinear",
        "d_model": 128, "n_layers": 3, "n_heads": 4, "d_ff": 512,
        "batch_size": 256, "epochs": 120, "lr": 5e-6,
        "dropout": 0.35, "use_uncertainty": True,
    },
    {
        "name": "exp3_nlinear_uncertainty",
        "desc": "NLinear + Uncertainty Weighting (normalized baseline)",
        "model_type": "nlinear",
        "d_model": 128, "n_layers": 3, "n_heads": 4, "d_ff": 512,
        "batch_size": 256, "epochs": 120, "lr": 5e-6,
        "dropout": 0.35, "use_uncertainty": True,
    },
    {
        "name": "exp4_transformer_uncertainty",
        "desc": "Transformer + Uncertainty Weighting (current model + fix)",
        "model_type": "transformer",
        "d_model": 128, "n_layers": 3, "n_heads": 4, "d_ff": 512,
        "batch_size": 256, "epochs": 120, "lr": 5e-6,
        "dropout": 0.35, "use_uncertainty": True,
    },
    # === Phase 2: Architecture Variations ===
    {
        "name": "exp5_tsmixer_large",
        "desc": "TSMixer larger model (d=256, L=6)",
        "model_type": "tsmixer",
        "d_model": 256, "n_layers": 6, "n_heads": 8, "d_ff": 1024,
        "batch_size": 256, "epochs": 120, "lr": 3e-6,
        "dropout": 0.4, "use_uncertainty": True,
    },
    {
        "name": "exp6_tsmixer_small_lr",
        "desc": "TSMixer with lower LR (3e-6)",
        "model_type": "tsmixer",
        "d_model": 128, "n_layers": 3, "n_heads": 4, "d_ff": 512,
        "batch_size": 256, "epochs": 120, "lr": 3e-6,
        "dropout": 0.35, "use_uncertainty": True,
    },
    # === Phase 3: Different Batch Sizes ===
    {
        "name": "exp7_tsmixer_batch512",
        "desc": "TSMixer with batch=512",
        "model_type": "tsmixer",
        "d_model": 128, "n_layers": 3, "n_heads": 4, "d_ff": 512,
        "batch_size": 512, "epochs": 120, "lr": 7e-6,
        "dropout": 0.35, "use_uncertainty": True,
    },
    {
        "name": "exp8_tsmixer_batch128",
        "desc": "TSMixer with batch=128 (small batch generalization)",
        "model_type": "tsmixer",
        "d_model": 128, "n_layers": 3, "n_heads": 4, "d_ff": 512,
        "batch_size": 128, "epochs": 120, "lr": 5e-6,
        "dropout": 0.35, "use_uncertainty": True,
    },
]


def run_experiment(exp, data_dir, base_dir):
    """Run a single experiment."""
    from src.config import Config
    from src.data.generator import generate_dataset
    from src.data.features import FeatureStore
    from src.models.dlinear import DLinear
    from src.models.transformer import ColossusTransformer
    from src.models.tsmixer import TSMixer, NLinear
    from src.models.losses import SpikeLoss
    from src.engine.trainer import Trainer

    exp_dir = f"{base_dir}/{exp['name']}"
    ckpt_dir = f"{exp_dir}/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {exp['name']}")
    print(f"  {exp['desc']}")
    print(f"  Model: {exp['model_type']} | D={exp['d_model']} L={exp['n_layers']}")
    print(f"  Batch: {exp['batch_size']} | LR: {exp['lr']} | Epochs: {exp['epochs']}")
    print(f"{'='*70}")

    # Config
    cfg = Config()
    cfg.sim.num_days = 30
    cfg.sim.pattern_spikes = True
    cfg.model.model_type = exp['model_type']
    cfg.model.d_model = exp['d_model']
    cfg.model.n_layers = exp['n_layers']
    cfg.model.n_heads = exp['n_heads']
    cfg.model.d_ff = exp['d_ff']
    cfg.model.dropout = exp['dropout']
    cfg.train.batch_size = exp['batch_size']
    cfg.train.epochs = exp['epochs']
    cfg.train.lr = exp['lr']
    cfg.train.patience = 0
    cfg.cluster.num_gpus = 150_000

    # Data
    x_path = f"{data_dir}/X.npy"
    if not os.path.exists(x_path):
        print("  Generating data...")
        df = generate_dataset(cfg)
        fs = FeatureStore(cfg)
        X, Y, S, D = fs.prepare(df, fit=True)
        cfg.model.num_features = X.shape[2]
        np.save(x_path, X)
        np.save(f"{data_dir}/Y.npy", Y)
        np.save(f"{data_dir}/S.npy", S)
        np.save(f"{data_dir}/D.npy", D)
        import pickle
        pickle.dump(cfg, open(f"{data_dir}/cfg.pkl", "wb"))
        del df, fs; gc.collect()
    else:
        X = np.load(x_path)
        Y = np.load(f"{data_dir}/Y.npy")
        S = np.load(f"{data_dir}/S.npy")
        D = np.load(f"{data_dir}/D.npy")
        import pickle
        cfg_loaded = pickle.load(open(f"{data_dir}/cfg.pkl", "rb"))
        cfg.model.num_features = cfg_loaded.model.num_features

    # Standardize Y
    split = int(0.8 * len(X))
    y_mean = float(Y[:split].mean())
    y_std = float(Y[:split].std()) + 1e-8
    Y = (Y - y_mean) / y_std

    # Resume
    resume_ep = 0
    resume_ckpt = f"{ckpt_dir}/latest.pt"
    if not os.path.exists(resume_ckpt):
        resume_ckpt = f"{ckpt_dir}/best.pt"
    if os.path.exists(resume_ckpt):
        ckpt_info = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        resume_ep = ckpt_info.get("ep", 0)
        print(f"  Resuming from epoch {resume_ep}")

    # Model
    if exp['model_type'] == 'dlinear':
        model = DLinear(cfg.model)
    elif exp['model_type'] == 'tsmixer':
        model = TSMixer(cfg.model)
    elif exp['model_type'] == 'nlinear':
        model = NLinear(cfg.model)
    else:
        model = ColossusTransformer(cfg.model)

    # NOTE: torch.compile disabled — conflicts with DataParallel
    # Can enable later if using single GPU or DDP

    trainer = Trainer(model, cfg, y_mean=y_mean, y_std=y_std,
                      use_dp=True, num_workers=2, use_amp=True)

    if resume_ep > 0:
        ckpt = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        model_to_load = trainer.model_raw if hasattr(trainer, 'model_raw') else model
        model_to_load.load_state_dict(ckpt["model"])
        trainer.opt.load_state_dict(ckpt["opt"])
        if "scaler" in ckpt and trainer.scaler is not None:
            trainer.scaler.load_state_dict(ckpt["scaler"])

    # Baseline
    Y_val_orig = Y[split:] * y_std + y_mean
    persistence_mae = np.abs(Y_val_orig[:, -1:] - Y_val_orig).mean()

    # Train
    t0 = time.time()
    history = trainer.fit(
        X[:split], Y[:split], S[:split], D[:split],
        X[split:], Y[split:], S[split:], D[split:],
        drive_dir=ckpt_dir, save_every=1, resume_from=resume_ep
    )
    t_total = time.time() - t0

    # Save results
    results = {
        "name": exp['name'],
        "desc": exp['desc'],
        "config": exp,
        "best_mae": min(history['vm']),
        "best_sigacc": max(history['vs']),
        "best_diracc": max(history['vd']),
        "persistence_mae": persistence_mae,
        "time_min": t_total / 60,
        "epochs_trained": len(history['tl']),
    }

    with open(f"{exp_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  RESULTS: {exp['name']}")
    print(f"  MAE: {results['best_mae']:.2f} MW (target: <3.00)")
    print(f"  SigAcc: {results['best_sigacc']*100:.1f}% (target: >95%)")
    print(f"  DirAcc: {results['best_diracc']*100:.1f}% (target: >55%)")
    print(f"  Time: {t_total/60:.1f} min")

    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return results


def main():
    """Run all experiments sequentially."""
    base_dir = "/kaggle/working/energivanu_prod/experiments"
    data_dir = "/kaggle/working/energivanu_prod/data"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Heartbeat
    import threading
    t_start = time.time()
    def _heartbeat():
        while True:
            time.sleep(120)
            elapsed = time.time() - t_start
            print(f"  [heartbeat] {time.strftime('%H:%M:%S')} | "
                  f"elapsed: {elapsed/3600:.1f}h", flush=True)
    threading.Thread(target=_heartbeat, daemon=True).start()

    print(f"{'='*70}")
    print(f"  ENERGIVANU — Experiment Runner")
    print(f"  Experiments: {len(EXPERIMENTS)}")
    print(f"  Data: {data_dir}")
    print(f"  Results: {base_dir}")
    print(f"{'='*70}")

    all_results = []

    for i, exp in enumerate(EXPERIMENTS):
        print(f"\n{'#'*70}")
        print(f"  Experiment {i+1}/{len(EXPERIMENTS)}: {exp['name']}")
        print(f"{'#'*70}")

        try:
            result = run_experiment(exp, data_dir, base_dir)
            all_results.append(result)
        except Exception as e:
            print(f"\n  ERROR in {exp['name']}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"name": exp['name'], "error": str(e)})

        # Save summary after each experiment
        with open(f"{base_dir}/summary.json", "w") as f:
            json.dump(all_results, f, indent=2)

    # Final summary
    print(f"\n{'='*70}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"{'='*70}")
    print(f"\n  {'Experiment':<35} {'MAE':>8} {'SigAcc':>8} {'DirAcc':>8} {'Time':>8}")
    print(f"  {'─'*35} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for r in all_results:
        if "error" in r:
            print(f"  {r['name']:<35} {'ERROR':>8}")
        else:
            print(f"  {r['name']:<35} {r['best_mae']:>7.2f}M "
                  f"{r['best_sigacc']*100:>7.1f}% {r['best_diracc']*100:>7.1f}% "
                  f"{r['time_min']:>7.1f}m")

    # Find best
    valid = [r for r in all_results if "error" not in r]
    if valid:
        best_mae = min(valid, key=lambda r: r['best_mae'])
        best_dir = max(valid, key=lambda r: r['best_diracc'])
        print(f"\n  Best MAE: {best_mae['name']} ({best_mae['best_mae']:.2f} MW)")
        print(f"  Best DirAcc: {best_dir['name']} ({best_dir['best_diracc']*100:.1f}%)")

    print(f"\n  Results saved to: {base_dir}/summary.json")


if __name__ == "__main__":
    main()
