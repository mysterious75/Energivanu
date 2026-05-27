# ENERGIVANU — Kaggle Cells

> Copy-paste these cells into Kaggle notebook.
> Enable GPU (2x T4) in notebook settings.
> Enable Internet in notebook settings.

---

## Cell 1: Setup (run first)

```python
# Clean up old data
!rm -rf /kaggle/working/energivanu /kaggle/working/energivanu_prod

# Clone repo
!git clone https://github.com/mysterious75/Energivanu.git /kaggle/working/energivanu

# Add to path
import sys
sys.path.insert(0, "/kaggle/working/energivanu")

# Install dependencies
!pip install -q pyarrow tqdm

# Verify
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem/1e9:.1f} GB)")
```

---

## Cell 2: Run ALL Experiments (recommended)

```python
# Run all 8 experiments one by one
# Each experiment: different model/architecture/settings
# Results saved to /kaggle/working/energivanu_prod/experiments/
!cd /kaggle/working/energivanu && python run_experiments.py
```

---

## Cell 2 (Alternative): Run Single Experiment

```python
# Run single experiment with default settings (TSMixer + Uncertainty)
!cd /kaggle/working/energivanu && python kaggle_run.py
```

---

## Cell 3: Check Results

```python
import json, os

# Check experiment results
exp_dir = "/kaggle/working/energivanu_prod/experiments"
if os.path.exists(f"{exp_dir}/summary.json"):
    with open(f"{exp_dir}/summary.json") as f:
        results = json.load(f)

    print(f"\n{'Experiment':<35} {'MAE':>8} {'SigAcc':>8} {'DirAcc':>8}")
    print(f"{'─'*35} {'─'*8} {'─'*8} {'─'*8}")

    for r in results:
        if "error" in r:
            print(f"{r['name']:<35} {'ERROR':>8}")
        else:
            print(f"{r['name']:<35} {r['best_mae']:>7.2f}M "
                  f"{r['best_sigacc']*100:>7.1f}% {r['best_diracc']*100:>7.1f}%")

    # Best results
    valid = [r for r in results if "error" not in r]
    if valid:
        best = min(valid, key=lambda r: r['best_mae'])
        print(f"\nBest MAE: {best['name']} ({best['best_mae']:.2f} MW)")
        print(f"Target: MAE < 3.00 MW")
else:
    print("No results yet. Run Cell 2 first.")
```

---

## Cell 4: View Training History

```python
import json
import matplotlib.pyplot as plt

history_path = "/kaggle/working/energivanu_prod/experiments/exp1_tsmixer_uncertainty/checkpoints/history.json"
if os.path.exists(history_path):
    with open(history_path) as f:
        hist = json.load(f)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    axes[0].plot(hist['tl'], label='Train')
    axes[0].plot(hist['vl'], label='Val')
    axes[0].set_title('Loss')
    axes[0].legend()

    axes[1].plot(hist['vm'])
    axes[1].set_title('Val MAE (MW)')
    axes[1].axhline(y=3.0, color='r', linestyle='--', label='Target')
    axes[1].legend()

    axes[2].plot(hist['vs'])
    axes[2].set_title('Signal Accuracy')
    axes[2].axhline(y=0.95, color='r', linestyle='--', label='Target')
    axes[2].legend()

    axes[3].plot(hist['vd'])
    axes[3].set_title('Direction Accuracy')
    axes[3].axhline(y=0.55, color='r', linestyle='--', label='Target')
    axes[3].legend()

    plt.tight_layout()
    plt.savefig("/kaggle/working/training_curves.png", dpi=150)
    plt.show()
else:
    print("No history yet. Run training first.")
```

---

## Cell 5: Upload to Hugging Face (optional)

```python
# Set your HF token (get from https://huggingface.co/settings/tokens)
HF_TOKEN = ""  # Add your token here

if HF_TOKEN:
    from huggingface_hub import HfApi, create_repo
    import os

    api = HfApi()
    repo = "vedkumr/energivanu"
    create_repo(repo, token=HF_TOKEN, exist_ok=True)

    # Upload best model
    best_path = "/kaggle/working/energivanu_prod/experiments/exp1_tsmixer_uncertainty/checkpoints/best_final.pt"
    if os.path.exists(best_path):
        api.upload_file(
            path_or_fileobj=best_path,
            path_in_repo="best.pt",
            repo_id=repo,
            token=HF_TOKEN
        )
        print(f"Uploaded to HF: {repo}")
    else:
        print("No model to upload. Run training first.")
else:
    print("Set HF_TOKEN to upload to Hugging Face")
```

---

## Cell 6: Disk Usage Check

```python
import shutil

# Check disk usage
total, used, free = shutil.disk_usage("/kaggle/working")
print(f"Disk: {used/1e9:.1f} GB used / {total/1e9:.1f} GB total ({free/1e9:.1f} GB free)")

# Check experiment sizes
import os
exp_dir = "/kaggle/working/energivanu_prod/experiments"
if os.path.exists(exp_dir):
    for d in os.listdir(exp_dir):
        path = f"{exp_dir}/{d}"
        if os.path.isdir(path):
            size = sum(os.path.getsize(f) for f in os.listdir(path) if os.path.isfile(f))
            print(f"  {d}: {size/1e6:.1f} MB")
```

---

## Notes

- **Session limit**: 9 hours max
- **Inactivity timeout**: Heartbeat prints every 2 min to keep alive
- **Disk**: 20 GB free, we use ~5 GB total
- **Checkpoints**: Auto-saved every epoch, latest.pt + best.pt
- **Resume**: If session disconnects, re-run Cell 1 + Cell 2 (auto-resumes from checkpoint)
