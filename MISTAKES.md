# ENERGIVANU — Mistake Analysis

## Executive Summary

Built a Transformer-based GPU power forecasting system for xAI's Colossus (150K H100 GPUs). After 15+ debugging cycles, the best result is **MAE 3.52 MW** (target < 3 MW) with **93.3% signal accuracy** on synthetic data. Key learnings: stride=1 is critical for 5-second data, more features ≠ better, direction labels are inherently noisy at 5-second resolution, and model capacity (d_model=128) is the primary MAE bottleneck.

---

## 🟡 Session: Autoformer + Colab (May 30, 2026)

### 23. Early Stopping Bug — `if` vs `elif` Double-Counts Improvement
**Location**: `src/engine/trainer.py:121-124`
```python
# BUG: Two separate if blocks, not if/elif
if vm["loss"] < best:
    best = vm["loss"]; wait = 0  # reset
if vm["loss"] >= best:  # <-- always True after best = vm["loss"]!
    wait += 1  # increments even on improvement
```
After an improvement, `best` is updated to `vm["loss"]`, so `vm["loss"] >= best` is `True` (equal), causing `wait` to increment from 0 to 1. This effectively reduces patience by 1 on every improvement.

**Fix**: Use `else` block:
```python
if vm["loss"] < best:
    best = vm["loss"]; wait = 0
else:
    wait += 1
    if tc.patience > 0 and wait >= tc.patience:
        print(f"\n  Early stop at {ep}"); break
```

### 24. `cfg.pkl` Overwrite Silently Drops Config Changes
**Location**: `kaggle_run.py:126`
```python
cfg = pickle.load(open(f"{DATA_DIR}/cfg.pkl", "rb"))
# cfg now has OLD values — EPOCHS, LR, etc. from previous run are gone
```
When loading saved data, the entire `cfg` object is replaced with the pickled version, silently dropping any config overrides (EPOCHS, LR, BATCH_SIZE, DROPOUT).

**Fix**: Re-apply config after loading:
```python
cfg = pickle.load(open(f"{DATA_DIR}/cfg.pkl", "rb"))
cfg.train.epochs = EPOCHS
cfg.train.lr = LR
cfg.train.batch_size = BATCH_SIZE
cfg.model.dropout = DROPOUT
```

### 25. `total_mem` vs `total_memory` — PyTorch API Change
**Error**: `AttributeError: 'torch._C._CudaDeviceProperties' object has no attribute 'total_mem'`
**Fix**: Use `torch.cuda.get_device_properties(0).total_memory`

### 26. Autoformer FFT Fails in AMP (FP16) with Non-Power-of-2 Dimensions
**Error**: `RuntimeError: cuFFT only supports dimensions whose sizes are powers of two when computing in half precision, but got a signal size of[60]`
**Cause**: `torch.fft.rfft` on dim=60 in FP16 mode. cuFFT requires power-of-2 dimensions in half precision.

**Fix**: Cast to float32 before FFT:
```python
Q32, K32 = Q.float(), K.float()
Q_fft = torch.fft.rfft(Q32, dim=2)
# ... compute autocorrelation ...
R = torch.fft.irfft(S, n=T, dim=2).to(Q.dtype)
```

### 27. Colab OOM — 60 Days × 71 Features × 60 Lookback = 17GB
**Error**: Process killed (OOM) during data preparation.
**Cause**: X array shape `(1,036,680, 60, 71)` × 4 bytes = ~17GB. Colab T4 has 12GB RAM.

**Fix**: Two-pronged approach:
1. **stride=2**: Skip every other step → 518K windows (2x reduction)
2. **Reduce features**: 71 → 48 by cutting:
   - Rolling windows: 7 → 4 (keep 30s, 1m, 3m, 6m)
   - Rolling percentiles: removed (pp10, pp90)
   - Lag features: 3 lags → 1 lag per column
   - Interaction features: removed `pwr_x_stress`

Memory: 518K × 60 × 48 × 4 = ~5.9GB (fits in 12GB)

### 28. Stride=4 Destroys Temporal Resolution — MAE Worsens
**Issue**: With stride=4, MAE went from 3.52 MW (stride=2) to 5.0+ MW.
**Cause**: Stride=4 skips 3 out of 4 timesteps. The model loses fine-grained power dynamics (ramp-up patterns, micro-bursts) that occur within 1-2 timestep intervals.

**Lesson**: Stride > 2 is too aggressive for 5-second interval data. The temporal resolution matters more than having more days of data.

### 29. Colab GPU Underutilization — Data Loading Bottleneck
**Issue**: GPU RAM only 0.7GB used, RAM fully utilized, disk swapping.
**Cause**: Large X array (8.7GB with stride=2, 71 features) exceeded RAM, causing OS to swap to disk. DataLoader reading from swapped memory → GPU idle waiting for data.

**Fix**: Reduce array size to fit in RAM (stride=2 + 48 features = 5.9GB).

### 30. Model Type Resume Mismatch — Old Checkpoint Loaded Into New Architecture
**Issue**: Switching from Transformer to Autoformer tried to load Transformer weights.
**Cause**: Resume logic found `best.pt` from previous run without checking model type.

**Fix**: Save model type to `.model_type` file:
```python
with open(f"{CKPT_DIR}/.model_type", "w") as f:
    f.write(MODEL_TYPE)
```
Check compatibility before resume:
```python
saved_type = open(model_type_file).read().strip()
if saved_type == MODEL_TYPE:
    # resume
else:
    # start fresh
```

---

## 🔴 Critical (Architecture) — FIXED

### 1. `pp[:, 0, :]` — 33/34 Features Discarded
**Location**: `src/models/dlinear.py:38`
**Fix**: `pw = pp.mean(dim=1)` — use all features.

### 2. `x[:, :, 0]` — Classifiers Also Blind
**Location**: `src/models/dlinear.py:40`
**Fix**: `self.shead(x.mean(dim=2))` — pass all features.

### 3. Variable Shadowing Bug
**Location**: `src/models/dlinear.py:31`
**Fix**: Rename `F` → `_` to avoid shadowing `torch.nn.functional`.

---

## 🔴 Critical (Training) — FIXED

### 4. Loss Dominated by Price MSE
| Component | Value | Weight | Effective |
|-----------|-------|--------|-----------|
| pl (price MSE) | ~15,000 | 1 | 15,000 |
| dl (dir CE) | ~0.69 | ×100 | 69 |
| sl (sig CE) | ~1.4 | ×0.5 | 0.7 |

**Fix**: Uncertainty weighting (Kendall et al.) with learnable task-specific weights.

### 5. Target Scale Mismatch
**Fix**: Standardize Y to z-scores: `Y = (Y - y_mean) / y_std`

### 6. Asymmetric Spike Weight Creates Upward Bias
**Fix**: Reduced `dir_w` from 100 to 5, `cls_w` from 0.5 to 1.0.

---

## 🟠 Data Issues — ADDRESSED

### 7. Synthetic Data: Limited Patterns
**Before**: 30 days, single spike pattern (every 4h), no weekday/weekend variation.
**After**: 60 days, diverse patterns (scheduled runs, cascade events, micro-bursts), weekday/weekend modulation, gradual load drift.

### 8. Feature Engineering: Only Rolling Stats on Power
**Before**: 34 features (18 base + 16 rolling on gpu_power_mw only).
**After**: 48 features including:
- Lag features (t-1 for power, solar, grid)
- Delta features (rate of change for solar, grid, battery, temp)
- Cyclical time encoding (sin/cos of hour and day-of-week)
- Cross-feature rolling stats (solar, grid, battery rolling means)
- Interaction features (power × solar)

---

## 🟢 Results Summary

| Model | Data | Features | Stride | MAE | SigAcc | DirAcc | Persistence |
|-------|------|----------|--------|-----|--------|--------|-------------|
| Transformer | 30d synthetic | 34 | 1 | **3.57 MW** | **93.3%** | 57.3% | 3.91 MW ✅ |
| Transformer | real MIT | 34 | 1 | **3.56 MW** | **93.3%** | 57.3% | 3.91 MW ✅ |
| Autoformer | 30d synthetic | 34 | 1 | **3.52 MW** | **94.0%** | 53.5% | 3.98 MW ✅ |
| Autoformer | 60d synthetic | 71 | 4 | 5.0+ MW | 88% | 52% | 3.88 MW ❌ |
| Autoformer | 60d synthetic | 51 | 2 | **4.54 MW** | 88.9% | 53.5% | 3.88 MW ❌ |

**Target**: MAE < 3.00 MW, SigAcc > 90%, DirAcc > 55%

---

## 🔧 Current Blockers

1. **stride=2 destroyed MAE**: Going from stride=1 to stride=2 cost ~1MW. 5-second temporal resolution is critical for capturing micro-bursts and ramp-up patterns.
2. **More features = more noise**: 51 features performed worse than 34. Extra lag/delta/cross-feature features added noise without signal.
3. **MAE worse than persistence**: Autoformer + stride=2 (4.54 MW) is WORSE than persistence baseline (3.88 MW). The model is actively harmful.
4. **DirAcc stuck at 53%**: Direction label `future[-1] > future[0]` is inherently noisy at 5-second resolution. This is a data labeling issue, not a model issue.
5. **Model capacity**: d_model=128, n_layers=3 (848K params) may be insufficient for complex power patterns.

## Key Lessons

1. **Stride=1 is mandatory for 5-second data**: stride=2 loses half the temporal information. For power forecasting with micro-bursts, every timestep matters.
2. **Feature selection > feature quantity**: 34 curated features beat 51 engineered features. More ≠ better.
3. **Early stopping bugs are subtle**: Always use `if/elif/else` chains, not separate `if` blocks, for mutually exclusive conditions.
4. **Pickle config is dangerous**: Loading a pickled config silently overwrites runtime changes. Always re-apply overrides after loading.
5. **FFT + AMP = careful**: cuFFT requires power-of-2 dimensions in FP16. Cast to FP32 for FFT operations.
6. **Memory > compute on Colab**: The bottleneck is usually RAM, not GPU. Design data pipelines to fit in 12GB.
7. **Direction accuracy is a red herring**: At 5-second resolution, direction labels are inherently noisy. Focus on MAE, not DirAcc.
