# ENERGIVANU — Mistake Analysis

## Executive Summary

DLinear (15K params) achieves **MAE 53 MW** vs persistence baseline **3.91 MW** — model is **13.5x worse** than trivial baseline. All metrics are broken: SigAcc 91% is meaningless (class imbalance), DirAcc 53% is random. The model fundamentally cannot learn.

---

## 🔴 Colab/Chronos Session Mistakes (May 2026)

### 14. `amazon/chronos-2` — Model Doesn't Exist
**Error**: `TypeError: ChronosConfig.__init__() got an unexpected keyword argument 'input_patch_size'`
**Cause**: Tried to load `amazon/chronos-2` which doesn't exist. Chronos-Bolt needs newer `chronos-forecasting` package.
**Fix**: Use `amazon/chronos-t5-base` (works with older package) OR install from GitHub: `pip install git+https://github.com/amazon-science/chronos-forecasting.git`

### 15. `torch_dtype` vs `dtype` — Deprecated Parameter
**Warning**: `torch_dtype is deprecated! Use dtype instead!`
**Fix**: Use `dtype=torch.float16` instead of `torch_dtype=torch.float16`

### 16. `total_mem` vs `total_memory` — Wrong Attribute
**Error**: `AttributeError: 'torch._C._CudaDeviceProperties' object has no attribute 'total_mem'`
**Fix**: Use `torch.cuda.get_device_properties(0).total_memory` (with try/except fallback)

### 17. `--upgrade` All Packages — Broke numpy
**Error**: `ImportError: cannot import name '_center' from 'numpy._core.umath'`
**Cause**: `pip install --upgrade chronos-forecasting torch numpy pandas matplotlib` upgraded numpy to 2.4.6 which is incompatible with Colab's environment.
**Fix**: Only upgrade chronos-forecasting: `pip install --upgrade chronos-forecasting -q`

### 18. `pipeline` Not Defined — Wrong Cell Order
**Error**: `NameError: name 'pipeline' is not defined`
**Cause**: Ran Cell 6 before Cell 4. Cell 4 creates the `pipeline` variable.
**Fix**: Always run cells in order: 1→2→3→4→5→6→7→8→9→10

### 19. `median_forecast` Shape — Need `.flatten()`
**Error**: Signal classification and direction prediction failed on multi-dim tensor
**Cause**: `median_forecast` is 2D tensor, needs `.flatten()` for iteration
**Fix**: Use `median_forecast.flatten()` before iterating

### 20. Prediction Length Mismatch
**Issue**: Code had `prediction_length = 120` (10 min) but results showed 2400 points
**Cause**: User ran different version or modified prediction_length
**Fix**: Verify prediction_length matches expected forecast horizon

### 21. LoRA Params `requires_grad` Not Set
**Error**: `ValueError: optimizer got an empty parameter list`
**Cause**: Froze all model params with `param.requires_grad = False`, then added LoRA layers, but LoRA params inherited `requires_grad = False`
**Fix**: After adding LoRA, explicitly set `param.requires_grad = True` for all LoRA params:
```python
for name, param in model.named_parameters():
    if 'lora' in name:
        param.requires_grad = True
```

### 22. LoRA Layers Not Added (setattr on dict copy)
**Error**: Still 0 trainable params after LoRA setup
**Cause**: `dict(model.named_modules())` creates a copy, so `setattr(parent, child_name, lora_layer)` doesn't modify the actual model
**Fix**: Use recursive `named_children()` traversal instead:
```python
def replace_linear_with_lora(model, rank=16):
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear):
            lora = LoRALinear(child.in_features, child.out_features, rank)
            lora.original.weight.data = child.weight.data.clone()
            setattr(model, name, lora)
        else:
            replace_linear_with_lora(child, rank)
```

---

## 🟡 Chronos-T5 Zero-Shot Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Model | Chronos-T5 (201.4M params) | Zero-shot, no training |
| Context | 17,280 points (24 hours) | 5-second intervals |
| Prediction | 2,400 points (~3.3 hours) | Longer than expected 10 min |
| Power Range | 44.7 - 71.3 MW | Synthetic data, mostly SAFE |
| SAFE | 2,398 (99.9%) | Expected for synthetic data |
| PREPARE | 2 (0.1%) | Barely triggered |
| CRITICAL | 0 (0%) | Never triggered |
| Direction UP | 1,244 (51.8%) | Near random (50%) |
| Direction DOWN | 1,156 (48.2%) | Model didn't learn direction |

**Conclusion**: Zero-shot Chronos-T5 works but doesn't learn patterns from synthetic data. Need real data for meaningful results.

---

## 🔴 Critical (Architecture)

### 1. `pp[:, 0, :]` — 33/34 Features Discarded
**Location**: `src/models/dlinear.py:38`
```python
pp = self.trend_l(trend) + self.seasonal_l(seasonal)  # (B, 34, 60)
pw = pp[:, 0, :]  # ⚡ only feature 0!
```
DLinear processes all 34 features independently through moving average + 2×Linear(60,60), but then **keeps only feature 0** (`gpu_power_mw`). Weather, solar, grid state, battery — all ignored.

**Fix**: `pw = pp.mean(dim=1)` or `pw = self.proj(pp.reshape(B, -1))`

### 2. `x[:, :, 0]` — Classifiers Also Blind
**Location**: `src/models/dlinear.py:40`
```python
return pw, self.shead(x[:, :, 0]), self.dir_head(x[:, :, 0])
```
Both `shead` (spike classifier) and `dir_head` (direction classifier) only see feature 0. The entire weather + grid + battery signal is invisible to all heads.

**Fix**: Pass all features: `self.shead(x.mean(dim=2))` or `self.shead(x.reshape(B, -1))`

### 3. Variable Shadowing Bug
**Location**: `src/models/dlinear.py:31` (fixed)
```python
B, T, F = x.shape  # F = 34 (int) → overwrites import torch.nn.functional as F
F.pad(...)  # Boom: 'int' object has no attribute 'pad'
```
Classic shadowing. Fixed by renaming to `_ = x.shape`.

---

## 🔴 Critical (Training & Loss)

### 4. Loss Dominated by Price MSE — Other Heads Starve
| Component | Value | Weight | Effective Contribution |
|-----------|-------|--------|----------------------|
| pl (price MSE) | ~15,000 | 1 | **15,000** |
| dl (dir CE) | ~0.69 | ×100 | ~69 |
| sl (sig CE) | ~1.4 | ×0.5 | ~0.7 |

**Gradient ratio** `pl : dl : sl ≈ 174 : 1 : 0.01`

Direction head and signal head receive **negligible gradient**. They never learn.
DirAcc=0.535 = random. SigAcc=0.912 = plateau at prior (always-SAFE gives ~56%).

**Fix**: Normalize price loss (use relative error instead of absolute MW), or balance contributions.

### 5. Target Scale Mismatch
- Input **X**: z-scored per-sample (mean=0, std=1)
- Target **Y**: raw MW (33–119)

Model outputs start at ~0 (xavier init), target is ~60 MW → initial MSE ≈ 3,600. With spike weights, effective loss ≈ 15,000–20,000. This creates huge gradients that destabilize training.

**Fix**: Standardize Y to similar scale, or add output normalization layer.

### 6. Asymmetric Spike Weight Creates Upward Bias
```python
w = torch.where(err > 0, self.uw + self.uw*sp, self.ow)
# Under-predict normal:  5× penalty
# Under-predict spike:  10× penalty
# Over-predict:          1× penalty
```
Model learns to **predict high** to avoid under-prediction penalty, regardless of actual signal. This inflates MAE while reducing spike-loss.

---

## 🟠 Serious (Data)

### 7. Synthetic Data: No Persistent Patterns
- Auto-correlation at 5-min horizon: **0.85** (very predictable)
- Persistence baseline MAE: **3.91 MW**
- DLinear MAE: **53 MW** (13.5x worse → actively harmful)

The data has strong sinusoidal patterns (daily + sub-daily cycles) plus random spikes. A proper model should easily beat persistence. DLinear's failure suggests the synthetic data might lack predictable structure that a small linear model can capture.

### 8. Grid Never Stressed
```
Net load range: [-252, 119] MW
Net load > 120MW: 0.00%
```
Grid max import = 150 MW. Net load never exceeds 120 MW. The "grid emergency" scenario never occurs. Battery barely cycles.

### 9. Spike Detection Threshold Per-Batch
```python
th = tp.mean() + self.ss * tp.std()  # per-batch threshold
```
With batch=4096, this threshold varies every batch. Different batches have different spike definitions. Model can't learn a consistent mapping.

---

## 🟡 Moderate

### 10. Moving Average Kernel=25 for 5s Data
125-second rolling average captures only micro-trends, not daily cycles (24h = 17,280 steps). Original DLinear paper used kernel=25 for **hourly** data.

### 11. DataParallel for 15K Model
```python
if torch.cuda.device_count() > 1:
    self.model = nn.DataParallel(self.model)
```
Two GPUs synchronizing 15K parameters adds ~50μs overhead per batch. For a batch that takes 2ms to compute, this is 25% overhead for zero benefit.

### 12. Batch Size Initially 128
6–11% GPU utilization with 2× T4. Fixed to 4096, utilization expected >80%.

### 13. No Validation Before Training
Never validated that the synthetic data has predictable patterns. Never established baselines. First analysis (this file) reveals model broken at architectural level.

---

## 🟢 Fixed Already

| # | Issue | Fix | Commit |
|---|-------|-----|--------|
| 3 | Variable shadowing (`F.pad` vs `F` features) | Rename `F` → `_` | `947a808` |
| 11 | DataParallel for 15K model | `use_dp=False` for DLinear | `3bb2738` |
| 12 | batch=128 → GPU idle | batch=4096 for DLinear | `3bb2738` |
| — | `F.pad` mode="reflect" bug | `nn.AvgPool1d` with built-in padding | `947a808` |
| — | LR too low for large batch | 1e-3 with warmup=15 | `3bb2738` |

---

## 🔧 Recommended Fixes (Priority Order)

### P0 — Architecture
1. **Use all features**: `pw = pp.mean(dim=1)` or `pw = self.proj(pp.view(B, -1))`
2. **Fix classifiers**: `self.shead(x.mean(dim=2))` or flatten all features
3. **Normalize Y**: Standardize target to match prediction scale

### P0 — Loss
4. **Balance loss components**: Normalize pl to comparable magnitude. Use relative error (MAPE) instead of MSE.
5. **Fix per-batch threshold**: Use global or running statistics for spike definition.

### P1 — Data
6. **Validate data patterns**: Run analysis script before training. Compare against random/persistence baselines.
7. **Increase data complexity**: Add more realistic weather patterns, grid stress scenarios.

### P2 — Performance
8. **Increase num_workers**: 8 instead of 4 for DataLoader (with 2 GPU kernels).
9. **Use `channels_last` memory format** for potential speedup.

---

## Key Metrics to Track

| Metric | Current | Target | Notes |
|--------|---------|--------|-------|
| Val MAE | 53 MW | <5 MW | Persistence gives 3.9 MW |
| Val SigAcc | 0.912 | >0.70 (balanced) | 0.912 = plateau, likely 56% is always-SAFE |
| Val DirAcc | 0.535 | >0.60 | Random = 0.50 |
| Epoch Time (DLinear) | 7.5s | — | Acceptable |
| GPU Utilization | ~11% | >80% | Fixed with batch=4096 |
