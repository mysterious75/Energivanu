# ENERGIVANU — Master Plan

> Goal: MAE < 3MW, SigAcc > 95%, DirAcc > 55%
> Platform: Kaggle (2x T4 GPU, 16GB VRAM each)
> Current Best: MAE=5.12 MW, SigAcc=90.9%, DirAcc=50%

## IMPLEMENTATION STATUS

### Files Modified
- `src/models/losses.py` — Uncertainty Weighting (Kendall et al.), label smoothing, fixed spike detection
- `src/engine/trainer.py` — AMP, heartbeat, loss parameter monitoring, latest.pt saving
- `src/models/tsmixer.py` — NEW: TSMixer + NLinear models
- `kaggle_run.py` — Complete rewrite with all optimizations
- `run_experiments.py` — NEW: Runs all experiments sequentially
- `src/data/real_data.py` — Fixed lb_real bug

### Research Completed (7 documents in research/)
1. `01_gpu_power_forecasting.md` — PI-DLinear, thermal RC network, H100 specs
2. `02_time_series_architectures.md` — DLinear, PatchTST, TSMixer, iTransformer comparison
3. `03_direction_classification_and_losses.md` — Uncertainty weighting, focal loss, gradient balancing
4. `04_kaggle_t4_optimization.md` — AMP, torch.compile, DataLoader settings
5. `05_real_data_sources.md` — DCGM, MIT Supercloud, NOAA, augmentation
6. `06_physics_informed_models.md` — PINNs, thermal modeling, battery dynamics
7. `07_foundation_models.md` — Chronos-2, TimesFM, Moirai

---

## Problem Analysis

### Critical Issues
1. **OVERFITTING** — Val loss increases from epoch 1, model memorizes training data
2. **DirAcc = 50%** — Model cannot predict power direction (up/down)
3. **Synthetic Data** — No real GPU telemetry, model may not generalize
4. **Loss Imbalance** — Power MSE dominates, direction/signal heads starve

### Root Causes
- Smooth synthetic data is too predictable → memorized, not learned
- Direction treated as regression (MSE) instead of classification (BCE)
- Gradient ratio 174:1 (power:direction) → direction head never learns
- No physics constraints → model can predict physically impossible values

---

## Solution Roadmap

### Phase 1: Fix Core Architecture (Highest Priority)

#### Solution 1.1: Direction Classification Head
**Problem**: DirAcc = 50% (random)
**Fix**: Binary classification head with BCE loss
- Target: Y[end] > Y[start] → UP(1), else DOWN(0)
- Loss: F.cross_entropy(dir_logits, dir_labels)
- Weight: dir_w calibrated to match power loss magnitude
**Expected**: DirAcc > 55%
**Effort**: Low (small code change)

#### Solution 1.2: Balanced Multi-Task Loss
**Problem**: Power MSE (~15,000) dominates direction loss (~69)
**Fix**: 
- Option A: Normalize power loss (use MAPE or relative error)
- Option B: Uncertainty weighting (learn task weights)
- Option C: GradNorm (adaptive gradient balancing)
**Expected**: All heads get meaningful gradients
**Effort**: Medium

#### Solution 1.3: Spike-Aware Direction Loss
**Problem**: Direction matters most during spikes, not steady state
**Fix**: Weight direction loss higher during spike periods
- During spikes: dir_w * 3
- During steady state: dir_w * 0.5
**Expected**: Better direction prediction during critical events
**Effort**: Low

### Phase 2: Try Alternative Architectures

#### Solution 2.1: DLinear (Simple Linear Model)
**Problem**: Transformer overfits on 518K samples
**Fix**: DLinear with ~15K params (vs 1M Transformer)
- Already implemented in src/models/dlinear.py
- Uses all 34 features (bug fixed)
- Much less overfitting risk
**Expected**: MAE < 5 MW, less overfitting
**Effort**: Low (code exists)

#### Solution 2.2: PI-DLinear (Physics-Informed)
**Problem**: Model doesn't respect physics constraints
**Fix**: Add thermal RC network constraints
- Newton's cooling law: dT/dt = (P - k*(T-T_amb)) / C
- GPU power = idle + dynamic * utilization^alpha
- Physics loss = conservation law violations
**Expected**: 0.78%-39% better MSE than pure data-driven
**Effort**: High (new model architecture)

#### Solution 2.3: PatchTST
**Problem**: Vanilla Transformer attention is permutation-invariant
**Fix**: Channel-independent patching
- Already have patch_size=10 in code
- Process each feature independently
- Better for multivariate time series
**Expected**: Better than vanilla Transformer
**Effort**: Medium (modify existing code)

#### Solution 2.4: NLinear
**Problem**: DLinear may be too simple
**Fix**: NLinear normalizes input before linear layer
- Handles distribution shift better
- Similar complexity to DLinear
**Expected**: Better than DLinear on some datasets
**Effort**: Low (new model file)

### Phase 3: Improve Data Quality

#### Solution 3.1: Pattern-Based Spike Generator
**Problem**: Smooth data too easy, random spikes too hard
**Fix**: Structured patterns:
- Scheduled job starts (every 4 hours)
- Cascading failures (GPU chain reactions)
- Temperature-induced throttling
- Grid frequency dips
**Expected**: More realistic, learnable patterns
**Effort**: Medium

#### Solution 3.2: Data Augmentation
**Problem**: 518K samples may not be enough
**Fix**: Time series augmentation:
- Jittering (add noise)
- Scaling (multiply by random factor)
- Window slicing
- Magnitude warping
**Expected**: 2-3x more effective data
**Effort**: Low

#### Solution 3.3: Real Data Integration
**Problem**: Synthetic data may not generalize
**Fix**: Use MIT Supercloud dataset (already in real_data.py)
- Download real GPU traces from S3
- Scale to 150K GPUs
- Add weather/grid features
**Expected**: Better generalization
**Effort**: Medium

### Phase 4: Training Optimization

#### Solution 4.1: Mixed Precision (AMP)
**Problem**: Training speed
**Fix**: FP16 training with automatic mixed precision
- 2x faster on T4 Tensor Cores
- Lower memory usage
**Expected**: 2x speedup, same accuracy
**Effort**: Low

#### Solution 4.2: Gradient Accumulation
**Problem**: Want larger effective batch without sharp minima
**Fix**: batch=32, accumulate 4 steps = effective 128
**Expected**: Better generalization than large batch
**Effort**: Low

#### Solution 4.3: Learning Rate Optimization
**Problem**: Current LR may not be optimal
**Fix**: 
- Cosine annealing with warm restarts
- One-cycle policy
- LR finder
**Expected**: Better convergence
**Effort**: Low

#### Solution 4.4: Regularization Enhancement
**Problem**: Overfitting
**Fix**:
- Increase dropout to 0.4-0.5
- Add DropPath (stochastic depth)
- Mixup augmentation
- CutMix for time series
**Expected**: Less overfitting
**Effort**: Medium

### Phase 5: Foundation Models (If Phase 1-4 Don't Reach Target)

#### Solution 5.1: Chronos-2
**Problem**: Need better pre-trained features
**Fix**: Use Amazon's Chronos-2 foundation model
- Supports multivariate + covariates
- Fine-tune on our data
**Expected**: State-of-art performance
**Effort**: High

#### Solution 5.2: TimesFM
**Problem**: Alternative foundation model
**Fix**: Google's TimesFM
- Pre-trained on large corpus
- Fine-tune for GPU power
**Expected**: Good zero-shot performance
**Effort**: High

---

## Implementation Order

### Round 1: Quick Wins (Try First)
1. Solution 1.1: Direction Classification Head
2. Solution 1.2: Balanced Multi-Task Loss
3. Solution 2.1: DLinear (already exists)
4. Solution 4.1: Mixed Precision

### Round 2: Architecture Exploration
5. Solution 2.2: PI-DLinear
6. Solution 2.3: PatchTST
7. Solution 2.4: NLinear
8. Solution 3.1: Pattern-Based Spikes

### Round 3: Data & Training
9. Solution 3.2: Data Augmentation
10. Solution 3.3: Real Data
11. Solution 4.2: Gradient Accumulation
12. Solution 4.3: LR Optimization
13. Solution 4.4: Regularization

### Round 4: Foundation Models
14. Solution 5.1: Chronos-2
15. Solution 5.2: TimesFM

---

## Kaggle T4 Optimization Settings

### GPU Settings
```python
# T4 GPU: 16GB VRAM, 2560 CUDA cores, 320 Tensor cores
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.set_float32_matmul_precision('high')  # Use TF32
```

### Mixed Precision
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
with autocast():
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### DataLoader Settings
```python
DataLoader(
    dataset,
    batch_size=128,  # or 64 with gradient accumulation
    num_workers=4,   # Kaggle has 4 CPU cores
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
    shuffle=True
)
```

### Memory Management
```python
# Clear cache periodically
torch.cuda.empty_cache()
import gc; gc.collect()
```

### Heartbeat (Prevent Kaggle Timeout)
```python
import threading, time
def heartbeat():
    while True:
        time.sleep(300)  # Every 5 minutes
        print(f"  [heartbeat] {time.strftime('%H:%M:%S')}", flush=True)
threading.Thread(target=heartbeat, daemon=True).start()
```

---

## Success Criteria

| Metric | Current | Target | Priority |
|--------|---------|--------|----------|
| MAE | 5.12 MW | < 3 MW | P0 |
| SigAcc | 90.9% | > 95% | P0 |
| DirAcc | 50% | > 55% | P0 |
| Inference Latency | Unknown | < 100ms | P1 |
| Training Time | ~1 hour | < 2 hours | P2 |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Overfitting continues | Try DLinear, increase regularization |
| DirAcc stays at 50% | Try different loss functions, architectures |
| Kaggle timeout | Heartbeat mechanism, checkpoint saving |
| OOM errors | Reduce batch size, use gradient accumulation |
| Training too slow | Mixed precision, optimize DataLoader |
| Real data unavailable | Improve synthetic data with patterns |

---

## Notes
- All training on Kaggle with 2x T4 GPU
- Use heartbeat to prevent timeout
- Save checkpoints every epoch
- Monitor all metrics separately (not just total loss)
- Run each solution for full 80 epochs (patience=0)
- Compare against persistence baseline (MAE ≈ 3.9 MW)
