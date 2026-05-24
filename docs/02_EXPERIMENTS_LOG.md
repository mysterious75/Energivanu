
EXPERIMENTS LOG — EVERYTHING WE TRIED
======================================

EXPERIMENT 1 — FIRST WORKING MODEL
  Config:
    days=15, d_model=128, layers=3, heads=4, d_ff=512
    lookback=60, horizon=60, batch=16, epochs=30
    lr=1e-4, wd=1e-5, dropout=0.1
  Data: Original smooth generator (no sync spikes)
  Result: MAE=5.12, SigAcc=90.9%, DirAcc=0.50
  Problem: CRITICAL count = 0 (thresholds 85/70 too high for data)
  Verdict: ✅ BEST RUN SO FAR, but no crisis events learned

EXPERIMENT 2 — LARGE MODEL (Kaggle, T4x2)
  Config:
    days=30, d_model=256, layers=6, heads=8, d_ff=1024
    lookback=120, horizon=120, batch=16, epochs=60
    lr=5e-5, wd=1e-4, dropout=0.2
  Data: Original smooth generator
  Result: Val loss kept increasing, MAE=5.62 (epoch 10 best)
  Problem: Overfitting from epoch 1, early stop at epoch 12
  Verdict: ❌ Large model on limited data overfits

EXPERIMENT 2b — LARGE MODEL + MORE DATA
  Config: Same as exp2, data=30 days
  Result: Same overfitting. 414K samples not enough for 6.5M params
  Verdict: ❌ Needs better regularization

EXPERIMENT 3 — LARGE MODEL + REGULARIZATION
  Config:
    days=30, d_model=128, layers=3, heads=4, d_ff=512
    lookback=60, horizon=60, batch=16, epochs=60
    lr=5e-5, wd=1e-4, dropout=0.2, patience=15
  Data: Original smooth generator
  Result: MAE=5.12 best, Val loss still increasing
  Problem: Overfitting at smaller scale too
  Verdict: ⚠️ Data is too predictable (smooth), model memorizes noise

EXPERIMENT 4 — SYNC SPIKES ADDED
  Config: Same as exp3
  Data: Generator with synchronized job-start spikes (0.4-0.6 util)
  Result: MAE=10.83, SigAcc=70.8%
  Problem: Data became UNPREDICTABLE (random spike timing/duration)
  Verdict: ❌ Model can't learn random events

EXPERIMENT 4b — LARGE BATCH (512) + SYNC SPIKES
  Config:
    batch=512, d_model=192, layers=4, d_ff=768
    Data: Sync spikes
  Result: MAE=13.14, SigAcc=70.0%, epoch time=38s
  Problem: Large batch → sharp minima → poor generalization
  Verdict: ❌ batch=512 hurts generalization

EXPERIMENT 5 — SMALL BATCH (64) + ANTIOVERFIT + SYNC SPIKES
  Config:
    batch=64, d_model=256, layers=6, dropout=0.4
  Data: Sync spikes
  Result: MAE=9.90, SigAcc=70.3%, epoch time=156s
  Problem: Val loss still increasing from epoch 1
  Verdict: ❌ Model still can't learn random sync spikes

EXPERIMENT 6 — SMOOTH DATA + LOWER THRESHOLDS
  Config:
    days=30, d_model=128, layers=3, heads=4, d_ff=512
    lookback=60, horizon=60, batch=128, epochs=80
    lr=1e-4, wd=3e-4, dropout=0.35
    DataParallel ON (2x T4), patience=0, thresholds=55/45
  Data: Original smooth generator
  Result: MAE=6.94@ep10, SigAcc=68.7%, DirAcc=50.2%
  Problem: Overfitting from ep10, SigAcc worse than ep1 (71.3%)
  Verdict: ❌ Smooth data memorized, low thresholds confused classifier

EXPERIMENT 7 — PATTERN SPIKES + THRESHOLDS 85/70 [COMPLETED]
  Config:
    days=30, d_model=128, layers=3, heads=4, d_ff=512
    lookback=60, horizon=60, batch=128, epochs=80
    lr=1e-4, wd=3e-4, dropout=0.35, dir_w=0.3
    DataParallel ON (2x T4), patience=0, thresholds=85/70
  Data: Pattern-based spikes (every 4hr + cascading failures)
  Loss: SpikeLoss + DirectionLoss sign() [BROKEN — zero gradient]
  Result: MAE=2.43@ep50, SigAcc=99.1%, DirAcc=50.1%
  Verdict: ✅ MAE target achieved! SigAcc 99%! But DirAcc still
           random because torch.sign() has zero gradient.
           Direction loss was NOT backpropagating.
  Fix: Replace sign() with cosine_similarity for differentiable DirLoss

EXPERIMENT 7b — DIFFERENTIABLE DIRECTION LOSS [COMPLETED]
  Config: Same as Exp7 + dir_w=10
  Loss: SpikeLoss + DirectionLoss (cosine_similarity, stride=1)
  Result: MAE=2.57@ep25, SigAcc=98.9%, DirAcc=50.1%
  Problem:
    - DirAcc still 50% (cosine_similarity gradient too small)
    - CRITICAL count = 0 (thresholds 85/70 impossible with 100K GPUs)
    - SigAcc 98.9% is fake (no CRITICAL class, binary problem)
  Verdict: ⚠️ Three separate bugs found: DirLoss wrong, GPUs too few,
           threshold mismatch. All fixed in Exp7c.

EXPERIMENT 7c — ALL FIXES: 140K GPUs + MSE DIRLOSS + DIR_W=30 [CURRENT]
  Config:
    days=30, d_model=128, layers=3, heads=4, d_ff=512
    lookback=60, horizon=60, batch=128, epochs=80
    lr=1e-4, wd=3e-4, dropout=0.35, dir_w=30, stride=12
    DataParallel ON (2x T4), patience=0, thresholds=85/70
    num_gpus=140,000 (max power 98 MW → CRITICAL events possible)
  Data: Pattern-based spikes + 140K GPUs
  Loss: SpikeLoss + DirectionLoss (MSE on 1-min differences, stride=12)
  Status: READY TO RUN
  Expected: MAE ~3-4 MW, SigAcc ~90-95% (all 3 classes), DirAcc >55%
  Verdict: 🏁 CURRENT BEST APPROACH

GPU UTILIZATION EXPERIMENTS:
  - batch=16, 1 GPU → GPU 30%, epoch 9min
  - batch=256, 1 GPU → GPU 80%, epoch 48s  
  - batch=256, DataParallel 2 GPU → each 60%, epoch 38s
  - batch=64, 1 GPU → GPU 50%, epoch 160s
  - batch=128, DataParallel 2 GPU → each 50-60%, epoch 45s [CURRENT]
