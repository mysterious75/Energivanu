
CURRENT STATE — WHERE WE ARE
=============================

LATEST COMMIT: main 32df5af
  Message: "fix: batch=128 + dropout=0.35 + wd=3e-4 for anti-overfit"
  Date: May 23, 2026

CURRENT TRAINING CONFIG (kaggle_run.py):
  Data: 30 days, smooth generator, no sync spikes
  Thresholds: critical=55MW, warning=45MW
  Model: d_model=128, layers=3, heads=4, d_ff=512 (1M params)
  Training: batch=128, DataParallel, epochs=80, patience=0
  Regularization: dropout=0.35, weight_decay=3e-4, lr=1e-4

CURRENT RUN STATUS:
  Platform: Kaggle (2x T4 GPU)
  Data generation: Fresh (purana delete karna hoga)
  Training: NOT STARTED on this config yet (was running on older config)

LATEST RESULTS (from previous config with sync spikes):
  MAE: 9.90 MW (best at epoch 1!) 
  SigAcc: 70.3%
  DirAcc: 50.2% (random)
  Problem: Val loss never improved — overfit from start

EXPECTED RESULTS (current config):
  MAE: ~5-6 MW
  SigAcc: ~85-90%
  Critical events: ~30-40% (due to lower thresholds)
  Epoch time: ~45 seconds (batch=128, DataParallel)
  Total time: ~1 hour for 80 epochs

FILES ON DISK (repo):
  kaggle_run.py:              Training entrypoint (key file)
  src/config.py:              Config dataclasses
  src/data/generator.py:      Synthetic data (smooth + spikes options)
  src/data/features.py:       Feature engineering → 34 features
  src/models/transformer.py:  Model with PE, GRN, FreqBranch
  src/models/losses.py:       SpikeLoss with asymmetric weighting
  src/engine/trainer.py:       Training loop with save/resume
  src/engine/signal.py:       Signal engine for SAFE/PREPARE/CRITICAL
  src/engine/inference.py:    Production inference engine
  src/engine/scheduler.py:    Co-optimization scheduler
  docs/                       All documentation

GITHUB REPO: https://github.com/mysterious75/Energivanu
