
CODE ARCHITECTURE
=================

DATA PIPELINE:
  Config → Generator (30 days @ 5sec) → DataFrame (518K rows, 19 cols)
    → FeatureStore (34 features: rolling stats, deltas, stress, solar)
    → X(518K, 120, 34), Y(518K, 120), S(518K,) signal labels
    → Train/Val split (80/20) → DataLoader

MODEL ARCHITECTURE (ColossusTransformer):
  Input (batch, 120, 34)
    → PatchEmbed (patch_size=10 → 12 patches, Linear 340→128)
    → PositionalEncoding (sinusoidal)
    → TransformerEncoder ×3 layers (d_model=128, n_heads=4, d_ff=512)
    → Mean pooling over patches
    → FreqBranch (FFT features) + Adaptive fusion gate
    → GatedResidualNetwork (feature selection)
    → PowerHead (Linear → horizon=60 MW values)
    → SignalHead (Linear → 3 classes: SAFE/PREPARE/CRITICAL)

LOSS FUNCTION:
  SpikeLoss:
    - MSE with asymmetric weighting (under-predict spikes penalized 5x)
    - Dynamic threshold: mean + 1.5*std of batch
    - Weighted CrossEntropy for signal (classes weighted 1:2:5)
    - Total = power_loss + 0.5 * signal_loss

TRAINING:
  - AdamW optimizer with cosine LR schedule + warmup
  - Gradient clipping at 1.0
  - DataParallel for multi-GPU
  - Checkpoint save every epoch to /kaggle/working/
  - Auto-resume from latest checkpoint

INFERENCE:
  InferenceEngine class:
    - 5-second step cycle
    - SignalEngine generates SAFE/PREPARE/CRITICAL
    - SOC tracking (discharge 2% on CRITICAL, charge 0.5% on PREPARE)
    - Latency measurement

CO-OPTIMIZATION:
  CoOptimizer class:
    - Jobs with priority (CRIT/HIGH/MED/LOW)
    - DVFS throttling for low-priority jobs
    - Battery dispatch when grid import limit exceeded
    - Emergency actions for critical jobs
