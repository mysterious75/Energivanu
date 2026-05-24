
LESSONS LEARNED — MISTAKES TO NEVER REPEAT
===========================================

MISTAKE 1: RANDOM SYNC SPIKES
  What: Added "synchronized job-start" events at random times
  Why: Thought model needs to learn crisis events
  Result: Random events are UNPREDICTABLE by ANY model → MAE doubled
  Lesson: Synthetic data must have PATTERNS. Random ≠ realistic.
  Fix: Use smooth data + lower signal thresholds instead

MISTAKE 2: LARGE BATCH SIZE FOR GENERALIZATION
  What: Used batch=256, 512 thinking "more data per step = better"
  Why: Wanted GPU utilization high
  Result: Large batches converge to SHARP minima → poor val loss
  Lesson: Small batches (32-64) generalize better. Batch size sweet spot.
  Reference: "On Large-Batch Training for Deep Learning" (Keskar et al.)
  Fix: batch=128 as compromise (64/GPU with DataParallel)

MISTAKE 3: DataParallel ON BY DEFAULT
  What: Always wrapped model in DataParallel if 2+ GPUs
  Why: Wanted both T4s utilized
  Result: For small models (<5M params), DP overhead hurts more than helps
  Lesson: DataParallel worth it only when batch per GPU is 64+
  Fix: batch=128 total (64/GPU) makes DP worthwhile

MISTAKE 4: EARLY STOPPING WITHOUT MONITORING
  What: patience=15 (stops after 15 epochs no improvement)
  Why: Standard practice
  Result: Training stopped at epoch 12-31, never saw full curve
  Lesson: Sometimes val loss increases then decreases (double descent)
  Fix: patience=0 to run all epochs, inspect curve manually

MISTAKE 5: NOT SAVING HISTORY PROPERLY
  What: history object was python variable in Colab
  Why: Colab crash → lost training curves
  Result: Couldn't analyze val/train loss trends
  Lesson: Always save to JSON/Drive at end of training
  Fix: kaggle_run.py now saves history.json to checkpoint dir

MISTAKE 6: GOOGLE DRIVE MOUNT ON KAGGLE
  What: Tried google.colab.drive.mount() on Kaggle
  Why: Wanted persistent storage across sessions
  Result: NotImplementedError (Kaggle doesn't support colab.drive)
  Lesson: Kaggle ≠ Colab for Drive mounting
  Fix: Use /kaggle/working/ locally + HF Hub for model persistence

MISTAKE 7: NOT DELETING OLD DATA ON NEW SESSION
  What: Re-ran training without deleting /kaggle/working/energivanu_prod
  Why: Assumed overwrite
  Result: OSError (disk full) from 16GB X.npy + old data
  Lesson: Always rm -rf before fresh start on Kaggle
  Fix: Cell 1 now includes cleanup

MISTAKE 8: torch.elu INSTEAD OF F.elu
  What: Used torch.elu() in GRN layer
  Why: Assumed torch.elu exists like torch.relu
  Result: AttributeError: module 'torch' has no attribute 'elu'
  Lesson: torch.elu() doesn't exist. Use F.elu(x) or nn.ELU()
  Fix: Changed to F.elu(x)

MISTAKE 9: LEARNED POSITIONAL ENCODING
  What: Used nn.Parameter(randn) for positions
  Why: Simpler to implement
  Result: Cannot extrapolate to unseen sequence lengths
  Lesson: Sinusoidal PE (sin/cos) handles any sequence length
  Fix: Added PositionalEncoding class with fixed sin/cos frequencies

MISTAKE 10: LOWERING THRESHOLDS WITHOUT CHECKING DATA DISTRIBUTION
  What: Changed signal thresholds from 85/70 to 55/45 MW
  Why: Wanted more realistic critical event distribution
  Result: SigAcc collapsed from 90.9% to 68.3% (May 24 run)
  Detail: New thresholds (55/45) sit within average data range → borderline
          samples dominate → classifier confused → random guessing
  Lesson: Signal thresholds MUST match actual power distribution.
          Plot power histogram first, set thresholds at 90th/95th percentile.
          Don't lower thresholds artificially to "create" events.
  Fix: Keep thresholds at 85/70 OR increase data variance so natural peaks
       cross lower thresholds naturally.

MISTAKE 11: SMOOTH DATA STILL OVERFITS DESPITE REGULARIZATION
  What: dropout=0.35, wd=3e-4, batch=128 — still overfitting by ep10
  Why: Believed smooth data + regularization = no overfit
  Result: Val loss increased while train loss dropped (May 24 run)
          MAE 6.94 MW but still far from 3 MW target
  Evidence: Ep5 VL=117.03 → Ep10 VL=124.34 (⬆6%)
            Ep5 TL=125.17 → Ep10 TL=111.29 (⬇11%)
  Lesson: Smooth sinusoidal data is MEMORIZED, not LEARNED.
          Model fits training sine waves by rote → zero generalization.
          Data needs STRUCTURED COMPLEXITY (pattern-based events,
          realistic variance, scheduled spikes) not uniform smoothness.
  Fix: Add pattern-based event generator (not random, not smooth).
       Or: replace Transformer with simpler model (DLinear ~10K params).

MISTAKE 13: RESTORING THRESHOLDS WITHOUT VERIFYING DATA CAPABILITY
  What: Restored signal thresholds to 85/70 MW from 55/45
  Why: Wanted 90%+ SigAcc like Experiment 1
  Result: CRITICAL count = 0 (May 24 run, generation output confirmed)
  Detail: Max GPU power = 100K GPUs × (700W - 75W)/1e6 + 75W*100K/1e6
          = 62.5 MW + 7.5 MW = 70 MW MAX (with clipping at util=1.0)
          Threshold 85 MW is PHYSICALLY IMPOSSIBLE in current generator
  Lesson: After changing thresholds, ALWAYS check the data distribution.
          Run generator → plot power histogram → verify thresholds are
          within achievable range. Don't assume thresholds "worked before"
          without checking current data range.
  Fix: Increase ClusterConfig.num_gpus to 140K so max power = 98 MW.
       Add assertion in FeatureStore: check max value vs thresholds.

MISTAKE 14: DIRECTION LOSS ON 5-SECOND DIFFERENCES
  What: Used sign() then cosine_similarity on consecutive 5-second diffs
  Why: Wanted model to learn up/down direction of power changes
  Result: DirAcc stuck at 50% across all approaches
  Detail:
    - sign() → zero gradient (loss stayed at 1.0)
    - cosine_similarity → tiny diff values (~0.01-0.05 MW) numerically
      unstable; gradient overwhelmed by power loss (100x larger)
    - Power time series at 5-second level is mostly random walk noise
  Lesson: 5-second directional changes are essentially random in smooth
           power data. Direction loss must operate on MINUTE-level
           windows (stride=12 = 1 minute) to capture meaningful trends.
           MSE on differences works better than cosine_similarity for
           this use case because it provides stronger gradients.
  Fix: Use stride=12 (1-minute windows) and F.mse_loss(pd, td) instead
       of cosine_similarity on 5-second diffs.

MISTAKE 15: SigAcc 99% IS DECEPTIVE WITH ZERO CRITICAL EVENTS
  What: Celebrated SigAcc 98.9% without checking class distribution
  Why: Thought 99% accuracy means model is perfect
  Result: CRITICAL events = 0, only SAFE + PREPARE exist
          → Model is doing binary classification (SAFE vs PREPARE)
          → 98% SAFE, 2% PREPARE → SigAcc inflated by class imbalance
          → Model useless in production where CRITICAL spikes matter
  Lesson: Always check the distribution of signal classes (SAFE/PREPARE/
          CRITICAL) in the generated data. High accuracy with missing
          classes means the model learned NOTHING about crisis events.
          Accuracy on imbalanced data is misleading.
  Fix: Ensure ALL 3 classes exist in training data. Verify with
       "(S==0).sum() (S==1).sum() (S==2).sum()" before training.
       If CRITICAL=0, increase GPU count or data variance.

MISTAKE 12: COMPOSITE LOSS MASKING SIGNAL DEGRADATION
  What: SpikeLoss combines power MSE + signal CE into one total loss
  Why: Single loss number easier to monitor
  Result: MAE improved (8.87→6.94) while SigAcc dropped (71.3%→68.3%)
          Val loss increase was attributed to power head, not signal head
  Detail: Signal accuracy PEAKED at epoch 1 and never recovered.
          Classifier actively degraded as training progressed.
  Lesson: Always log power_loss and signal_loss SEPARATELY.
          A single "loss" metric hides conflicting trends.
          When signal weight (cls_w=0.5) is small, power dominates.
  Fix: Split loss logging: track power_mae, sig_acc, dir_acc separately.
       Increase cls_w if signal matters.

KEY PRINCIPLES LEARNED:
   1. Synthetic data must be REGULAR (patterns) or model fails
   2. Batch size 32-64 optimal, 256+ hurts generalization
   3. DataParallel only helps if batch/GPU ≥ 64
   4. Early stopping hides trends — run full epochs + manual inspect
   5. Always save history to disk (JSON), not just in memory
   6. Kaggle and Colab have different APIs for Drive auth
   7. Clean up /kaggle/working/ before each fresh run
   8. torch.nn.functional.elu(), not torch.elu()
   9. Sinusoidal PE > learned PE for time series
  10. Signal thresholds must match data distribution (90th/95th %ile)
  11. Smooth data = memorization, not learning. Need structured complexity.
  12. Composite loss hides conflicting trends. Always split power/signal logs.
