
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
