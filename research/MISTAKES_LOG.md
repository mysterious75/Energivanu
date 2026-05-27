# ENERGIVANU — Mistakes Log (Research & Implementation)

> This file tracks every mistake encountered during research and implementation.
> Updated continuously as we work through solutions.

---

## Format
```
### MISTAKE #X — [SHORT TITLE]
**Date**: YYYY-MM-DD
**Context**: What were we doing?
**What happened**: What went wrong?
**Root cause**: Why did it happen?
**Fix**: How was it resolved?
**Lesson**: What to remember for future
```

---

## Pre-Existing Mistakes (from project history)

### MISTAKE #1 — Random Sync Spikes
**Context**: Added synchronized job-start events at random times
**What happened**: Random events are UNPREDICTABLE by ANY model → MAE doubled
**Root cause**: Synthetic data must have PATTERNS. Random ≠ realistic.
**Fix**: Use smooth data + lower signal thresholds instead
**Lesson**: Data must have learnable patterns, not random noise

### MISTAKE #2 — Large Batch Size
**Context**: Used batch=256, 512 thinking "more data per step = better"
**What happened**: Large batches converge to SHARP minima → poor val loss
**Root cause**: Batch size sweet spot is 32-64 for generalization
**Fix**: batch=128 as compromise (64/GPU with DataParallel)
**Lesson**: Small batches generalize better

### MISTAKE #3 — DataParallel for Small Models
**Context**: Always wrapped model in DataParallel if 2+ GPUs
**What happened**: For small models (<5M params), DP overhead hurts more than helps
**Fix**: Only use DP when batch/GPU ≥ 64
**Lesson**: DataParallel has overhead cost

### MISTAKE #4 — Early Stopping Without Monitoring
**Context**: patience=15
**What happened**: Training stopped at epoch 12-31, never saw full curve
**Fix**: patience=0 to run all epochs, inspect curve manually
**Lesson**: Sometimes val loss increases then decreases (double descent)

### MISTAKE #5 — Not Saving History
**Context**: history object was python variable in Colab
**What happened**: Colab crash → lost training curves
**Fix**: Always save to JSON/Drive at end of training
**Lesson**: Persist data to disk, not just memory

### MISTAKE #6 — Google Drive Mount on Kaggle
**Context**: Tried google.colab.drive.mount() on Kaggle
**What happened**: NotImplementedError
**Fix**: Use /kaggle/working/ locally + HF Hub for persistence
**Lesson**: Kaggle ≠ Colab for Drive mounting

### MISTAKE #7 — Not Deleting Old Data
**Context**: Re-ran training without deleting /kaggle/working/energivanu_prod
**What happened**: OSError (disk full) from 16GB X.npy + old data
**Fix**: Always rm -rf before fresh start on Kaggle
**Lesson**: Clean up before each fresh run

### MISTAKE #8 — torch.elu vs F.elu
**Context**: Used torch.elu() in GRN layer
**What happened**: AttributeError: module 'torch' has no attribute 'elu'
**Fix**: Changed to F.elu(x)
**Lesson**: torch.elu() doesn't exist. Use F.elu(x) or nn.ELU()

### MISTAKE #9 — Learned Positional Encoding
**Context**: Used nn.Parameter(randn) for positions
**What happened**: Cannot extrapolate to unseen sequence lengths
**Fix**: Sinusoidal PE (sin/cos) handles any sequence length
**Lesson**: Sinusoidal PE > learned PE for time series

### MISTAKE #10 — Lowering Thresholds Without Checking Data
**Context**: Changed signal thresholds from 85/70 to 55/45 MW
**What happened**: SigAcc collapsed from 90.9% to 68.3%
**Fix**: Keep thresholds at 85/70 OR increase data variance
**Lesson**: Thresholds must match data distribution

### MISTAKE #11 — Smooth Data Still Overfits
**Context**: dropout=0.35, wd=3e-4, batch=128
**What happened**: Val loss increased while train loss dropped
**Fix**: Add pattern-based event generator or replace with simpler model
**Lesson**: Smooth sinusoidal data is memorized, not learned

### MISTAKE #12 — Composite Loss Masking Signal Degradation
**Context**: SpikeLoss combines power MSE + signal CE into one total loss
**What happened**: MAE improved while SigAcc dropped
**Fix**: Split loss logging: track power_mae, sig_acc, dir_acc separately
**Lesson**: Composite loss hides conflicting trends

### MISTAKE #13 — Restoring Thresholds Without Verifying Data
**Context**: Restored thresholds to 85/70 MW from 55/45
**What happened**: CRITICAL count = 0 (max power was only 70 MW)
**Fix**: Increase ClusterConfig.num_gpus to 140K so max power = 98 MW
**Lesson**: Always verify data distribution matches thresholds

### MISTAKE #14 — Direction Loss on 5-Second Differences
**Context**: Used sign() then cosine_similarity on consecutive 5-second diffs
**What happened**: DirAcc stuck at 50%
**Fix**: Use stride=12 (1-minute windows) and F.mse_loss
**Lesson**: 5-second directional changes are essentially random

### MISTAKE #15 — SigAcc 99% Is Deceptive
**Context**: Celebrated SigAcc 98.9% without checking class distribution
**What happened**: CRITICAL events = 0, only SAFE + PREPARE exist
**Fix**: Ensure ALL 3 classes exist in training data
**Lesson**: High accuracy with missing classes means model learned nothing

### MISTAKE #16 — Loss Imbalance Between Power and Direction
**Context**: Used dir_w=10 with DirLoss ≈ 24 → 240 gradient vs pl ≈ 50
**What happened**: Direction gradient 5x power gradient → model optimized sign at cost of power
**Fix**: dir_w=3 gives 72 contribution vs 50 power → 59/41 split
**Lesson**: Measure both losses → set weights so contributions are equal

### MISTAKE #17 — Kaggle Inactivity Timeout
**Context**: Long training runs without user interaction
**What happened**: Kaggle auto-disconnects after 10-15 min of no interaction
**Fix**: Add daemon thread printing "[heartbeat] HH:MM:SS" every 5 min
**Lesson**: Training scripts need heartbeat mechanism

### MISTAKE #18 — Direction as Regression on Noisy Diffs
**Context**: Used MSE on stride=12 power differences to learn direction
**What happened**: DirAcc stuck at 50.8-51.0% across 65 epochs
**Fix**: Replace stride-based MSE with Direction Classification Head using BCE
**Lesson**: Direction is CLASSIFICATION, not regression. BCE > MSE for sign learning

---

## New Mistakes (from current research & implementation)

### MISTAKE #19 — Spike Detection Used Signal Target Instead of Power Target
**Date**: 2026-05-26
**Context**: SpikeLoss computed `sp = (tp > th).float()` where `tp` was actually `ts` (signal target)
**What happened**: Spike mask was based on signal class (0/1/2), not actual power values
**Root cause**: Variable naming confusion in loss function
**Fix**: Changed to `sp = (tp > th).float()` where `tp` is actual power target
**Lesson**: Always verify variable names match their semantic meaning

### MISTAKE #20 — Gradient Starvation in Multi-Task Loss
**Date**: 2026-05-26
**Context**: Power MSE (~15,000) dominated direction CE (~0.69) by 21,739:1 ratio
**What happened**: Direction head received negligible gradients, never learned
**Root cause**: Manual loss weighting (dir_w=5) was insufficient
**Fix**: Implemented Uncertainty Weighting (Kendall et al., CVPR 2018) - learned task weights
**Lesson**: Use automatic loss balancing for multi-task learning, not manual weights

### MISTAKE #21 — real_data.py Variable Name Bug
**Date**: 2026-05-26
**Context**: `lb_real` referenced but variable defined as `lb`
**What happened**: NameError when using real data source
**Fix**: Changed `lb_real` to `lb` on line 199
**Lesson**: Always test code paths, even if not frequently used

### MISTAKE #22 — Not Including Loss Function Parameters in Optimizer
**Date**: 2026-05-26
**Context**: Uncertainty weighting has learnable parameters (log_var_*)
**What happened**: Loss weights never updated during training
**Fix**: Added `list(self.loss_fn.parameters())` to optimizer
**Lesson**: When loss has learnable parameters, include them in optimizer

### MISTAKE #23 — torch.compile Conflicts with DataParallel
**Date**: 2026-05-26
**Context**: Applied torch.compile before DataParallel wrapping
**What happened**: AttributeError: 'TSMixer' object has no attribute 'blocks'
**Root cause**: torch.compile creates wrapper that doesn't expose module attributes when DataParallel replicates
**Fix**: Disabled torch.compile, removed from both kaggle_run.py and run_experiments.py
**Lesson**: torch.compile must be applied AFTER DataParallel, or use DDP instead

---

## Key Principles Learned
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
11. Smooth data = memorization, not learning
12. Composite loss hides conflicting trends
13. Direction is classification, not regression. BCE > MSE for sign learning
