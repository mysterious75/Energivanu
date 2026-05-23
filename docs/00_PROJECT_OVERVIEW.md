
ENERGIVANU — COMPLETE PROJECT DOCUMENTATION
=============================================

PROJECT: ENERGIVANU (AI Energy Management System)
==================================================
GOAL: Predict GPU power demand 10 minutes ahead for xAI's Colossus
      (100K+ H100 GPUs) and signal Tesla Megapack batteries BEFORE
      grid instability occurs.

TEAM:
  - User: vedkumar755 (Colab Gmail), mysterious75 (GitHub)
  - Model: opencode (big-pickle)
  - Platform: Windows laptop (8GB RAM) + Kaggle (2x T4 GPU)
  - Repo: https://github.com/mysterious75/Energivanu

THREE CORE PROBLEMS (from research):
  1. POWER SPIKES: 100K GPUs sync-start → 100+ MW swings in seconds
     → Gas turbines flicker → $30K H100 chips fry
  2. NO CO-OPTIMIZATION: xAI + Tesla MegaPack = separate systems
     Autobidder is REACTIVE (minute-level), not predictive
  3. WEATHER NOWCASTING: Solar/wind unpredictable at 60-second level

OUR SOLUTION:
  Transformer model that:
  - Takes 10 min of GPU/weather/grid data (60 steps × 34 features)
  - Predicts power demand 10 min ahead (60 steps of MW values)
  - Generates battery signal: SAFE / PREPARE / CRITICAL
  - Co-optimizes workloads via DVFS + scheduling

TARGET METRICS:
  - MAE < 3 MW (< 5% error at 60MW avg)  
  - Signal Accuracy > 90%
  - Direction Accuracy > 55% (better than random 50%)
  - Inference latency < 100ms per prediction

CURRENT BEST RESULTS:
  - MAE: 5.12 MW (smooth data, small model)
  - MAE: ~10 MW (harder data, large model) — still improving
  - SigAcc: 90.9% (best)
  - DirAcc: ~50% (random — NEEDS IMPROVEMENT)

KEY FILES:
  kaggle_run.py          — Single-script training with auto-resume
  src/config.py          — All configuration dataclasses
  src/data/generator.py  — Synthetic data generator
  src/data/features.py   — Feature engineering (34 features)
  src/models/transformer.py — Transformer model (PE, GRN, FreqBranch)
  src/models/losses.py   — Asymmetric SpikeLoss
  src/engine/trainer.py  — Training loop (LR warmup, cosine, saving)
  src/engine/signal.py   — Battery SignalEngine
  src/engine/inference.py — Production inference engine
  src/engine/scheduler.py — Co-optimization scheduler
