# ENERGIVANU

**AI-Powered GPU Power Forecasting for xAI's Colossus Supercomputer**

Predicts GPU cluster power demand 10 minutes ahead and signals Tesla Megapack batteries before grid instability occurs.

---

## Problem

xAI's Colossus cluster (150,000+ H100 GPUs) consumes up to 105 MW. When thousands of GPUs sync-start training jobs, power can swing 100+ MW in seconds. This causes:

- **Grid instability**: Gas turbines flicker, frequency deviations
- **Hardware damage**: $30K H100 chips fry from power surges
- **Revenue loss**: Training jobs interrupted, cluster downtime

Current solutions (Tesla Autobidder) are **reactive** — they respond to power changes after they happen. We need **predictive** forecasting to pre-position battery charge/discharge.

---

## Our Approach

### Architecture

```
Input (60 steps × 48 features)
    ↓
┌─────────────────────────────────┐
│  Autoformer Encoder             │
│  ├─ Series Decomposition        │  ← Separates trend + seasonal
│  ├─ Auto-Correlation (FFT)      │  ← Finds repeating patterns
│  └─ Feed-Forward Network        │
├─────────────────────────────────┤
│  Three Output Heads:            │
│  ├─ Power Head → 60 MW values   │  ← 10-min power forecast
│  ├─ Signal Head → SAFE/PREPARE/ │  ← Battery signal
│  │                CRITICAL       │
│  └─ Direction Head → UP/DOWN    │  ← Power trend
└─────────────────────────────────┘
    ↓
Battery Pre-positioning + Grid Stability
```

### Why Autoformer?

We tried 4 architectures:

| Model | Params | MAE | SigAcc | DirAcc | Notes |
|-------|--------|-----|--------|--------|-------|
| DLinear | 12K | 53 MW | 91% | — | Too simple, ignores 33/34 features |
| TSMixer | ~50K | 5 MW | 93% | — | Good baseline, all-MLP |
| Transformer | 1M | 3.57 MW | 93.3% | 57.3% | Best balance |
| **Autoformer (Colab)** | **851K** | **3.52 MW** | **94.0%** | **53.5%** | **All-time best MAE** |
| Autoformer (Kaggle, 60d, stride=2) | 851K | 4.54 MW | 88.9% | 53.5% | stride=2 cost ~1MW |
| Transformer (Kaggle, log_var explosion) | 1M | ~5 MW | ~87% | ~55% | W[1]→29.56, VL→32 |

Autoformer uses **series decomposition** (moving average to separate trend/seasonal) and **auto-correlation via FFT** instead of standard attention. This is better for time series because:
1. Power data has strong periodic patterns (daily cycles, scheduled jobs)
2. FFT efficiently captures these patterns at all time scales
3. Series decomposition prevents the model from confusing trend with seasonal

---

## Data Pipeline

### Synthetic Data (Current)

We generate realistic GPU cluster data:

- **GPU Power**: Sinusoidal base load (8h + 24h cycles) + scheduled training spikes + cascade events + micro-bursts
- **Weather**: Temperature, humidity, cloud cover, solar irradiance, wind
- **Grid**: Battery SOC, grid frequency, voltage, import power
- **Temporal**: Hour-of-day, day-of-week (with weekday/weekend modulation)

**60 days × 5-second intervals = 1,036,800 data points**

### Real Data (Planned)

MIT Supercloud cluster telemetry from AWS Open Data:
- 100ms sampling, per-job GPU CSV files
- Real power draw, utilization, temperature
- Stride=50 downsampling to match 5-second resolution

### Feature Engineering (48 features)

| Category | Features | Count |
|----------|----------|-------|
| Base | GPU power, load, temp, weather, grid, time | 18 |
| Rolling stats | Mean/std/range at 30s, 1m, 3m, 6m windows | 12 |
| Lag | Power/solar/grid at t-1 | 3 |
| Cross-feature | Solar/grid/battery rolling means | 3 |
| Delta | Rate of change for solar, grid, battery, temp | 4 |
| Cyclical time | sin/cos(hour), sin/cos(day-of-week) | 4 |
| Other | Acceleration, grid stress, solar availability, battery headroom | 4 |

---

## Target Metrics

| Metric | Target | Current Best | Status |
|--------|--------|--------------|--------|
| **MAE** | < 3.00 MW | 3.52 MW (Autoformer, Colab) | 🔴 19% off |
| **Signal Accuracy** | > 90% | 94.0% | ✅ |
| **Direction Accuracy** | > 55% | 57.3% (Transformer — marginal, likely noise) | 🟡 |
| **Inference Latency** | < 100ms | ~50ms | ✅ |

---

## Project Structure

```
energivanu/
├── colab_run.py              # Single-file auto-train for Colab
├── kaggle_run.py             # Training script for Kaggle (2× T4)
├── src/
│   ├── config.py             # All configuration dataclasses
│   ├── data/
│   │   ├── generator.py      # Synthetic data generator (v2)
│   │   ├── features.py       # Feature engineering (v2, 48 features)
│   │   └── real_data.py      # MIT Supercloud data loader
│   ├── models/
│   │   ├── transformer.py    # ColossusTransformer (1M params)
│   │   ├── autoformer.py     # Autoformer with FFT auto-correlation
│   │   ├── dlinear.py        # DLinear (baseline)
│   │   ├── tsmixer.py        # TSMixer + NLinear
│   │   └── losses.py         # SpikeLoss with uncertainty weighting
│   └── engine/
│       ├── trainer.py        # Training loop (AMP, GradClip, heartbeat)
│       ├── signal.py         # Battery signal engine
│       └── inference.py      # Production inference
├── docs/                     # Detailed documentation
│   ├── 00_PROJECT_OVERVIEW.md
│   ├── 02_EXPERIMENTS_LOG.md
│   ├── 03_LESSONS_LEARNED.md
│   └── 04_CURRENT_STATE.md
├── CONTRIBUTING.md           # Contribution guidelines (optional)
└── tests/                    # Test suite
```

---

## Quick Start

### Google Colab (Recommended)

```python
# Single cell — auto-clones, generates data, trains
!wget -O /content/colab_run.py "https://raw.githubusercontent.com/mysterious75/Energivanu/main/colab_run.py?$(date +%s)"
!python /content/colab_run.py
```

### Kaggle

```python
# Cell 1 — clone fresh (use for new Kaggle session)
!git clone https://github.com/mysterious75/Energivanu.git
%cd /kaggle/working/Energivanu
!pip install -r requirements.txt

# Cell 2
!python kaggle_run.py
```

### Configuration

Edit `colab_run.py` or `kaggle_run.py` to change:

```python
MODEL_TYPE = "autoformer"    # transformer | autoformer | dlinear | tsmixer
DAYS = 60                    # Days of synthetic data
EPOCHS = 120                 # Training epochs
LR = 1e-4                   # Learning rate
BATCH_SIZE = 256             # Batch size (reduce if OOM)
```

---

## Key Learnings

1. **Architecture < Data Quality**: Both Transformer and Autoformer plateau at ~3.5 MW. The bottleneck is data diversity, not model capacity.

2. **FFT + AMP = careful**: cuFFT requires power-of-2 dimensions in FP16. Cast to FP32 for FFT operations.

3. **Stride > 1 destroys temporal patterns**: For 5-second interval data, stride=2 costs ~1MW MAE. Every timestep matters.

4. **Memory > compute on Colab**: The bottleneck is RAM (12GB), not GPU. Design data pipelines to fit: stride=2 + 48 features = ~5.9GB.

5. **Early stopping bugs are subtle**: Always use `if/elif/else` chains, not separate `if` blocks.

6. **Uncertainty weighting is fragile**: Kendall uncertainty weighting can destabilize training when loss scales differ. `log_var` clamp [-5, 5] insufficient — W[1] reached 29.56. Fixed weights are safer.

7. **Direction accuracy is a red herring**: Both models stuck at 53-57% on synthetic data with 5-second labels. This is a data labeling issue, not a model issue.

---

## Roadmap

1. **More diverse training data** — Increase spike frequency variation, add more cascade patterns
2. **Real MIT Supercloud data** — Train on actual cluster telemetry
3. **Larger model** — d_model=256, n_layers=6 (4M params) to see if capacity helps
4. **Informer** — Long-sequence attention for longer lookback windows
5. **Co-optimization** — Integrate with workload scheduler for proactive GPU throttling

---

## Team

- **vedkumar75** — Data pipeline, feature engineering, training
- **opencode (big-pickle)** — Model architecture, debugging, optimization
- **Platform**: Windows laptop + Kaggle/Colab (T4 GPU)
- **Repo**: [github.com/mysterious75/Energivanu](https://github.com/mysterious75/Energivanu)
