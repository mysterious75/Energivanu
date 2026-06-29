# Model Architecture — EnergivanuPEB

## Overview
TCN (Temporal Convolutional Network) + Multi-Head Attention for predictive energy buffering.

## Best Run Configuration
- **Parameters:** 613,612
- **Input:** (batch, 30, 15) — 30 timesteps, 15 features
- **Output:** Power prediction (batch, 10) + Signal classification (batch, 3)

## Architecture
```
Input (B, 30, 15)
  → Adaptive Domain Normalization (3 LayerNorm groups)
    - Power features (indices 0-6): LayerNorm(7)
    - Telemetry features (indices 7-13): LayerNorm(7)
    - Temporal features (indices 14): LayerNorm(1)
  → Input Projection: Linear(15 → 128)
  → TCN Backbone:
    - TemporalBlock(128→32, kernel=5, dilation=1)
    - TemporalBlock(32→64, kernel=3, dilation=2)
    - TemporalBlock(64→128, kernel=3, dilation=4)
  → Multi-Head Attention (8 heads, dim=128)
  → Gated Aggregation:
    - alpha = sigmoid(linear(last_step))
    - output = alpha * last_step + (1-alpha) * mean_pool
  → Power Head: Linear(128→256→128→10)
  → Signal Head: Linear(140→256→128→3)
```

## TemporalBlock
Each block has:
- Two dilated causal conv1d layers
- LayerNorm (not BatchNorm — works with batch_size=1)
- ReLU activation
- Dropout (0.1)
- Residual connection

## Training Config
| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-3 |
| Weight Decay | 1e-4 |
| Gradient Clipping | 1.0 |
| Optimizer | AdamW |
| Scheduler | CosineAnnealing (T_max=200, eta_min=1e-5) |
| Power Loss | HuberLoss |
| Signal Loss | CrossEntropyLoss |
| Loss Weight | power + 0.3 * signal |
| Batch Size | 512 |
| Early Stopping | patience=25 |

## Versions

| Version | Params | Val Loss | MAPE |
|---------|--------|----------|------|
| v1 (default config) | 338,252 | 34.59 | 37% |
| **v2 (larger channels)** | **613,612** | **5.95** | **21%** |
