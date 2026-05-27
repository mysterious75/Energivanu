"""
ENERGIVANU — TSMixer Model
All-MLP architecture for time series forecasting.
Simpler than Transformer, less overfitting, good cross-feature learning.

Reference: "TSMixer: An All-MLP Architecture for Time Series Forecasting" (Google, 2023)
"""

import torch, torch.nn as nn, torch.nn.functional as F
from typing import Tuple
from src.config import ModelConfig


class MixingBlock(nn.Module):
    """Temporal mixing + Feature mixing block."""
    def __init__(self, n_features, seq_len, d_ff, dropout=0.1):
        super().__init__()
        # Temporal mixing: mix across time steps
        self.temporal_norm = nn.LayerNorm(n_features)
        self.temporal_fc1 = nn.Linear(seq_len, d_ff)
        self.temporal_fc2 = nn.Linear(d_ff, seq_len)
        self.temporal_dropout = nn.Dropout(dropout)

        # Feature mixing: mix across features
        self.feature_norm = nn.LayerNorm(n_features)
        self.feature_fc1 = nn.Linear(n_features, d_ff)
        self.feature_fc2 = nn.Linear(d_ff, n_features)
        self.feature_dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, T, F) - batch, time, features
        """
        # Temporal mixing (transpose to mix across time)
        h = self.temporal_norm(x)
        h = h.transpose(1, 2)  # (B, F, T)
        h = F.gelu(self.temporal_fc1(h))
        h = self.temporal_dropout(h)
        h = self.temporal_fc2(h)
        h = h.transpose(1, 2)  # (B, T, F)
        x = x + h  # residual

        # Feature mixing
        h = self.feature_norm(x)
        h = F.gelu(self.feature_fc1(h))
        h = self.feature_dropout(h)
        h = self.feature_fc2(h)
        x = x + h  # residual

        return x


class TSMixer(nn.Module):
    """TSMixer for ENERGIVANU: power + signal + direction prediction.

    Architecture:
    - Input: (B, lookback, n_features)
    - N MixingBlocks (temporal + feature mixing)
    - Output heads: power (regression), signal (3-class), direction (binary)
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        nf = cfg.num_features
        lb = cfg.lookback
        hz = cfg.horizon

        # Mixing blocks
        self.blocks = nn.ModuleList([
            MixingBlock(nf, lb, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])

        # Global average pooling over time
        self.norm = nn.LayerNorm(nf)

        # Power head: predict horizon values
        self.phead = nn.Sequential(
            nn.Linear(nf, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, hz)
        )

        # Signal head: SAFE/PREPARE/CRITICAL
        self.shead = nn.Sequential(
            nn.Linear(nf, cfg.d_ff // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff // 2, cfg.n_classes)
        )

        # Direction head: UP/DOWN (binary classification)
        self.dir_head = nn.Sequential(
            nn.Linear(nf, cfg.d_ff // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff // 2, 2)
        )

        self._init()
        print(f"  TSMixer Params: {sum(p.numel() for p in self.parameters()):,}")

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, lookback, n_features)
        Returns:
            power: (B, horizon) predicted power
            signal: (B, 3) signal logits
            direction: (B, 2) direction logits
        """
        # Mixing blocks
        h = x
        for block in self.blocks:
            h = block(h)

        # Global pooling
        h = self.norm(h)
        h = h.mean(dim=1)  # (B, F)

        return self.phead(h), self.shead(h), self.dir_head(h)


class NLinear(nn.Module):
    """NLinear: Normalized Linear Model for Time Series.

    Simple linear model that normalizes input before linear layer.
    Handles distribution shift better than DLinear.

    Reference: "Are Transformers Effective for Time Series Forecasting?" (Zeng et al., 2023)
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        nf = cfg.num_features
        lb = cfg.lookback
        hz = cfg.horizon

        # Linear layer: lookback → horizon (per feature)
        self.linear = nn.Linear(lb, hz)

        # Signal head
        self.shead = nn.Sequential(
            nn.Linear(nf, 64),
            nn.ReLU(),
            nn.Linear(64, cfg.n_classes)
        )

        # Direction head
        self.dir_head = nn.Sequential(
            nn.Linear(nf, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        self._init()
        print(f"  NLinear Params: {sum(p.numel() for p in self.parameters()):,}")

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, lookback, n_features)
        """
        B, T, F = x.shape

        # Normalize: subtract last value (handles distribution shift)
        last_val = x[:, -1:, :]  # (B, 1, F)
        x_norm = x - last_val

        # Linear: (B, F, lookback) → (B, F, horizon)
        x_t = x_norm.transpose(1, 2)  # (B, F, T)
        pred_norm = self.linear(x_t)  # (B, F, horizon)

        # Denormalize: add last value back
        pred = pred_norm + last_val.transpose(1, 2)  # (B, F, horizon)

        # Power head: average across features → (B, horizon)
        pw = pred.mean(dim=1)

        # Signal and direction heads: pool over time
        x_pool = x.mean(dim=1)  # (B, F)

        return pw, self.shead(x_pool), self.dir_head(x_pool)
