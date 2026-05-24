import torch, torch.nn as nn, torch.nn.functional as F
from typing import Tuple
from src.config import ModelConfig


class DLinear(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        nf = cfg.num_features
        lb = cfg.lookback
        hz = cfg.horizon
        inp = lb * nf

        self.kernel = 25
        self.pad = self.kernel // 2
        self.avg = nn.AvgPool1d(self.kernel, 1, padding=self.pad)

        self.trend = nn.Linear(inp, hz)
        self.seasonal = nn.Linear(inp, hz)
        self.shead = nn.Linear(inp, cfg.n_classes)

        self._init()
        print(f"  DLinear Params: {sum(p.numel() for p in self.parameters()):,}")

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, F = x.shape
        x_flat = x.reshape(B, T * F)

        x_t = x_flat.unsqueeze(1)
        trend_part = self.avg(x_t).squeeze(1)
        seasonal_part = x_flat - trend_part

        power = self.trend(trend_part) + self.seasonal(seasonal_part)
        signal = self.shead(x_flat)
        return power, signal
