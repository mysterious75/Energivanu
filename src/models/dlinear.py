import torch, torch.nn as nn, torch.nn.functional as F
from typing import Tuple
from src.config import ModelConfig


class DLinear(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        nf = cfg.num_features
        lb = cfg.lookback
        hz = cfg.horizon

        self.kernel = 25
        self.pad = self.kernel // 2

        self.trend_l = nn.Linear(lb, hz)
        self.seasonal_l = nn.Linear(lb, hz)
        self.shead = nn.Sequential(nn.Linear(lb, 64), nn.ReLU(), nn.Linear(64, cfg.n_classes))
        self.dir_head = nn.Sequential(nn.Linear(lb, 64), nn.ReLU(), nn.Linear(64, 2))

        self._init()
        print(f"  DLinear Params: {sum(p.numel() for p in self.parameters()):,}")

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, F = x.shape

        x_t = x.transpose(1, 2)
        x_p = F.pad(x_t, (self.pad, self.pad), mode="reflect")
        trend = F.avg_pool1d(x_p, self.kernel, stride=1)
        seasonal = x_t - trend

        pp = self.trend_l(trend) + self.seasonal_l(seasonal)
        pw = pp[:, 0, :]

        return pw, self.shead(x[:, :, 0]), self.dir_head(x[:, :, 0])
