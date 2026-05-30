"""
ENERGIVANU — Autoformer
Series decomposition + auto-correlation for time series forecasting.
Paper: Autoformer (Wu et al., NeurIPS 2021)
"""

import torch, torch.nn as nn, torch.nn.functional as F, math
from typing import Tuple
from src.config import ModelConfig


class SeriesDecomp(nn.Module):
    """Moving average decomposition into trend + seasonal."""
    def __init__(self, kernel_size=25):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size, stride=1, padding=padding, count_include_pad=False)

    def forward(self, x):
        trend = self.avg(x)
        return trend, x - trend


class AutoCorrelation(nn.Module):
    """Auto-correlation with top-k time-delay aggregation.

    Uses FFT to compute autocorrelation efficiently.
    Aggregates values at top-k time delays weighted by correlation strength.
    """
    def __init__(self, d_model, n_heads, top_k=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.top_k = top_k
        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        B, T, D = q.shape
        dh = D // self.n_heads

        Q = self.q_proj(q).reshape(B, T, self.n_heads, dh).permute(0, 2, 1, 3)
        K = self.k_proj(k).reshape(B, T, self.n_heads, dh).permute(0, 2, 1, 3)
        V = self.v_proj(v).reshape(B, T, self.n_heads, dh).permute(0, 2, 1, 3)

        Q_fft = torch.fft.rfft(Q, dim=2)
        K_fft = torch.fft.rfft(K, dim=2)
        S = Q_fft * torch.conj(K_fft)
        R = torch.fft.irfft(S, n=T, dim=2)
        scores = R.sum(dim=-1).sum(dim=1)

        scores_delays = scores[:, 1:]
        weights, indices = torch.topk(scores_delays, self.top_k, dim=-1)
        weights = F.softmax(weights / math.sqrt(self.d_model), dim=-1)
        indices = indices + 1

        delays = indices.float().mean(dim=0).long()
        out = torch.zeros_like(V)
        for i in range(self.top_k):
            d = int(delays[i].item())
            if d == 0:
                d = 1
            v_rolled = torch.roll(V, shifts=-d, dims=2)
            w = weights[:, i, None, None, None]
            out = out + w * v_rolled

        out = out.permute(0, 2, 1, 3).reshape(B, T, D)
        return self.dropout(self.o_proj(out))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, top_k, dropout, decomp_kernel):
        super().__init__()
        self.decomp1 = SeriesDecomp(decomp_kernel)
        self.decomp2 = SeriesDecomp(decomp_kernel)
        self.auto_corr = AutoCorrelation(d_model, n_heads, top_k, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = x.transpose(1, 2)
        trend1, seasonal1 = self.decomp1(x)
        x = seasonal1.transpose(1, 2)
        x = self.auto_corr(x, x, x)
        x = (x + residual).transpose(1, 2)
        _, seasonal2 = self.decomp2(x)
        x = seasonal2.transpose(1, 2)
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        return x + residual


class AutoformerModel(nn.Module):
    """Autoformer for GPU power forecasting.

    Encoder-only architecture with:
    - Input projection: (B, T, F) -> (B, T, d_model)
    - N EncoderLayers: SeriesDecomp + AutoCorrelation + FF
    - Output heads: power (B, H), signal (B, 3), direction (B, 2)
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        nf = cfg.num_features
        dm = cfg.d_model
        nh = cfg.n_heads
        nl = cfg.n_layers
        df = cfg.d_ff
        dp = cfg.dropout
        tk = getattr(cfg, 'top_k', 3)
        dk = getattr(cfg, 'decomp_kernel', 25)

        self.input_proj = nn.Linear(nf, dm)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(dm, nh, df, tk, dp, dk) for _ in range(nl)
        ])

        self.norm = nn.LayerNorm(dm)

        self.phead = nn.Sequential(
            nn.Linear(dm, df), nn.GELU(), nn.Dropout(dp),
            nn.Linear(df, df // 2), nn.GELU(), nn.Dropout(dp),
            nn.Linear(df // 2, cfg.horizon),
        )
        self.shead = nn.Sequential(
            nn.Linear(dm, df // 2), nn.GELU(), nn.Dropout(dp),
            nn.Linear(df // 2, cfg.n_classes),
        )
        self.dir_head = nn.Linear(dm, 2)

        self._init()
        print(f"  Autoformer Params: {sum(p.numel() for p in self.parameters()):,}")

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        for layer in self.encoder_layers:
            h = layer(h)
        h = self.norm(h)
        pooled = h.mean(dim=1)
        return self.phead(pooled), self.shead(pooled), self.dir_head(pooled)
