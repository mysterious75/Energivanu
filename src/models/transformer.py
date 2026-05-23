"""
ENERGIVANU — Time-Series Transformer
Dual branch: Time (Transformer) + Frequency (FFT) → Power + Signal
"""

import torch, torch.nn as nn, math
from typing import Tuple
from src.config import ModelConfig


class PatchEmbed(nn.Module):
    def __init__(self, nf, dm, ps=10):
        super().__init__()
        self.ps = ps
        self.proj = nn.Linear(nf*ps, dm)
        self.norm = nn.LayerNorm(dm)
    def forward(self, x):
        B,T,F = x.shape; np_ = T//self.ps
        x = x[:,:np_*self.ps,:].reshape(B,np_,F*self.ps)
        return self.norm(self.proj(x))


class GRN(nn.Module):
    def __init__(self, di, dh, do, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(di,dh); self.fc2 = nn.Linear(dh,do)
        self.gate = nn.Linear(di,do)
        self.skip = nn.Linear(di,do) if di!=do else nn.Identity()
        self.norm = nn.LayerNorm(do); self.d = nn.Dropout(drop)
    def forward(self, x):
        r = self.skip(x)
        h = self.d(torch.elu(self.fc1(x)))
        h = self.d(self.fc2(h))
        g = torch.sigmoid(self.gate(x))
        return self.norm(r + g*h)


class FreqBranch(nn.Module):
    def __init__(self, sl, dm):
        super().__init__()
        fb = sl//2+1
        self.p = nn.Sequential(nn.Linear(fb,dm), nn.GELU(), nn.Dropout(0.1))
    def forward(self, x):
        return self.p(torch.abs(torch.fft.rfft(x, dim=-1)))


class ColossusTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        np_ = cfg.lookback // cfg.patch_size

        self.patch = PatchEmbed(cfg.num_features, cfg.d_model, cfg.patch_size)
        self.pos = nn.Parameter(torch.randn(1, np_, cfg.d_model)*0.02)
        el = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout,
            batch_first=True, activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(el, cfg.n_layers)

        self.freq = FreqBranch(cfg.lookback, cfg.d_model) if cfg.use_freq else None
        self.fgate = nn.Linear(cfg.d_model*2, cfg.d_model) if cfg.use_freq else None

        self.grn = GRN(cfg.d_model, cfg.d_ff, cfg.d_model, cfg.dropout)
        self.phead = nn.Sequential(
            nn.Linear(cfg.d_model,cfg.d_ff), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff,cfg.d_ff//2), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff//2, cfg.horizon))
        self.shead = nn.Sequential(
            nn.Linear(cfg.d_model,cfg.d_ff//2), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff//2, cfg.n_classes))

        self._init()
        print(f"  Params: {sum(p.numel() for p in self.parameters()):,}")

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.patch(x) + self.pos
        h = self.enc(h).mean(1)
        if self.freq and self.fgate:
            hf = self.freq(x[:,:,0])
            g = torch.sigmoid(self.fgate(torch.cat([h,hf],-1)))
            h = g*h + (1-g)*hf
        h = self.grn(h)
        return self.phead(h), self.shead(h)
