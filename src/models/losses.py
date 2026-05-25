"""
ENERGIVANU — Asymmetric Spike Loss
Under-predicting spikes = 5x penalty (grid damage risk)
Direction: binary classification (up/down) instead of MSE on diffs
"""

import torch, torch.nn as nn, torch.nn.functional as F
from typing import Tuple, Dict


class SpikeLoss(nn.Module):
    def __init__(self, uw=5.0, ow=1.0, ss=1.5, cw=1.0, dw=5.0):
        super().__init__()
        self.uw=uw; self.ow=ow; self.ss=ss; self.cw=cw; self.dw=dw

    def forward(self, pp, tp, ps, ts, pdir, tdir) -> Tuple[torch.Tensor, Dict]:
        err = tp - pp
        th = tp.mean() + self.ss*tp.std()
        sp = (tp>th).float()
        w = torch.where(err>0, self.uw+self.uw*sp, torch.tensor(self.ow,device=err.device))
        pl = (w*err.pow(2)).mean()

        dl = F.cross_entropy(pdir, tdir)

        cw = torch.tensor([1.,2.,5.], device=ps.device)
        sl = F.cross_entropy(ps, ts, weight=cw)

        total = pl + self.cw*sl + self.dw*dl
        with torch.no_grad():
            mae = err.abs().mean()
            da = (pdir.argmax(-1)==tdir).float().mean()
            sa = (ps.argmax(-1)==ts).float().mean()
        return total, {"pl":pl.item(),"sl":sl.item(),"dl":dl.item(),"loss":total.item(),
                        "mae":mae.item(),"da":da.item(),"sa":sa.item()}
