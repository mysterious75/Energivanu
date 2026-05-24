"""
ENERGIVANU — Asymmetric Spike Loss
Under-predicting spikes = 5x penalty (grid damage risk)
"""

import torch, torch.nn as nn, torch.nn.functional as F
from typing import Tuple, Dict


class SpikeLoss(nn.Module):
    def __init__(self, uw=5.0, ow=1.0, ss=1.5, cw=0.5, dw=0.3):
        super().__init__()
        self.uw=uw; self.ow=ow; self.ss=ss; self.cw=cw; self.dw=dw

    def forward(self, pp, tp, ps, ts) -> Tuple[torch.Tensor, Dict]:
        err = tp - pp
        th = tp.mean() + self.ss*tp.std()
        sp = (tp>th).float()
        w = torch.where(err>0, self.uw+self.uw*sp, torch.tensor(self.ow,device=err.device))
        pl = (w*err.pow(2)).mean()

        pd = pp[:,1:]-pp[:,:-1]
        td = tp[:,1:]-tp[:,:-1]
        dl = F.mse_loss(pd.sign(), td.sign())

        cw = torch.tensor([1.,2.,5.], device=ps.device)
        sl = F.cross_entropy(ps, ts, weight=cw)

        total = pl + self.cw*sl + self.dw*dl
        with torch.no_grad():
            mae = err.abs().mean()
            da = (pd.sign()==td.sign()).float().mean()
            sa = (ps.argmax(-1)==ts).float().mean()
        return total, {"pl":pl.item(),"sl":sl.item(),"dl":dl.item(),"loss":total.item(),
                        "mae":mae.item(),"da":da.item(),"sa":sa.item()}
