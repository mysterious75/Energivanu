"""
ENERGIVANU — Trainer with warmup + cosine LR + early stopping
"""

import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np, time
from pathlib import Path
from src.config import Config
from src.models.transformer import ColossusTransformer
from src.models.losses import SpikeLoss


class Trainer:
    def __init__(self, model, cfg: Config):
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.dev)
        self.cfg = cfg
        self.loss_fn = SpikeLoss(cfg.train.under_w, cfg.train.over_w,
                                  cfg.train.spike_std, cfg.train.cls_w)
        self.opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                      weight_decay=cfg.train.weight_decay, betas=(0.9,0.98))
        self.step = 0
        self.hist = {"tl":[],"vl":[],"vm":[],"vs":[]}

    def _lr(self, cur, total):
        c = self.cfg.train
        if cur < c.warmup: return c.lr * cur / max(c.warmup,1)
        p = (cur-c.warmup)/max(total-c.warmup,1)
        return c.lr * 0.5 * (1+np.cos(np.pi*p))

    def _epoch(self, dl, train, total=0):
        self.model.train() if train else self.model.eval()
        s = {k:0. for k in ["pl","sl","loss","mae","da","sa"]}; n=0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for x,yp,ys in dl:
                x,yp,ys = x.to(self.dev),yp.to(self.dev),ys.to(self.dev)
                if train:
                    self.step+=1
                    for g in self.opt.param_groups:
                        g["lr"]=self._lr(self.step,total)
                    self.opt.zero_grad()
                pp,ps = self.model(x)
                l,m = self.loss_fn(pp,yp,ps,ys)
                if train:
                    l.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
                    self.opt.step()
                for k in s: s[k]+=m[k]
                n+=1
        return {k:v/max(n,1) for k,v in s.items()}

    def fit(self, Xtr,Ytr,Str,Xvl,Yvl,Svl):
        tc = self.cfg.train
        tds = TensorDataset(torch.FloatTensor(Xtr),torch.FloatTensor(Ytr),torch.LongTensor(Str))
        vds = TensorDataset(torch.FloatTensor(Xvl),torch.FloatTensor(Yvl),torch.LongTensor(Svl))
        tdl = DataLoader(tds, tc.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        vdl = DataLoader(vds, tc.batch_size, shuffle=False, num_workers=0, pin_memory=True)

        total = len(tdl)*tc.epochs; best=float("inf"); wait=0
        Path("checkpoints").mkdir(exist_ok=True)

        print(f"\n  Device: {self.dev} | Train: {len(tds):,} | Val: {len(vds):,}")
        print(f"  Batch: {tc.batch_size} | Epochs: {tc.epochs}\n")

        for ep in range(1,tc.epochs+1):
            t0=time.time()
            tm=self._epoch(tdl,True,total)
            vm=self._epoch(vdl,False)
            dt=time.time()-t0

            self.hist["tl"].append(tm["loss"])
            self.hist["vl"].append(vm["loss"])
            self.hist["vm"].append(vm["mae"])
            self.hist["vs"].append(vm["sa"])

            if ep%5==0 or ep==1:
                print(f"  Ep {ep:3d}/{tc.epochs} | TL:{tm['loss']:.4f} VL:{vm['loss']:.4f} "
                      f"| MAE:{vm['mae']:.2f}MW SigAcc:{vm['sa']:.3f} "
                      f"DirAcc:{vm['da']:.3f} LR:{self.opt.param_groups[0]['lr']:.2e} {dt:.1f}s")

            if vm["loss"]<best:
                best=vm["loss"]; wait=0
                torch.save({"ep":ep,"model":self.model.state_dict(),
                            "opt":self.opt.state_dict(),"vl":best},
                           "checkpoints/best.pt")
            else:
                wait+=1
                if wait>=tc.patience:
                    print(f"\n  Early stop at {ep}"); break

        print(f"\n  Best val loss: {best:.4f}")
        return self.hist
