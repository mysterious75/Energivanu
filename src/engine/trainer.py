"""
ENERGIVANU — Trainer with warmup + cosine LR + early stopping
"""

import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np, time
from pathlib import Path
from src.config import Config
from src.models.transformer import ColossusTransformer
from src.models.dlinear import DLinear
from src.models.losses import SpikeLoss


class Trainer:
    def __init__(self, model, cfg: Config, y_mean=0.0, y_std=1.0, use_dp=True, num_workers=4):
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.dev)
        self.model_raw = model
        self.is_parallel = False
        if use_dp and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            self.is_parallel = True
        self.cfg = cfg
        self.y_mean = y_mean
        self.y_std = y_std
        self.num_workers = num_workers
        self.loss_fn = SpikeLoss(cfg.train.under_w, cfg.train.over_w,
                                  cfg.train.spike_std, cfg.train.cls_w,
                                  cfg.train.dir_w)
        self.opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                      weight_decay=cfg.train.weight_decay, betas=(0.9,0.98))
        self.step = 0
        self.hist = {"tl":[],"vl":[],"vm":[],"vs":[],"vd":[]}

    def _lr(self, cur, total):
        c = self.cfg.train
        if cur < c.warmup: return c.lr * cur / max(c.warmup,1)
        p = (cur-c.warmup)/max(total-c.warmup,1)
        return c.lr * 0.5 * (1+np.cos(np.pi*p))

    def _epoch(self, dl, train, total=0):
        self.model.train() if train else self.model.eval()
        s = {k:0. for k in ["pl","sl","dl","loss","mae","da","sa"]}; n=0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for x,yp,ys,yd in dl:
                x,yp,ys,yd = x.to(self.dev),yp.to(self.dev),ys.to(self.dev),yd.to(self.dev)
                if train:
                    self.step+=1
                    for g in self.opt.param_groups:
                        g["lr"]=self._lr(self.step,total)
                    self.opt.zero_grad()
                pp,ps,pd = self.model(x)
                l,m = self.loss_fn(pp,yp,ps,ys,pd,yd)
                if train:
                    l.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
                    self.opt.step()
                for k in s: s[k]+=m[k]
                n+=1
        return {k:v/max(n,1) for k,v in s.items()}

    def fit(self, Xtr, Ytr, Str, Dtr, Xvl, Yvl, Svl, Dvl, drive_dir=None, save_every=5, resume_from=0):
        tc = self.cfg.train
        tds = TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(Ytr),
                            torch.LongTensor(Str), torch.LongTensor(Dtr))
        vds = TensorDataset(torch.FloatTensor(Xvl), torch.FloatTensor(Yvl),
                            torch.LongTensor(Svl), torch.LongTensor(Dvl))
        tdl = DataLoader(tds, tc.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, prefetch_factor=2)
        vdl = DataLoader(vds, tc.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, prefetch_factor=2)

        total = len(tdl)*tc.epochs; best=float("inf"); wait=0
        Path("checkpoints").mkdir(exist_ok=True)
        if drive_dir:
            Path(drive_dir).mkdir(parents=True, exist_ok=True)

        start_ep = resume_from + 1
        if resume_from > 0:
            self.hist = {"tl":[],"vl":[],"vm":[],"vs":[],"vd":[]}
            print(f"\n  Resuming from epoch {start_ep}...")

        ngpu = torch.cuda.device_count()
        print(f"\n  Device: {self.dev} | GPUs: {ngpu} | Workers: {self.num_workers}")
        print(f"  Train: {len(tds):,} | Val: {len(vds):,}")
        print(f"  Batch: {tc.batch_size} | Epochs: {tc.epochs} (remaining: {tc.epochs-resume_from})\n")

        for ep in range(start_ep, tc.epochs+1):
            t0=time.time()
            tm=self._epoch(tdl,True,total)
            vm=self._epoch(vdl,False)
            dt=time.time()-t0

            vm_mae_mw = vm['mae'] * self.y_std
            self.hist["tl"].append(tm["loss"])
            self.hist["vl"].append(vm["loss"])
            self.hist["vm"].append(vm_mae_mw)
            self.hist["vs"].append(vm["sa"])
            self.hist["vd"].append(vm["da"])

            if ep%5==0 or ep==1:
                print(f"  Ep {ep:3d}/{tc.epochs} | TL:{tm['loss']:.4f} VL:{vm['loss']:.4f} "
                      f"| MAE:{vm_mae_mw:.2f}MW SigAcc:{vm['sa']:.3f} "
                      f"DirAcc:{vm['da']:.3f} DirLoss:{vm['dl']:.4f} "
                      f"LR:{self.opt.param_groups[0]['lr']:.2e} {dt:.1f}s")

            sd = self.model_raw.state_dict() if self.is_parallel else self.model.state_dict()
            ckpt_base = {"ep": ep, "model": sd, "opt": self.opt.state_dict(),
                         "y_mean": self.y_mean, "y_std": self.y_std}
            if vm["loss"]<best:
                best=vm["loss"]; wait=0
                ckpt = {**ckpt_base, "vl": best}
                torch.save(ckpt, "checkpoints/best.pt")
                if drive_dir:
                    torch.save(ckpt, f"{drive_dir}/best.pt")

            if drive_dir and ep % save_every == 0:
                torch.save({**ckpt_base, "vl": vm["loss"]},
                           f"{drive_dir}/checkpoint_ep{ep}.pt")

            elif tc.patience > 0:
                wait+=1
                if wait>=tc.patience:
                    print(f"\n  Early stop at {ep}"); break

        print(f"\n  Best val loss: {best:.4f}")
        return self.hist
