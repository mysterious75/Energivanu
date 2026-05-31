"""
ENERGIVANU — Trainer v3
Features: AMP, gradient accumulation, uncertainty weight monitoring, heartbeat
"""

import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np, time, json, threading
from pathlib import Path
from src.config import Config
from src.models.losses import SpikeLoss


class Trainer:
    def __init__(self, model, cfg: Config, y_mean=0.0, y_std=1.0,
                 use_dp=True, num_workers=4, use_amp=True, grad_accum=1, use_uncertainty=True):
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
        self.use_amp = use_amp and torch.cuda.is_available()
        self.grad_accum = grad_accum

        # Loss with uncertainty weighting
        self.loss_fn = SpikeLoss(
            uw=cfg.train.under_w, ow=cfg.train.over_w,
            ss=cfg.train.spike_std, use_uncertainty=use_uncertainty, dir_smoothing=0.1,
            dir_weight=cfg.train.dir_w
        )

        # Move loss parameters to device (for uncertainty weighting)
        self.loss_fn = self.loss_fn.to(self.dev)

        self.opt = torch.optim.AdamW(
            list(model.parameters()) + list(self.loss_fn.parameters()),
            lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
            betas=(0.9, 0.98)
        )

        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        self.step = 0
        self.hist = {"tl": [], "vl": [], "vm": [], "vs": [], "vd": [], "vdc": []}

        self._start_heartbeat()

    def _start_heartbeat(self):
        def _heartbeat():
            while True:
                time.sleep(120)
                print(f"  [heartbeat] {time.strftime('%H:%M:%S')} — training", flush=True)
        threading.Thread(target=_heartbeat, daemon=True).start()

    def _lr(self, cur, total):
        c = self.cfg.train
        if cur < c.warmup:
            return c.lr * cur / max(c.warmup, 1)
        p = (cur - c.warmup) / max(total - c.warmup, 1)
        return c.lr * 0.5 * (1 + np.cos(np.pi * p))

    def _epoch(self, dl, train, total=0):
        self.model.train() if train else self.model.eval()
        s = {k: 0. for k in ["pl", "sl", "dl", "loss", "mae", "da", "sa", "dir_conf",
                               "w_power", "w_signal", "w_direction"]}
        n = 0
        ctx = torch.enable_grad() if train else torch.no_grad()

        with ctx:
            for batch_idx, (x, yp, ys, yd) in enumerate(dl):
                x = x.to(self.dev, non_blocking=True)
                yp = yp.to(self.dev, non_blocking=True)
                ys = ys.to(self.dev, non_blocking=True)
                yd = yd.to(self.dev, non_blocking=True)

                if train:
                    self.step += 1
                    for g in self.opt.param_groups:
                        g["lr"] = self._lr(self.step, total)

                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            pp, ps, pd = self.model(x)
                            l, m = self.loss_fn(pp, yp, ps, ys, pd, yd)
                        l = l / self.grad_accum
                        self.scaler.scale(l).backward()

                        if (batch_idx + 1) % self.grad_accum == 0:
                            self.scaler.unscale_(self.opt)
                            nn.utils.clip_grad_norm_(
                                list(self.model.parameters()) + list(self.loss_fn.parameters()),
                                self.cfg.train.grad_clip
                            )
                            self.scaler.step(self.opt)
                            self.scaler.update()
                            self.opt.zero_grad(set_to_none=True)
                    else:
                        pp, ps, pd = self.model(x)
                        l, m = self.loss_fn(pp, yp, ps, ys, pd, yd)
                        l = l / self.grad_accum
                        l.backward()

                        if (batch_idx + 1) % self.grad_accum == 0:
                            nn.utils.clip_grad_norm_(
                                list(self.model.parameters()) + list(self.loss_fn.parameters()),
                                self.cfg.train.grad_clip
                            )
                            self.opt.step()
                            self.opt.zero_grad(set_to_none=True)
                else:
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            pp, ps, pd = self.model(x)
                            l, m = self.loss_fn(pp, yp, ps, ys, pd, yd)
                    else:
                        pp, ps, pd = self.model(x)
                        l, m = self.loss_fn(pp, yp, ps, ys, pd, yd)

                for k in s:
                    if k in m:
                        s[k] += m[k]
                n += 1

        return {k: v / max(n, 1) for k, v in s.items()}

    def fit(self, Xtr, Ytr, Str, Dtr, Xvl, Yvl, Svl, Dvl,
            drive_dir=None, save_every=5, resume_from=0):
        tc = self.cfg.train
        tds = TensorDataset(
            torch.FloatTensor(Xtr), torch.FloatTensor(Ytr),
            torch.LongTensor(Str), torch.LongTensor(Dtr)
        )
        vds = TensorDataset(
            torch.FloatTensor(Xvl), torch.FloatTensor(Yvl),
            torch.LongTensor(Svl), torch.LongTensor(Dvl)
        )

        nw = self.num_workers
        tdl = DataLoader(tds, tc.batch_size, shuffle=True,
                         num_workers=nw, pin_memory=True,
                         prefetch_factor=4 if nw > 0 else None,
                         persistent_workers=nw > 0)
        vdl = DataLoader(vds, tc.batch_size, shuffle=False,
                         num_workers=nw, pin_memory=True,
                         prefetch_factor=4 if nw > 0 else None,
                         persistent_workers=nw > 0)

        total = len(tdl) * tc.epochs
        best = float("inf")
        wait = 0
        Path("checkpoints").mkdir(exist_ok=True)
        if drive_dir:
            Path(drive_dir).mkdir(parents=True, exist_ok=True)

        start_ep = resume_from + 1
        if resume_from > 0:
            self.hist = {"tl": [], "vl": [], "vm": [], "vs": [], "vd": [], "vdc": []}

        ngpu = torch.cuda.device_count()
        amp_str = "FP16" if self.use_amp else "FP32"
        print(f"\n  Device: {self.dev} | GPUs: {ngpu} | Workers: {nw}")
        print(f"  AMP: {amp_str} | Grad Accum: {self.grad_accum}")
        print(f"  Train: {len(tds):,} | Val: {len(vds):,}")
        print(f"  Batch: {tc.batch_size} | Epochs: {tc.epochs} (remaining: {tc.epochs - resume_from})\n")

        for ep in range(start_ep, tc.epochs + 1):
            t0 = time.time()
            tm = self._epoch(tdl, True, total)
            vm = self._epoch(vdl, False)
            dt = time.time() - t0

            vm_mae_mw = vm['mae'] * self.y_std
            self.hist["tl"].append(tm["loss"])
            self.hist["vl"].append(vm["loss"])
            self.hist["vm"].append(vm_mae_mw)
            self.hist["vs"].append(vm["sa"])
            self.hist["vd"].append(vm["da"])
            self.hist["vdc"].append(vm.get("dir_conf", 0))

            # Print with uncertainty weights
            wp = vm.get("w_power", 1.0)
            ws = vm.get("w_signal", 1.0)
            wd = vm.get("w_direction", 1.0)
            print(f"  Ep {ep:3d}/{tc.epochs} | TL:{tm['loss']:.2f} VL:{vm['loss']:.2f} "
                  f"| MAE:{vm_mae_mw:.2f}MW Sig:{vm['sa']:.3f} "
                  f"Dir:{vm['da']:.3f} DirL:{vm['dl']:.3f} "
                  f"W:[{wp:.2f},{ws:.2f},{wd:.2f}] "
                  f"LR:{self.opt.param_groups[0]['lr']:.2e} {dt:.1f}s")

            # Save checkpoint
            sd = self.model_raw.state_dict() if self.is_parallel else self.model.state_dict()
            ckpt_base = {
                "ep": ep, "model": sd, "opt": self.opt.state_dict(),
                "loss_fn": self.loss_fn.state_dict(),
                "y_mean": self.y_mean, "y_std": self.y_std,
                "history": self.hist
            }

            # Always save latest
            torch.save({**ckpt_base, "vl": vm["loss"]}, f"checkpoints/latest.pt")
            if drive_dir:
                torch.save({**ckpt_base, "vl": vm["loss"]}, f"{drive_dir}/latest.pt")

            if vm["loss"] < best:
                best = vm["loss"]
                wait = 0
                torch.save({**ckpt_base, "vl": best}, "checkpoints/best.pt")
                if drive_dir:
                    torch.save({**ckpt_base, "vl": best}, f"{drive_dir}/best.pt")
            else:
                wait += 1
                if tc.patience > 0 and wait >= tc.patience:
                    print(f"\n  Early stop at {ep}")
                    break

            if drive_dir and ep % save_every == 0:
                torch.save({**ckpt_base, "vl": vm["loss"]},
                           f"{drive_dir}/checkpoint_ep{ep}.pt")

            # Save history every 10 epochs
            if ep % 10 == 0 and drive_dir:
                with open(f"{drive_dir}/history.json", "w") as f:
                    json.dump({k: [float(v) for v in vals] for k, vals in self.hist.items()}, f)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"\n  Best val loss: {best:.4f}")
        return self.hist
