# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Commercial-Safe Training Pipeline
===================================
Trains the EnergivanuPEB model using ONLY commercially-licensed data sources:
  - Alibaba Cluster Trace GPU v2020 (CC BY 4.0)
  - Self-collected Kaggle T4 data (own data)
  - Synthetic data (generated)

All data sources used here are verified commercial-safe.
See config/data_sources.yaml for the full data source registry.

Usage::

    # Train with default config
    python -m energivanu.train_commercial

    # Train with custom config
    python -m energivanu.train_commercial --config config/custom.yaml

    # Train with specific data sources
    python -m energivanu.train_commercial --sources alibaba_gpu_trace kaggle_t4

    # Resume from checkpoint
    python -m energivanu.train_commercial --resume models/checkpoints/commercial_best.pt
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler

from .config import EnergivanuConfig, get_config
from .distributed import (
    cleanup,
    get_device,
    get_local_rank,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    save_checkpoint,
    setup as ddp_setup,
)
from .logging_config import get_logger, setup_logging, timed
from .model import EnergivanuPEB

logger = get_logger("train_commercial")


# ---------------------------------------------------------------------------
# Commercial-safe dataset
# ---------------------------------------------------------------------------

@dataclass
class DataManifest:
    """Tracks which data sources were used in training."""
    sources: List[str]
    licenses: Dict[str, str]
    commercial_safe: bool
    total_samples: int
    train_samples: int
    val_samples: int


class CommercialDataset(Dataset):
    """
    Dataset that loads ONLY commercially-licensed data.

    Supports:
      - Alibaba GPU trace data (preprocessed CSV/npz)
      - Self-collected Kaggle T4 telemetry
      - Synthetic data

    Raises ValueError if any non-commercial source is requested.
    """

    # Sources confirmed safe for commercial use
    _COMMERCIAL_SOURCES = {"alibaba_gpu_trace", "kaggle_t4", "synthetic"}
    _LICENSE_MAP = {
        "alibaba_gpu_trace": "CC BY 4.0",
        "kaggle_t4": "Proprietary (own)",
        "synthetic": "N/A (generated)",
    }

    def __init__(
        self,
        X: np.ndarray,
        Y_power: np.ndarray,
        Y_signal: np.ndarray,
        sources: List[str],
    ) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y_power = torch.tensor(Y_power, dtype=torch.float32)
        self.Y_signal = torch.tensor(Y_signal, dtype=torch.long)
        self.sources = sources

        # Validate all sources are commercial-safe
        for src in sources:
            if src not in self._COMMERCIAL_SOURCES:
                raise ValueError(
                    f"Source '{src}' is NOT commercially licensed. "
                    f"Allowed: {self._COMMERCIAL_SOURCES}"
                )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y_power[idx], self.Y_signal[idx]

    def manifest(self) -> DataManifest:
        """Return a manifest of data sources used."""
        return DataManifest(
            sources=self.sources,
            licenses={s: self._LICENSE_MAP.get(s, "Unknown") for s in self.sources},
            commercial_safe=True,
            total_samples=len(self.X),
            train_samples=0,  # filled by caller
            val_samples=0,
        )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_alibaba_data(data_dir: str, seq_len: int, pred_horizon: int,
                       stride: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load preprocessed Alibaba GPU trace data.

    Expects preprocessed .npz files with keys: X, Y_power, Y_signal.
    Run the Alibaba processor first to generate these files.
    """
    npz_path = Path(data_dir) / "alibaba_processed.npz"
    if not npz_path.exists():
        logger.warning(
            "Alibaba data not found — skipping",
            extra={"path": str(npz_path), "source": "alibaba_gpu_trace"},
        )
        return (
            np.array([]).reshape(0, seq_len, 15),
            np.array([]).reshape(0, pred_horizon),
            np.array([], dtype=np.int64),
        )

    data = np.load(str(npz_path))
    X = data["X"].astype(np.float32)
    Y_power = data["Y_power"].astype(np.float32)
    Y_signal = data["Y_signal"].astype(np.int64)

    logger.info(
        "loaded Alibaba data",
        extra={"samples": len(X), "source": "alibaba_gpu_trace"},
    )
    return X, Y_power, Y_signal


def _load_kaggle_data(data_dir: str, seq_len: int, pred_horizon: int,
                      stride: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load self-collected Kaggle T4 telemetry data.

    Expects preprocessed .npz files with keys: X, Y_power, Y_signal.
    """
    npz_path = Path(data_dir) / "kaggle_processed.npz"
    if not npz_path.exists():
        logger.warning(
            "Kaggle data not found — skipping",
            extra={"path": str(npz_path), "source": "kaggle_t4"},
        )
        return (
            np.array([]).reshape(0, seq_len, 15),
            np.array([]).reshape(0, pred_horizon),
            np.array([], dtype=np.int64),
        )

    data = np.load(str(npz_path))
    X = data["X"].astype(np.float32)
    Y_power = data["Y_power"].astype(np.float32)
    Y_signal = data["Y_signal"].astype(np.int64)

    logger.info(
        "loaded Kaggle T4 data",
        extra={"samples": len(X), "source": "kaggle_t4"},
    )
    return X, Y_power, Y_signal


def _generate_synthetic_data(
    num_samples: int, seq_len: int, pred_horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic training data as a baseline/fallback."""
    t = np.linspace(0, 50 * np.pi, num_samples + seq_len + pred_horizon)
    base_power = np.sin(t) * 50 + 200
    noise = np.random.normal(0, 2, len(t))
    power = base_power + noise

    X: List[np.ndarray] = []
    Y_power: List[np.ndarray] = []
    Y_signal: List[int] = []

    for i in range(num_samples):
        features = np.zeros((seq_len, 15), dtype=np.float32)
        features[:, 0] = power[i:i + seq_len]
        features[:, 1] = np.gradient(features[:, 0])
        features[:, 2] = np.gradient(features[:, 1])
        features[:, 3] = np.convolve(features[:, 0], np.ones(5) / 5, mode="same")
        features[:, 4] = 2.0
        features[:, 5] = features[:, 0] / 700.0
        features[:, 6] = features[:, 0] / 700.0 * 1.1
        features[:, 7] = 0.5
        features[:, 8] = 0.6
        features[:, 9] = 0.8
        features[:, 10] = 0.8
        features[:, 11] = 0.4
        features[:, 12] = np.sin(t[i:i + seq_len])
        features[:, 13] = np.cos(t[i:i + seq_len])
        features[:, 14] = 0.0

        target_p = power[i + seq_len:i + seq_len + pred_horizon]
        change = target_p[0] - features[-1, 0]
        signal = 1 if change > 0.5 else (2 if change < -0.5 else 0)

        X.append(features)
        Y_power.append(target_p)
        Y_signal.append(signal)

    logger.info(
        "generated synthetic data",
        extra={"samples": num_samples, "source": "synthetic"},
    )
    return (
        np.array(X, dtype=np.float32),
        np.array(Y_power, dtype=np.float32),
        np.array(Y_signal, dtype=np.int64),
    )


def _get_sampler(dataset: Dataset, shuffle: bool):
    """Return a DistributedSampler if distributed, else None."""
    if is_distributed():
        return DistributedSampler(dataset, shuffle=shuffle)
    return None


def build_commercial_dataloaders(
    cfg: EnergivanuConfig,
    sources: Optional[List[str]] = None,
    data_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataManifest]:
    """
    Build train/val dataloaders from commercial-safe data sources only.

    Args:
        cfg: Energivanu configuration.
        sources: List of source names to use. Defaults to all commercial-safe sources.
        data_dir: Base directory for data files.

    Returns:
        train_loader, val_loader, data_manifest

    Raises:
        ValueError: If a non-commercial source is requested.
    """
    if sources is None:
        sources = ["alibaba_gpu_trace", "kaggle_t4", "synthetic"]

    if data_dir is None:
        data_dir = str(
            Path(__file__).resolve().parent.parent.parent / "data"
        )

    seq_len = cfg.model.seq_len
    pred_horizon = cfg.model.pred_horizon
    stride = cfg.training.stride

    all_X: List[np.ndarray] = []
    all_Yp: List[np.ndarray] = []
    all_Ys: List[np.ndarray] = []
    active_sources: List[str] = []

    loader_map = {
        "alibaba_gpu_trace": lambda: _load_alibaba_data(data_dir, seq_len, pred_horizon, stride),
        "kaggle_t4": lambda: _load_kaggle_data(data_dir, seq_len, pred_horizon, stride),
        "synthetic": lambda: _generate_synthetic_data(2000, seq_len, pred_horizon),
    }

    for src in sources:
        if src not in loader_map:
            raise ValueError(f"Unknown source: {src}")
        if src not in CommercialDataset._COMMERCIAL_SOURCES:
            raise ValueError(
                f"Source '{src}' is NOT commercially licensed. "
                f"Only {CommercialDataset._COMMERCIAL_SOURCES} are allowed."
            )

        X, Yp, Ys = loader_map[src]()
        if len(X) > 0:
            all_X.append(X)
            all_Yp.append(Yp)
            all_Ys.append(Ys)
            active_sources.append(src)

    if not all_X:
        raise RuntimeError(
            "No data loaded from any source. Ensure at least one data source "
            "exists or include 'synthetic' in the source list."
        )

    X = np.concatenate(all_X, axis=0)
    Y_power = np.concatenate(all_Yp, axis=0)
    Y_signal = np.concatenate(all_Ys, axis=0)

    logger.info(
        "combined commercial dataset",
        extra={"total_samples": len(X), "sources": active_sources},
    )

    dataset = CommercialDataset(X, Y_power, Y_signal, active_sources)

    # Train/val split
    val_size = int(len(dataset) * cfg.training.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_sampler = _get_sampler(train_ds, shuffle=True)
    val_sampler = _get_sampler(val_ds, shuffle=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=0,
        pin_memory=True,
    )

    manifest = dataset.manifest()
    manifest.train_samples = train_size
    manifest.val_samples = val_size

    logger.info(
        "dataloaders ready",
        extra={
            "train": train_size,
            "val": val_size,
            "sources": active_sources,
        },
    )
    return train_loader, val_loader, manifest


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@timed("train_commercial.epoch")
def _train_epoch(
    model: EnergivanuPEB,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    power_loss_fn: nn.Module,
    signal_loss_fn: nn.Module,
    device: torch.device,
    grad_clip_norm: float,
) -> Tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, avg_mape)."""
    model.train()
    total_loss = 0.0
    total_mape = 0.0
    n_batches = 0

    for X, Y_power, Y_signal in loader:
        X = X.to(device)
        Y_power = Y_power.to(device)
        Y_signal = Y_signal.to(device)

        power_pred, signal_logits = model(X)
        loss_power = power_loss_fn(power_pred, Y_power)
        loss_signal = signal_loss_fn(signal_logits, Y_signal)
        loss = loss_power + 0.5 * loss_signal

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        # MAPE computation
        with torch.no_grad():
            mape = (
                torch.abs((power_pred - Y_power) / (Y_power + 1e-8)).mean().item() * 100
            )

        total_loss += loss.item()
        total_mape += mape
        n_batches += 1

    return total_loss / max(n_batches, 1), total_mape / max(n_batches, 1)


@timed("train_commercial.validate")
def _validate(
    model: EnergivanuPEB,
    loader: DataLoader,
    power_loss_fn: nn.Module,
    signal_loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Run validation. Returns (avg_loss, mape)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    with torch.no_grad():
        for X, Y_power, Y_signal in loader:
            X = X.to(device)
            Y_power = Y_power.to(device)
            Y_signal = Y_signal.to(device)

            power_pred, signal_logits = model(X)
            loss_power = power_loss_fn(power_pred, Y_power)
            loss_signal = signal_loss_fn(signal_logits, Y_signal)
            loss = loss_power + 0.5 * loss_signal

            total_loss += loss.item()
            n_batches += 1
            all_preds.append(power_pred.cpu().numpy())
            all_targets.append(Y_power.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    mape = np.mean(np.abs((preds - targets) / (targets + 1e-8))) * 100

    return avg_loss, mape


def train_commercial(
    config_path: Optional[str] = None,
    sources: Optional[List[str]] = None,
    resume_path: Optional[str] = None,
    distributed: bool = False,
) -> Dict[str, float]:
    """
    Full commercial-safe training pipeline.

    Only uses data from commercially-licensed sources (Alibaba CC BY 4.0,
    own Kaggle data, synthetic). Implements:
      - Train/val split with deterministic seeding
      - Early stopping on validation loss
      - Checkpoint saving (best model)
      - MAPE metric tracking
      - Cosine annealing + ReduceLROnPlateau LR scheduling
      - Gradient clipping

    Args:
        config_path: Path to YAML config. Uses default if None.
        sources: Data sources to use. Defaults to all commercial-safe sources.
        resume_path: Path to checkpoint to resume training from.

    Returns:
        Dictionary with final metrics: val_loss, val_mape, best_epoch, etc.
    """
    setup_logging()
    cfg = get_config(config_path)

    # Initialize DDP if requested
    ddp_enabled = distributed and ddp_setup()
    world_size = get_world_size()
    rank = get_rank()
    local_rank = get_local_rank()
    device = get_device()

    logger.info(
        "training started",
        extra={
            "device": str(device),
            "mode": "commercial",
            "distributed": ddp_enabled,
            "world_size": world_size,
            "rank": rank,
        },
    )

    # Scale batch size for distributed training
    if ddp_enabled and world_size > 1:
        cfg.training.batch_size = max(1, cfg.training.batch_size // world_size)
        logger.info(
            "batch size scaled for distributed training",
            extra={"per_gpu_batch": cfg.training.batch_size, "world_size": world_size},
        )

    # Build dataloaders
    train_loader, val_loader, manifest = build_commercial_dataloaders(cfg, sources)

    # Initialize model
    raw_model = EnergivanuPEB(
        n_features=cfg.model.n_features,
        seq_len=cfg.model.seq_len,
        pred_horizon=cfg.model.pred_horizon,
        tcn_channels=list(cfg.model.tcn_channels),
        tcn_kernels=list(cfg.model.tcn_kernels),
        attention_heads=cfg.model.attention_heads,
        attention_dim=cfg.model.attention_dim,
        hidden_dims=list(cfg.model.hidden_dims),
        n_signal_classes=cfg.model.n_signal_classes,
        dropout=cfg.model.dropout,
    ).to(device)

    param_count = raw_model.count_parameters()
    logger.info("model initialized", extra={"parameters": param_count})

    # Wrap with DDP if distributed
    model = raw_model
    if ddp_enabled:
        model = DDP(raw_model, device_ids=[local_rank] if torch.cuda.is_available() else None)
        logger.info("model wrapped with DDP", extra={"local_rank": local_rank})

    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        inner_model = model.module if ddp_enabled else model
        inner_model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info("resumed from checkpoint", extra={"path": resume_path, "epoch": start_epoch})

    # Optimizer + schedulers
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.epochs,
        eta_min=1e-6,
    )
    plateau_scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        verbose=False,
    )

    # Loss functions
    power_loss_fn = nn.MSELoss()
    signal_loss_fn = nn.CrossEntropyLoss()

    # Checkpoint directory
    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "commercial_best.pt"

    # Training state
    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_patience = 25
    history: List[Dict[str, float]] = []

    logger.info(
        "training config",
        extra={
            "epochs": cfg.training.epochs,
            "batch_size": cfg.training.batch_size,
            "lr": cfg.training.learning_rate,
            "grad_clip": cfg.training.grad_clip_norm,
            "early_stop_patience": early_stop_patience,
            "data_sources": manifest.sources,
            "train_samples": manifest.train_samples,
            "val_samples": manifest.val_samples,
        },
    )

    for epoch in range(start_epoch, cfg.training.epochs):
        t0 = time.time()

        # Set epoch for distributed sampler
        if ddp_enabled and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        # Train
        train_loss, train_mape = _train_epoch(
            model, train_loader, optimizer, power_loss_fn, signal_loss_fn,
            device, cfg.training.grad_clip_norm,
        )

        # Validate
        val_loss, val_mape = _validate(
            model, val_loader, power_loss_fn, signal_loss_fn, device,
        )

        # LR scheduling
        cosine_scheduler.step()
        plateau_scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        # Log
        logger.info(
            "epoch complete",
            extra={
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 6),
                "train_mape": round(train_mape, 3),
                "val_loss": round(val_loss, 6),
                "val_mape": round(val_mape, 3),
                "lr": current_lr,
                "elapsed_s": round(dt, 1),
            },
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_mape": train_mape,
            "val_loss": val_loss,
            "val_mape": val_mape,
            "lr": current_lr,
        })

        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            inner_model = model.module if ddp_enabled else model
            save_checkpoint({
                "model_state_dict": inner_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_mape": val_mape,
                "config": {
                    "n_features": cfg.model.n_features,
                    "seq_len": cfg.model.seq_len,
                    "pred_horizon": cfg.model.pred_horizon,
                    "tcn_channels": list(cfg.model.tcn_channels),
                    "tcn_kernels": list(cfg.model.tcn_kernels),
                    "attention_heads": cfg.model.attention_heads,
                    "attention_dim": cfg.model.attention_dim,
                    "hidden_dims": list(cfg.model.hidden_dims),
                    "n_signal_classes": cfg.model.n_signal_classes,
                    "dropout": cfg.model.dropout,
                },
                "data_manifest": {
                    "sources": manifest.sources,
                    "licenses": manifest.licenses,
                    "commercial_safe": manifest.commercial_safe,
                    "train_samples": manifest.train_samples,
                    "val_samples": manifest.val_samples,
                },
            }, str(best_path))
            logger.info(
                "checkpoint saved",
                extra={"path": str(best_path), "val_loss": round(val_loss, 6)},
            )
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info(
                    "early stopping triggered",
                    extra={"epoch": epoch + 1, "patience": early_stop_patience},
                )
                break

    # Final summary
    best_epoch = history[np.argmin([h["val_loss"] for h in history])]["epoch"] if history else 0
    results = {
        "best_val_loss": best_val_loss,
        "best_val_mape": min(h["val_mape"] for h in history) if history else 0.0,
        "best_epoch": best_epoch,
        "total_epochs": len(history),
        "parameters": param_count,
        "data_sources": manifest.sources,
        "commercial_safe": True,
    }

    logger.info("training complete", extra=results)

    if ddp_enabled:
        cleanup()

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for commercial training.

    For single-GPU training:
        python -m energivanu.train_commercial

    For multi-GPU training (requires ``torchrun``):
        torchrun --nproc_per_node=N python -m energivanu.train_commercial --distributed
    """
    parser = argparse.ArgumentParser(
        description="Train Energivanu PEB model on commercial-safe data only.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--sources", nargs="+", default=None,
        help="Data sources to use (default: alibaba_gpu_trace kaggle_t4 synthetic)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--distributed", action="store_true",
        help="Enable distributed training (requires torchrun or env vars)",
    )
    args = parser.parse_args()

    results = train_commercial(
        config_path=args.config,
        sources=args.sources,
        resume_path=args.resume,
        distributed=args.distributed,
    )

    if is_main_process():
        print("\n" + "=" * 60)
        print("COMMERCIAL TRAINING COMPLETE")
        print("=" * 60)
        print(f"  Best val loss:    {results['best_val_loss']:.6f}")
        print(f"  Best val MAPE:    {results['best_val_mape']:.2f}%")
        print(f"  Best epoch:       {results['best_epoch']}")
        print(f"  Total epochs:     {results['total_epochs']}")
        print(f"  Parameters:       {results['parameters']:,}")
        print(f"  Data sources:     {results['data_sources']}")
        print(f"  Commercial safe:  {results['commercial_safe']}")
        print("=" * 60)


if __name__ == "__main__":
    main()
