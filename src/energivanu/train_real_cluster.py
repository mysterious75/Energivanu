# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Real Cluster Fine-Tuning
==========================
Fine-tune the EnergivanuPEB model on real GPU cluster telemetry data
collected from a production multi-node cluster.

This script is the **final step** in the cluster training pipeline:

    1. Collect telemetry on each cluster node (``scripts/collect_data.py``)
    2. Merge per-node CSVs → .npz (``scripts/cluster/merge_telemetry.py``)
    3. Fine-tune the model here

The script:
  - Loads a pre-existing checkpoint (e.g., ``commercial_best.pt``)
  - Loads real cluster data from a merged .npz file
  - Runs DDP across all available GPUs
  - Logs loss/MAPE per epoch
  - Saves improved checkpoint when validation loss decreases
  - Supports early stopping

Usage::

    # Single GPU
    python -m energivanu.train_real_cluster \\
        --data data/cluster/merged.npz \\
        --checkpoint models/checkpoints/commercial_best.pt

    # Multi-GPU via torchrun
    torchrun --nproc_per_node=N -m energivanu.train_real_cluster \\
        --data data/cluster/merged.npz \\
        --checkpoint models/checkpoints/commercial_best.pt

    # With custom config
    torchrun --nproc_per_node=N -m energivanu.train_real_cluster \\
        --data data/cluster/merged.npz \\
        --checkpoint models/checkpoints/commercial_best.pt \\
        --config config/cluster_ft.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, random_split

from .config import get_config
from .distributed import (
    cleanup,
    get_device,
    get_local_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    save_checkpoint,
)
from .distributed import (
    setup as ddp_setup,
)
from .logging_config import get_logger, setup_logging
from .model import EnergivanuPEB

logger = get_logger("train_real_cluster")


# ---------------------------------------------------------------------------
# Mini dataset — wraps .npz arrays
# ---------------------------------------------------------------------------

class ClusterDataset(Dataset):
    """Dataset from merged cluster telemetry .npz file."""

    def __init__(self, X: np.ndarray, Y_power: np.ndarray, Y_signal: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y_power = torch.tensor(Y_power, dtype=torch.float32)
        self.Y_signal = torch.tensor(Y_signal, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y_power[idx], self.Y_signal[idx]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cluster_data(npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Load merged cluster .npz file.

    Returns:
        X, Y_power, Y_signal, metadata_dict
    """
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"Cluster data not found: {npz_path}")

    data = np.load(str(npz_path))
    X = data["X"].astype(np.float32)
    Y_power = data["Y_power"].astype(np.float32)
    Y_signal = data["Y_signal"].astype(np.int64)
    metadata = {}
    if "metadata" in data:
        try:
            metadata = json.loads(str(data["metadata"]))
        except (json.JSONDecodeError, TypeError):
            metadata = {"raw": str(data["metadata"])}

    logger.info(
        "cluster data loaded",
        extra={
            "path": npz_path,
            "samples": len(X),
            "gpus": metadata.get("num_gpus_facility", "?"),
            "nodes": metadata.get("num_nodes", "?"),
        },
    )
    return X, Y_power, Y_signal, metadata


def build_cluster_dataloaders(
    npz_path: str,
    batch_size: int,
    val_split: float = 0.15,
    seq_len: int = 30,
    pred_horizon: int = 10,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, int, Dict]:
    """Build train/val dataloaders from merged cluster .npz.

    Returns:
        train_loader, val_loader, n_features, metadata
    """
    X, Y_power, Y_signal, metadata = load_cluster_data(npz_path)
    dataset = ClusterDataset(X, Y_power, Y_signal)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    from torch.utils.data.distributed import DistributedSampler

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed() else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_distributed() else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        "dataloaders ready",
        extra={"train": train_size, "val": val_size},
    )
    return train_loader, val_loader, X.shape[2], metadata


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_real_cluster(
    npz_path: str,
    checkpoint_path: Optional[str] = None,
    config_path: Optional[str] = None,
    epochs: int = 100,
    batch_size: Optional[int] = None,
    learning_rate: float = 1e-4,
    distributed: bool = False,
    output_path: str = "models/checkpoints/cluster_finetuned.pt",
    early_stop_patience: int = 25,
) -> Dict[str, float]:
    """Fine-tune the model on real cluster telemetry data.

    Args:
        npz_path: Path to merged cluster .npz file.
        checkpoint_path: Path to existing checkpoint to fine-tune from.
            If ``None``, train from scratch.
        config_path: Path to YAML config.
        epochs: Number of training epochs.
        batch_size: Batch size per GPU. Defaults to config value.
        learning_rate: Learning rate for fine-tuning (lower than
            pre-training default).
        distributed: Enable DDP.
        output_path: Where to save the best checkpoint.
        early_stop_patience: Early stopping patience.

    Returns:
        Dictionary with final metrics.
    """
    setup_logging()
    cfg = get_config(config_path)

    ddp_enabled = distributed and ddp_setup()
    world_size = get_world_size()
    local_rank = get_local_rank()
    device = get_device()

    logger.info(
        "cluster fine-tuning started",
        extra={
            "device": str(device),
            "distributed": ddp_enabled,
            "world_size": world_size,
            "data": npz_path,
            "checkpoint": checkpoint_path or "none (from scratch)",
        },
    )

    bsize = batch_size or cfg.training.batch_size
    if ddp_enabled and world_size > 1:
        bsize = max(1, bsize // world_size)

    train_loader, val_loader, n_features, metadata = build_cluster_dataloaders(
        npz_path=npz_path,
        batch_size=bsize,
        val_split=cfg.training.val_split,
        seq_len=cfg.model.seq_len,
        pred_horizon=cfg.model.pred_horizon,
    )

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

    model = raw_model
    if ddp_enabled:
        model = DDP(raw_model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    start_epoch = 0
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        inner_model = model.module if ddp_enabled else model
        inner_model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(
            "loaded checkpoint",
            extra={"path": checkpoint_path, "epoch": start_epoch},
        )

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    plateau_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    power_loss_fn = nn.MSELoss()
    signal_loss_fn = nn.CrossEntropyLoss()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    history: List[Dict[str, float]] = []

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        if ddp_enabled and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0
        train_mape = 0.0
        n_batches = 0

        for X_batch, Y_power, Y_signal in train_loader:
            X_batch = X_batch.to(device)
            Y_power = Y_power.to(device)
            Y_signal = Y_signal.to(device)

            power_pred, signal_logits = model(X_batch)
            loss_power = power_loss_fn(power_pred, Y_power)
            loss_signal = signal_loss_fn(signal_logits, Y_signal)
            loss = loss_power + 0.5 * loss_signal

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)
            optimizer.step()

            with torch.no_grad():
                mape = torch.abs((power_pred - Y_power) / (Y_power + 1e-8)).mean().item() * 100

            train_loss += loss.item()
            train_mape += mape
            n_batches += 1

        avg_train_loss = train_loss / max(n_batches, 1)
        avg_train_mape = train_mape / max(n_batches, 1)

        model.eval()
        val_loss = 0.0
        n_val = 0
        all_preds: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []

        with torch.no_grad():
            for X_batch, Y_power, Y_signal in val_loader:
                X_batch = X_batch.to(device)
                Y_power = Y_power.to(device)
                Y_signal = Y_signal.to(device)

                power_pred, signal_logits = model(X_batch)
                loss_power = power_loss_fn(power_pred, Y_power)
                loss_signal = signal_loss_fn(signal_logits, Y_signal)
                loss = loss_power + 0.5 * loss_signal

                val_loss += loss.item()
                n_val += 1
                all_preds.append(power_pred.cpu().numpy())
                all_targets.append(Y_power.cpu().numpy())

        avg_val_loss = val_loss / max(n_val, 1)
        preds = np.concatenate(all_preds) if all_preds else np.array([])
        targets = np.concatenate(all_targets) if all_targets else np.array([])
        val_mape = (
            float(np.mean(np.abs((preds - targets) / (targets + 1e-8))) * 100)
            if len(preds) > 0 else 0.0
        )

        cosine_scheduler.step()
        plateau_scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        logger.info(
            "epoch complete",
            extra={
                "epoch": epoch + 1,
                "train_loss": round(avg_train_loss, 6),
                "train_mape": round(avg_train_mape, 3),
                "val_loss": round(avg_val_loss, 6),
                "val_mape": round(val_mape, 3),
                "lr": current_lr,
                "elapsed_s": round(dt, 1),
            },
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_mape": avg_train_mape,
            "val_loss": avg_val_loss,
            "val_mape": val_mape,
            "lr": current_lr,
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            inner_model = model.module if ddp_enabled else model
            save_checkpoint({
                "model_state_dict": inner_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": avg_val_loss,
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
                "cluster_metadata": metadata,
                "fine_tune_from": checkpoint_path,
            }, str(out_path))
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info("early stopping triggered", extra={"epoch": epoch + 1})
                break

    best_epoch = history[np.argmin([h["val_loss"] for h in history])]["epoch"] if history else 0
    results = {
        "best_val_loss": best_val_loss,
        "best_val_mape": min(h["val_mape"] for h in history) if history else 0.0,
        "best_epoch": best_epoch,
        "total_epochs": len(history),
        "parameters": param_count,
        "data_source": "real_cluster",
        "data_path": npz_path,
        "checkpoint": checkpoint_path,
    }

    logger.info("cluster fine-tuning complete", extra=results)

    if ddp_enabled:
        cleanup()

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune EnergivanuPEB on real GPU cluster telemetry data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single GPU
  python -m energivanu.train_real_cluster --data data/cluster/merged.npz

  # Multi-GPU
  torchrun --nproc_per_node=8 -m energivanu.train_real_cluster \\
      --data data/cluster/merged.npz \\
      --checkpoint models/checkpoints/commercial_best.pt

  # Custom config
  python -m energivanu.train_real_cluster --data data/cluster/merged.npz \\
      --config config/cluster_finetune.yaml
        """,
    )
    parser.add_argument("--data", required=True, help="Merged cluster .npz path")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to fine-tune from")
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output",
                        default="models/checkpoints/cluster_finetuned.pt",
                        help="Output checkpoint path")
    parser.add_argument("--distributed", action="store_true", help="Enable DDP (requires torchrun)")
    args = parser.parse_args()

    results = train_real_cluster(
        npz_path=args.data,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        distributed=args.distributed,
        output_path=args.output,
    )

    if is_main_process():
        print("\n" + "=" * 60)
        print("CLUSTER FINE-TUNING COMPLETE")
        print("=" * 60)
        print(f"  Best val loss:    {results['best_val_loss']:.6f}")
        print(f"  Best val MAPE:    {results['best_val_mape']:.2f}%")
        print(f"  Best epoch:       {results['best_epoch']}")
        print(f"  Total epochs:     {results['total_epochs']}")
        print(f"  Parameters:       {results['parameters']:,}")
        print(f"  Output:           {args.output}")
        print("=" * 60)


if __name__ == "__main__":
    main()
