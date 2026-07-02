# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Cluster Telemetry Merger
==========================
Merge per-GPU telemetry CSVs from multiple cluster nodes into a unified
training dataset suitable for the Energivanu PEB model.

Workflow::

    1. Each cluster node runs ``NvidiaSmiCollector`` or
       ``scripts/collect_data.py``, producing a per-GPU CSV with columns:

           timestamp, unix_ts, gpu_id, power_w, temp_c, util_pct,
           mem_util_pct, sm_clock_mhz, mem_clock_mhz

    2. Copy all node CSVs to a central machine.

    3. Run ``ClusterMerger`` on the set of CSVs::

           from energivanu.data.cluster_merger import ClusterMerger

           merger = ClusterMerger()
           merger.merge_and_convert(
               node_csvs=["node0.csv", "node1.csv", ...],
               output_path="data/cluster/merged.npz",
               seq_len=30,
               pred_horizon=10,
           )

    4. Train with the merged dataset::

           python -m energivanu.train_real_cluster \\
               --data data/cluster/merged.npz \\
               --checkpoint models/checkpoints/commercial_best.pt

The merger computes **actual facility-level power** by summing power_w across
every GPU in every node.  Rolling statistics, derivatives, and all 15 training
features are computed on this real facility trace, not a scaled-up estimate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..logging_config import get_logger, timed

logger = get_logger("cluster_merger")

_GPU_TDP_W: float = 700.0
_MAX_TEMP: float = 100.0
_ROLLING_WINDOW: int = 250
_NUM_FEATURES: int = 15

FEATURE_NAMES: List[str] = [
    "facility_mw",
    "power_roc",
    "power_roc2",
    "power_roll_mean",
    "power_roll_std",
    "gpu_avg_power_norm",
    "gpu_max_power_norm",
    "gpu_avg_temp_norm",
    "gpu_max_temp_norm",
    "gpu_avg_util_norm",
    "gpu_avg_mem_util_norm",
    "cpu_util_est_norm",
    "hour_sin",
    "hour_cos",
    "is_allreduce",
]


class ClusterMerger:
    """Merge per-node telemetry CSVs into a unified training dataset.

    Accepts a list of CSV paths (one per cluster node), each containing
    per-GPU telemetry rows collected by ``NvidiaSmiCollector``.  Produces
    a .npz file with keys ``X``, ``Y_power``, ``Y_signal`` ready for
    direct consumption by ``EnergivanuPEB`` training pipelines.

    Args:
        num_gpus_facility: Total GPUs in the cluster (inferred from data
            if ``None``).
        gpus_per_node: GPUs per node (used to parse CPU utilization; not
            critical if column is absent).
        cpu_util_estimate: Default CPU utilization (0--1) when CPU util
            is not present in the telemetry CSV.
        rolling_window: Window size for rolling mean/std.
    """

    def __init__(
        self,
        num_gpus_facility: Optional[int] = None,
        gpus_per_node: int = 8,
        cpu_util_estimate: float = 0.4,
        rolling_window: int = _ROLLING_WINDOW,
    ) -> None:
        self.num_gpus_facility = num_gpus_facility
        self.gpus_per_node = gpus_per_node
        self.cpu_util_estimate = cpu_util_estimate
        self.rolling_window = rolling_window
        self.node_csvs: List[str] = []
        self.merged_df: Optional[pd.DataFrame] = None
        self.features_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @timed("cluster_merger.merge_and_convert")
    def merge_and_convert(
        self,
        node_csvs: List[str],
        output_path: Optional[str] = None,
        seq_len: int = 30,
        pred_horizon: int = 10,
        stride: int = 50,
        val_split: float = 0.15,
        random_seed: int = 42,
    ) -> Dict[str, object]:
        """End-to-end: load, merge, convert, and optionally save.

        Args:
            node_csvs: List of paths to per-node telemetry CSV files.
            output_path: If set, save the merged dataset as .npz.
            seq_len: Number of timesteps per input sequence.
            pred_horizon: Number of timesteps to predict.
            stride: Stride between consecutive sequences.
            val_split: Fraction of sequences used for validation.
            random_seed: Random seed for train/val split.

        Returns:
            Dictionary with keys:
              - X: (N, seq_len, 15) feature sequences
              - Y_power: (N, pred_horizon) power targets in MW
              - Y_signal: (N,) signal labels
              - metadata: dict with cluster info

        Raises:
            FileNotFoundError: If any CSV does not exist.
            ValueError: If no valid data after merge.
        """
        self.node_csvs = node_csvs
        self._validate_paths()

        df = self._load_and_merge()
        features = self._extract_features(df)

        if self.num_gpus_facility is None:
            self.num_gpus_facility = features.attrs.get("total_gpus", 0)
            if self.num_gpus_facility == 0:
                raise ValueError(
                    "Could not infer num_gpus_facility.  Set explicitly "
                    "or ensure CSVs contain valid power data."
                )

        X, Y_power, Y_signal = self._create_sequences(
            features, seq_len, pred_horizon, stride,
        )

        if len(X) == 0:
            raise ValueError(
                f"No sequences created.  Need at least "
                f"{seq_len + pred_horizon} timestamps, got {len(features)}."
            )

        metadata = {
            "num_gpus_facility": self.num_gpus_facility,
            "gpus_per_node": self.gpus_per_node,
            "num_nodes": len(node_csvs),
            "num_timestamps": len(features),
            "num_sequences": len(X),
            "seq_len": seq_len,
            "pred_horizon": pred_horizon,
            "stride": stride,
            "csv_files": node_csvs,
        }

        result = {
            "X": X,
            "Y_power": Y_power,
            "Y_signal": Y_signal,
            "metadata": metadata,
        }

        if output_path:
            self._save_npz(X, Y_power, Y_signal, metadata, output_path)

        self.features_df = features
        return result

    # ------------------------------------------------------------------
    # CSV I/O & merge
    # ------------------------------------------------------------------

    def _validate_paths(self) -> None:
        missing = [p for p in self.node_csvs if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(
                f"CSV files not found: {missing}"
            )

    def _load_and_merge(self) -> pd.DataFrame:
        """Load all CSVs, concat per-GPU rows, group by timestamp."""
        all_rows: List[pd.DataFrame] = []
        total_gpus = 0

        for node_idx, csv_path in enumerate(self.node_csvs):
            df = pd.read_csv(csv_path)
            if df.empty:
                logger.warning("empty CSV skipped", extra={"path": csv_path})
                continue

            df["node_id"] = node_idx
            gpus_in_node = df["gpu_id"].nunique() if "gpu_id" in df.columns else 0
            total_gpus += gpus_in_node

            all_rows.append(df)

        if not all_rows:
            raise ValueError("No data loaded from any CSV.")

        merged = pd.concat(all_rows, ignore_index=True)
        merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="coerce")
        merged = merged.dropna(subset=["timestamp"]).sort_values("timestamp")

        if merged.empty:
            raise ValueError("No valid timestamps after merge.  Check your CSV files.")

        total_gpus = merged["gpu_id"].nunique() if "gpu_id" in merged.columns else total_gpus
        if self.num_gpus_facility is None:
            self.num_gpus_facility = total_gpus

        self.merged_df = merged
        return merged

    # ------------------------------------------------------------------
    # Feature extraction (similar to FormatAdapter but across all nodes)
    # ------------------------------------------------------------------

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate per-GPU rows across all nodes → 15 features."""
        required = {"power_w", "temp_c", "util_pct", "mem_util_pct"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in telemetry data: {missing}")

        agg = df.groupby("timestamp").agg({
            "power_w": "sum",
            "temp_c": ["mean", "max"],
            "util_pct": "mean",
            "mem_util_pct": "mean",
        }).reset_index()

        agg.columns = [
            "timestamp", "facility_power_w",
            "gpu_avg_temp_c", "gpu_max_temp_c",
            "gpu_avg_util_pct", "gpu_avg_mem_util_pct",
        ]

        n = len(agg)
        facility_mw = agg["facility_power_w"].values / 1e6
        gpu_avg_power = facility_mw.copy()
        gpu_max_power = facility_mw.copy()
        gpu_avg_temp = agg["gpu_avg_temp_c"].values
        gpu_max_temp = agg["gpu_max_temp_c"].values
        gpu_avg_util = agg["gpu_avg_util_pct"].values
        gpu_avg_mem_util = agg["gpu_avg_mem_util_pct"].values

        power_roc = np.diff(facility_mw, prepend=facility_mw[0])
        power_roc2 = np.diff(power_roc, prepend=power_roc[0])

        window = min(self.rolling_window, n)
        if window > 1:
            series = pd.Series(facility_mw)
            power_roll_mean = series.rolling(window, min_periods=1).mean().values
            power_roll_std = series.rolling(window, min_periods=1).std().fillna(0).values
        else:
            power_roll_mean = facility_mw.copy()
            power_roll_std = np.zeros(n)

        gpu_avg_power_norm = gpu_avg_power / _GPU_TDP_W * 1000.0 / 1e6
        gpu_max_power_norm = gpu_max_power / _GPU_TDP_W * 1000.0 / 1e6

        gpu_avg_temp_norm = gpu_avg_temp / _MAX_TEMP
        gpu_max_temp_norm = gpu_max_temp / _MAX_TEMP
        gpu_avg_util_norm = gpu_avg_util / 100.0
        gpu_avg_mem_util_norm = gpu_avg_mem_util / 100.0
        cpu_util_est_norm = np.full(n, self.cpu_util_estimate)

        hours = agg["timestamp"].dt.hour + agg["timestamp"].dt.minute / 60.0
        hour_sin = np.sin(2 * np.pi * hours.values / 24)
        hour_cos = np.cos(2 * np.pi * hours.values / 24)

        is_allreduce = (
            (gpu_avg_util > 80) & (gpu_avg_mem_util < 30)
        ).astype(np.float64)

        features = pd.DataFrame({
            "facility_mw": facility_mw,
            "power_roc": power_roc,
            "power_roc2": power_roc2,
            "power_roll_mean": power_roll_mean,
            "power_roll_std": power_roll_std,
            "gpu_avg_power_norm": gpu_avg_power_norm,
            "gpu_max_power_norm": gpu_max_power_norm,
            "gpu_avg_temp_norm": gpu_avg_temp_norm,
            "gpu_max_temp_norm": gpu_max_temp_norm,
            "gpu_avg_util_norm": gpu_avg_util_norm,
            "gpu_avg_mem_util_norm": gpu_avg_mem_util_norm,
            "cpu_util_est_norm": cpu_util_est_norm,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "is_allreduce": is_allreduce,
            "timestamp": agg["timestamp"].values,
        })

        for col in FEATURE_NAMES:
            features[col] = np.nan_to_num(features[col], nan=0.0, posinf=0.0, neginf=0.0)

        features.attrs["total_gpus"] = self.num_gpus_facility
        features.attrs["num_nodes"] = len(self.node_csvs)

        self.features_df = features
        return features

    # ------------------------------------------------------------------
    # Sequence creation
    # ------------------------------------------------------------------

    def _create_sequences(
        self,
        features: pd.DataFrame,
        seq_len: int,
        pred_horizon: int,
        stride: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create (X, Y_power, Y_signal) sequences from feature DataFrame."""
        arrays = {name: features[name].values for name in FEATURE_NAMES}
        feature_matrix = np.column_stack([arrays[name] for name in FEATURE_NAMES])
        targets_mw = arrays["facility_mw"].copy()

        power_change = np.diff(targets_mw, prepend=targets_mw[0])
        signals = np.zeros(len(targets_mw), dtype=int)
        signals[power_change > 0.5] = 1
        signals[power_change < -0.5] = 2

        X_list, Yp_list, Ys_list = [], [], []
        for i in range(0, len(feature_matrix) - seq_len - pred_horizon, stride):
            X_list.append(feature_matrix[i:i + seq_len])
            Yp_list.append(targets_mw[i + seq_len:i + seq_len + pred_horizon])
            Ys_list.append(signals[i + seq_len])

        return (
            np.array(X_list, dtype=np.float32),
            np.array(Yp_list, dtype=np.float32),
            np.array(Ys_list, dtype=np.int64),
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    @staticmethod
    def _save_npz(
        X: np.ndarray,
        Y_power: np.ndarray,
        Y_signal: np.ndarray,
        metadata: Dict[str, object],
        output_path: str,
    ) -> str:
        """Save sequences to .npz with metadata."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(path),
            X=X,
            Y_power=Y_power,
            Y_signal=Y_signal,
            metadata=json.dumps(
                {k: v for k, v in metadata.items() if not isinstance(k, np.integer)},
                default=str,
            ),
        )
        logger.info(
            "cluster dataset saved",
            extra={
                "path": str(path),
                "sequences": len(X),
                "gpus": metadata.get("num_gpus_facility"),
                "nodes": metadata.get("num_nodes"),
            },
        )
        return str(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="energivanu-cluster-merge",
        description="Merge per-node telemetry CSVs into training .npz",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s node0.csv node1.csv node2.csv -o data/cluster/merged.npz
  %(prog)s data/cluster/nodes/*.csv --seq-len 60 --pred-horizon 20
        """,
    )
    parser.add_argument("node_csvs", nargs="+", help="Per-node telemetry CSV files")
    parser.add_argument("--output", "-o", default="data/cluster/merged.npz",
                        help="Output .npz path")
    parser.add_argument("--seq-len", type=int, default=30,
                        help="Sequence length (default: 30)")
    parser.add_argument("--pred-horizon", type=int, default=10,
                        help="Prediction horizon (default: 10)")
    parser.add_argument("--stride", type=int, default=50,
                        help="Sequence stride (default: 50)")
    parser.add_argument("--cpu-util", type=float, default=0.4,
                        help="CPU util estimate 0-1 (default: 0.4)")
    parser.add_argument("--gpus-per-node", type=int, default=8,
                        help="GPUs per node (default: 8)")
    parser.add_argument("--rolling-window", type=int, default=250,
                        help="Rolling stats window (default: 250)")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    merger = ClusterMerger(
        gpus_per_node=args.gpus_per_node,
        cpu_util_estimate=args.cpu_util,
        rolling_window=args.rolling_window,
    )

    try:
        result = merger.merge_and_convert(
            node_csvs=args.node_csvs,
            output_path=args.output,
            seq_len=args.seq_len,
            pred_horizon=args.pred_horizon,
            stride=args.stride,
        )
        meta = result["metadata"]
        print(f"Sequences:     {meta['num_sequences']}")
        print(f"GPUs:          {meta['num_gpus_facility']}")
        print(f"Nodes:         {meta['num_nodes']}")
        print(f"Timestamps:    {meta['num_timestamps']}")
        print(f"Output:        {args.output}")
        return 0
    except Exception as exc:
        print(f"Merge failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
