# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Alibaba GPU Trace 2020 Processor
=================================
Parse and process the Alibaba GPU Trace 2020 dataset (CC BY 4.0) from the
NSDI '22 paper "Characterizing and Profiling GPU Workloads on Alibaba".

The trace contains per-instance GPU telemetry at ~1-minute resolution:
instance_id, machine_id, timestamp, cpu_util, mem_util, gpu_util,
gpu_mem_util, etc.  This processor maps those columns to the 15-feature
format used by the Energivanu PEB model.

Dataset source:
    https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020

Citation:
    @inproceedings{wen2022characterizing,
        title={Characterizing and Profiling GPU Workloads on Alibaba},
        author={Wen, Mingshu and others},
        booktitle={NSDI},
        year={2022}
    }

Usage::

    from energivanu.data.alibaba_processor import AlibabaTraceProcessor

    processor = AlibabaTraceProcessor(
        data_dir="data/alibaba/cluster-trace-gpu-v2020",
    )
    features_df = processor.process()
    print(features_df.shape)  # (N, 15)

    # Or load and process a single file
    df = processor.load_trace_file("path/to/trace.csv")
    features = processor.extract_features(df)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import get_config
from ..logging_config import get_logger, timed

logger = get_logger("data")

# ---------------------------------------------------------------------------
# Constants — Alibaba trace expected columns
# ---------------------------------------------------------------------------

# Column names in the Alibaba GPU Trace v2020 CSV files
_ALIBABA_COLUMNS: Dict[str, str] = {
    "instance_id": "instance_id",
    "machine_id": "machine_id",
    "timestamp": "timestamp",
    "cpu_util": "cpu_util",
    "mem_util": "mem_util",
    "gpu_util": "gpu_util",
    "gpu_mem_util": "gpu_mem_util",
}

# Normalization constants (matching data.py / nvidia_smi_collector.py)
_GPU_TDP_W: float = 700.0       # H100 TDP for normalization
_MAX_GPU_UTIL: float = 100.0    # max GPU utilization percent
_MAX_CPU_UTIL: float = 100.0    # max CPU utilization percent
_MAX_TEMP: float = 100.0        # max temperature for normalization
_NUM_FEATURES: int = 15

# Column name aliases — different versions of the dataset may use slightly
# different header names.  Map each canonical name to known aliases.
_COLUMN_ALIASES: Dict[str, List[str]] = {
    "instance_id": ["instance_id", "inst_id", "instance", "vid", "inst_name"],
    "machine_id": ["machine_id", "machine", "mid", "host_id", "worker_name"],
    "timestamp": ["timestamp", "ts", "time", "datetime", "start_time"],
    "cpu_util": ["cpu_util", "cpu_util_percent", "avg_cpu", "cpu",
                 "cpu_usage", "machine_cpu_usr", "machine_cpu"],
    "mem_util": ["mem_util", "mem_util_percent", "avg_mem", "mem"],
    "gpu_util": ["gpu_util", "gpu_util_percent", "avg_gpu_util", "gpu",
                 "gpu_wrk_util", "machine_gpu"],
    "gpu_mem_util": ["gpu_mem_util", "gpu_mem_util_percent", "avg_gpu_mem",
                     "gpu_mem", "gpu_memory_util", "avg_gpu_wrk_mem"],
}


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class AlibabaTraceProcessor:
    """
    Process Alibaba GPU Trace 2020 data into Energivanu 15-feature format.

    The processor handles:
    - Multiple CSV files in the trace directory
    - Column name normalization (handles aliases)
    - Missing data imputation
    - Resampling to uniform time intervals
    - Feature extraction matching the PEB model input format

    Args:
        data_dir: Path to directory containing Alibaba trace CSV files.
        resample_interval_s: Target resampling interval in seconds.
            The raw trace is ~60 s; set to ``None`` to skip resampling.
        num_gpus_facility: Total GPU count for facility-level MW scaling.
        gpus_per_node: GPUs per node for scaling.
        max_power_per_gpu_w: Assumed max power per GPU (W) for normalization
            since the Alibaba trace only provides utilization, not power.
        fill_method: How to fill missing values — ``"ffill"`` (forward fill),
            ``"interpolate"``, or ``"zero"``.
    """

    def __init__(
        self,
        data_dir: str = "data/alibaba/cluster-trace-gpu-v2020",
        resample_interval_s: Optional[int] = 60,
        num_gpus_facility: int = 200_000,
        gpus_per_node: int = 8,
        max_power_per_gpu_w: float = 700.0,
        fill_method: str = "interpolate",
    ):
        self.data_dir = Path(data_dir)
        self.resample_interval_s = resample_interval_s
        self.num_gpus_facility = num_gpus_facility
        self.gpus_per_node = gpus_per_node
        self.max_power_per_gpu_w = max_power_per_gpu_w
        self.fill_method = fill_method

        if not self.data_dir.exists():
            logger.warning(
                "alibaba data directory does not exist",
                extra={"path": str(self.data_dir)},
            )

        logger.info(
            "alibaba processor initialized",
            extra={
                "data_dir": str(self.data_dir),
                "resample_interval_s": self.resample_interval_s,
                "fill_method": self.fill_method,
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @timed("alibaba.discover_files")
    def discover_files(self) -> List[Path]:
        """
        Find all CSV trace files in the data directory.

        Returns:
            Sorted list of CSV file paths.
        """
        if not self.data_dir.exists():
            return []

        csv_files: List[Path] = []
        for ext in ("*.csv", "*.csv.gz", "*.parquet"):
            csv_files.extend(self.data_dir.rglob(ext))

        csv_files = sorted(set(csv_files))
        logger.info(
            "discovered trace files",
            extra={"count": len(csv_files), "dir": str(self.data_dir)},
        )
        return csv_files

    @timed("alibaba.load_trace")
    def load_trace_file(self, filepath: str) -> pd.DataFrame:
        """
        Load a single Alibaba trace file and normalize column names.

        Args:
            filepath: Path to a CSV, CSV.GZ, or Parquet file.

        Returns:
            DataFrame with standardized column names.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If required columns are missing after normalization.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Trace file not found: {path}")

        # Load based on extension
        try:
            if path.suffix == ".parquet":
                df = pd.read_parquet(str(path))
            elif path.suffixes[-1:] == [".gz"] or path.name.endswith(".csv.gz"):
                df = pd.read_csv(str(path), compression="gzip")
            else:
                df = pd.read_csv(str(path))
        except Exception as exc:
            logger.error(
                "failed to load trace file",
                extra={"path": str(path), "error": str(exc)},
            )
            raise

        # Normalize column names
        df = self._normalize_columns(df)

        # Generate synthetic timestamps if missing
        if "timestamp" not in df.columns:
            logger.info(
                "no timestamp column found, generating synthetic timestamps "
                "(60s intervals)",
                extra={"path": str(path), "rows": len(df)},
            )
            base_ts = pd.Timestamp("2020-01-01")
            df["timestamp"] = pd.date_range(
                start=base_ts, periods=len(df), freq="60s"
            )

        # Validate required columns
        required = {"timestamp", "gpu_util"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns after normalization: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        logger.debug(
            "trace file loaded",
            extra={"path": str(path), "rows": len(df), "cols": list(df.columns)},
        )
        return df

    @timed("alibaba.process")
    def process(
        self,
        max_files: Optional[int] = None,
        sample_instances: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Discover, load, and process all trace files into 15-feature format.

        Args:
            max_files: Maximum number of files to process.  ``None`` = all.
            sample_instances: If set, randomly sample this many instances
                from the data to reduce memory usage.

        Returns:
            DataFrame with 15 feature columns, sorted by timestamp.
        """
        files = self.discover_files()
        if not files:
            logger.error("no trace files found in %s", self.data_dir)
            return pd.DataFrame()

        if max_files is not None:
            files = files[:max_files]

        all_features: List[pd.DataFrame] = []
        for i, fpath in enumerate(files):
            logger.info(
                "processing file %d/%d", i + 1, len(files),
                extra={"file": fpath.name},
            )
            try:
                df = self.load_trace_file(str(fpath))
                if sample_instances and "instance_id" in df.columns:
                    instances = df["instance_id"].unique()
                    if len(instances) > sample_instances:
                        sampled = np.random.choice(
                            instances, sample_instances, replace=False
                        )
                        df = df[df["instance_id"].isin(sampled)]

                features = self.extract_features(df)
                if not features.empty:
                    all_features.append(features)
            except Exception as exc:
                logger.warning(
                    "skipping file due to error",
                    extra={"file": fpath.name, "error": str(exc)},
                )
                continue

        if not all_features:
            logger.error("no features extracted from any file")
            return pd.DataFrame()

        result = pd.concat(all_features, ignore_index=True)
        result = result.sort_values("timestamp").reset_index(drop=True)

        logger.info(
            "alibaba processing complete",
            extra={"total_rows": len(result), "files_processed": len(all_features)},
        )
        return result

    @timed("alibaba.extract_features")
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract the 15 training features from a normalized Alibaba trace DataFrame.

        The Alibaba trace provides utilization percentages but not power or
        temperature.  We estimate power from utilization using a linear model
        and derive the remaining features from time and utilization patterns.

        Feature mapping:
            0.  facility_mw       — estimated facility power (MW)
            1.  power_roc         — rate of change of power
            2.  power_roc2        — second derivative of power
            3.  power_roll_mean   — rolling mean power
            4.  power_roll_std    — rolling std power
            5.  gpu_avg_power_norm — avg GPU power / TDP
            6.  gpu_max_power_norm — max GPU power / TDP
            7.  gpu_avg_temp_norm  — estimated avg temp / 100 (from util)
            8.  gpu_max_temp_norm  — estimated max temp / 100 (from util)
            9.  gpu_avg_util_norm  — avg GPU util / 100
            10. gpu_avg_mem_util_norm — avg GPU mem util / 100
            11. cpu_util_est_norm — CPU util / 100
            12. hour_sin          — cyclical hour encoding
            13. hour_cos          — cyclical hour encoding
            14. is_allreduce      — all-reduce detection heuristic

        Args:
            df: Normalized Alibaba trace DataFrame.

        Returns:
            DataFrame with 15 named feature columns and a ``timestamp`` column.
        """
        if df.empty:
            return pd.DataFrame()

        # Resample if needed
        if self.resample_interval_s is not None:
            df = self._resample(df)

        # Fill missing values
        df = self._fill_missing(df)

        if df.empty:
            return pd.DataFrame()

        # Aggregate per-timestamp if multiple instances exist
        if "instance_id" in df.columns and df["instance_id"].nunique() > 1:
            df = self._aggregate_instances(df)

        # Ensure sorted by timestamp
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)

        # --- Estimate power from utilization ---
        # Alibaba trace has utilization (%) but not power (W).
        # Use a linear idle-to-peak model: P = idle + (peak - idle) * util/100
        # H100: idle ~70 W, peak ~700 W → P ≈ 70 + 630 * gpu_util/100
        idle_power_w = 70.0
        peak_power_w = self.max_power_per_gpu_w

        gpu_util = df["gpu_util"].values.astype(np.float64)
        gpu_mem_util = (
            df["gpu_mem_util"].values.astype(np.float64)
            if "gpu_mem_util" in df.columns
            else np.full(len(df), 0.0)
        )
        cpu_util = (
            df["cpu_util"].values.astype(np.float64)
            if "cpu_util" in df.columns
            else np.full(len(df), 0.0)
        )
        _mem_util = (
            df["mem_util"].values.astype(np.float64)
            if "mem_util" in df.columns
            else np.full(len(df), 0.0)
        )

        # Per-GPU power estimate (W)
        gpu_power_w = idle_power_w + (peak_power_w - idle_power_w) * (gpu_util / 100.0)
        node_power_w = gpu_power_w * self.gpus_per_node

        # Scale to facility MW
        scale = self.num_gpus_facility / self.gpus_per_node / 1e6
        facility_mw = node_power_w * scale

        # --- Derivatives ---
        power_roc = np.diff(facility_mw, prepend=facility_mw[0])
        power_roc2 = np.diff(power_roc, prepend=power_roc[0])

        # --- Rolling stats ---
        window = min(250, len(facility_mw))
        if window > 1:
            series = pd.Series(facility_mw)
            power_roll_mean = series.rolling(window, min_periods=1).mean().values
            power_roll_std = series.rolling(window, min_periods=1).std().fillna(0).values
        else:
            power_roll_mean = facility_mw.copy()
            power_roll_std = np.zeros_like(facility_mw)

        # --- Time encoding ---
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            hours = ts.dt.hour + ts.dt.minute / 60.0
        else:
            hours = np.zeros(len(df))

        hour_sin = np.sin(2 * np.pi * hours / 24)
        hour_cos = np.cos(2 * np.pi * hours / 24)

        # --- Temperature estimate ---
        # Approximate: temp scales with utilization (no direct measurement)
        # Idle ~40°C, peak ~85°C
        gpu_avg_temp = 40.0 + 45.0 * (gpu_util / 100.0)
        gpu_max_temp = gpu_avg_temp + 5.0  # hot-spot delta

        # --- All-reduce heuristic ---
        is_allreduce = ((gpu_util > 80) & (gpu_mem_util < 30)).astype(np.float64)

        # --- Build feature DataFrame ---
        features = pd.DataFrame({
            "facility_mw": facility_mw,
            "power_roc": power_roc,
            "power_roc2": power_roc2,
            "power_roll_mean": power_roll_mean,
            "power_roll_std": power_roll_std,
            "gpu_avg_power_norm": gpu_power_w / _GPU_TDP_W,
            "gpu_max_power_norm": gpu_power_w / _GPU_TDP_W,  # same source
            "gpu_avg_temp_norm": gpu_avg_temp / _MAX_TEMP,
            "gpu_max_temp_norm": gpu_max_temp / _MAX_TEMP,
            "gpu_avg_util_norm": gpu_util / _MAX_GPU_UTIL,
            "gpu_avg_mem_util_norm": gpu_mem_util / _MAX_GPU_UTIL,
            "cpu_util_est_norm": cpu_util / _MAX_CPU_UTIL,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "is_allreduce": is_allreduce,
        })

        # Carry timestamp for downstream sorting
        if "timestamp" in df.columns:
            features["timestamp"] = df["timestamp"].values

        # Final NaN check
        nan_count = features.drop(columns=["timestamp"], errors="ignore").isna().sum().sum()
        if nan_count > 0:
            logger.warning(
                "NaN values in features — filling with zeros",
                extra={"nan_count": int(nan_count)},
            )
            features = features.fillna(0.0)

        return features

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map aliased column names to canonical names."""
        rename_map: Dict[str, str] = {}
        existing_cols = set(df.columns)

        for canonical, aliases in _COLUMN_ALIASES.items():
            for alias in aliases:
                if alias in existing_cols:
                    rename_map[alias] = canonical
                    break

        df = df.rename(columns=rename_map)
        return df

    def _resample(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample trace data to a uniform time interval.

        Groups by instance_id (if present), resamples, and forward-fills.
        """
        if "timestamp" not in df.columns:
            return df

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])

        if df.empty:
            return df

        # If multiple instances, resample per instance
        if "instance_id" in df.columns and df["instance_id"].nunique() > 1:
            resampled_parts: List[pd.DataFrame] = []
            for inst_id, group in df.groupby("instance_id"):
                group = group.set_index("timestamp").sort_index()
                rule = f"{self.resample_interval_s}s"
                numeric_cols = group.select_dtypes(include=[np.number]).columns
                resampled = group[numeric_cols].resample(rule).ffill()
                resampled["instance_id"] = inst_id
                resampled_parts.append(resampled.reset_index())
            if resampled_parts:
                df = pd.concat(resampled_parts, ignore_index=True)
        else:
            df = df.set_index("timestamp").sort_index()
            rule = f"{self.resample_interval_s}s"
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
            resampled = df[numeric_cols].resample(rule).ffill()
            for col in non_numeric_cols:
                resampled[col] = df[col].resample(rule).first()
            df = resampled.reset_index()

        logger.debug("resampled to %ds intervals", self.resample_interval_s)
        return df

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values using the configured method."""
        if df.empty:
            return df

        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        nan_before = df[numeric_cols].isna().sum().sum()

        if self.fill_method == "ffill":
            df[numeric_cols] = df[numeric_cols].ffill().bfill()
        elif self.fill_method == "interpolate":
            df[numeric_cols] = df[numeric_cols].interpolate(method="linear").bfill().ffill()
        elif self.fill_method == "zero":
            df[numeric_cols] = df[numeric_cols].fillna(0.0)
        else:
            logger.warning("unknown fill_method '%s', using ffill", self.fill_method)
            df[numeric_cols] = df[numeric_cols].ffill().bfill()

        nan_after = df[numeric_cols].isna().sum().sum()
        if nan_before > 0:
            logger.debug(
                "filled missing values",
                extra={"nan_before": int(nan_before), "nan_after": int(nan_after)},
            )

        return df

    def _aggregate_instances(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate multiple instances per timestamp into a single row.

        Takes the mean of numeric columns grouped by timestamp.
        """
        if "timestamp" not in df.columns:
            return df

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if "instance_id" in numeric_cols:
            numeric_cols.remove("instance_id")

        aggregated = df.groupby("timestamp")[numeric_cols].mean().reset_index()

        logger.debug(
            "aggregated instances",
            extra={
                "instances": df["instance_id"].nunique() if "instance_id" in df.columns else 1,
                "timestamps": len(aggregated),
            },
        )
        return aggregated


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def create_alibaba_processor_from_config(config: Any = None) -> AlibabaTraceProcessor:
    """
    Build an :class:`AlibabaTraceProcessor` from the global configuration.

    Args:
        config: An :class:`EnergivanuConfig` instance.  If ``None``, loads
            the singleton config via :func:`get_config`.

    Returns:
        Configured processor instance.
    """
    if config is None:
        config = get_config()

    return AlibabaTraceProcessor(
        num_gpus_facility=config.training.num_gpus_target,
        gpus_per_node=config.training.gpus_per_node,
    )
