"""
ENERGIVANU — MIT Supercloud Dataset Downloader + Processor
Downloads real GPU telemetry from AWS Open Data, converts to X/Y/S/D format.
Usage: from src.data.real_data import prepare_real_data
"""

import os, sys, subprocess, glob, re, warnings, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List
from src.config import Config

warnings.filterwarnings("ignore")

S3_BUCKET = "s3://mit-supercloud-dataset/datacenter-challenge/202201"


def _aws(cmd: str) -> str:
    """Run aws CLI with --no-sign-request, return stdout."""
    r = subprocess.run(
        f"aws s3 {cmd} --no-sign-request",
        shell=True, capture_output=True, text=True, timeout=300
    )
    if r.returncode != 0:
        raise RuntimeError(f"AWS error: {r.stderr.strip()}")
    return r.stdout


def list_gpu_files(max_files: int = 200) -> List[str]:
    """List available GPU job CSV paths from S3 (sorted, newest first)."""
    raw = _aws(f"ls {S3_BUCKET}/gpu/ --recursive")
    files = []
    for line in raw.strip().split("\n"):
        parts = line.split()
        if len(parts) >= 4 and parts[-1].endswith(".csv"):
            files.append(parts[-1])
    files.sort(reverse=True)
    return files[:max_files]


def download_jobs(s3_paths: List[str], dest_dir: str, max_files: int = 50):
    """Download a batch of GPU CSV files from S3."""
    os.makedirs(dest_dir, exist_ok=True)
    existing = set(os.listdir(dest_dir))
    count = 0
    for s3p in s3_paths:
        fname = os.path.basename(s3p)
        if fname in existing:
            continue
        try:
            _aws(f"cp {S3_BUCKET}/{s3p} \"{dest_dir}/{fname}\"")
            count += 1
            if count >= max_files:
                break
        except Exception as e:
            print(f"  Skip {fname}: {e}")
    print(f"  Downloaded {count} new files ({len(os.listdir(dest_dir))} total)")
    return sorted(glob.glob(f"{dest_dir}/*.csv"))


def load_job_csv(path: str) -> Optional[pd.DataFrame]:
    """Parse a single GPU job CSV, aggregate GPUs per timestamp."""
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    required = ["timestamp", "gpu_index", "power_draw_W",
                "utilization_gpu_pct", "temperature_gpu"]
    if not all(c in df.columns for c in required):
        return None

    # Aggregate: across GPUs, sum power, avg util/temp/memory
    agg = df.groupby("timestamp").agg(
        power_draw_W=("power_draw_W", "sum"),
        gpu_util_pct=("utilization_gpu_pct", "mean"),
        mem_util_pct=("utilization_memory_pct", "mean") if "utilization_memory_pct" in df.columns else ("utilization_gpu_pct", lambda x: 0.0),
        temp_c=("temperature_gpu", "mean"),
    ).reset_index()

    agg = agg.sort_values("timestamp")
    # 100ms sampling → 10Hz
    agg["ElapsedTime"] = np.arange(len(agg)) * 0.1
    return agg


def stitch_jobs(files: List[str]) -> pd.DataFrame:
    """Stitch multiple job traces into a continuous dataframe."""
    pieces = []
    last_time = 0.0
    for path in files:
        df = load_job_csv(path)
        if df is None or len(df) < 200:
            continue
        # Add idle gap between jobs (30s at idle power)
        gap = pd.DataFrame({
            "ElapsedTime": [0],
            "power_draw_W": [0.0],
            "gpu_util_pct": [0.0],
            "mem_util_pct": [0.0],
            "temp_c": [30.0],
        })
        gap["ElapsedTime"] = last_time + 30.0
        pieces.append(gap)
        df["ElapsedTime"] += last_time + 30.0
        pieces.append(df)
        last_time = df["ElapsedTime"].iloc[-1]

    if not pieces:
        return pd.DataFrame()
    full = pd.concat(pieces, ignore_index=True)
    return full


def add_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add rolling features matching our synthetic pipeline."""
    t = df["ElapsedTime"].values
    hours = t / 3600
    days = hours / 24

    gpu_power_mw = df["power_draw_W"].values * 150_000 / 1e6  # scale to 150K GPUs
    gpu_load_pct = df["gpu_util_pct"].values
    gpu_temp_c = df["temp_c"].values

    out = pd.DataFrame({
        "gpu_power_mw": gpu_power_mw,
        "gpu_load_pct": gpu_load_pct,
        "gpu_temp_c": gpu_temp_c,
        "hour": hours % 24,
        "dow": days % 7,
    })

    # Rolling windows
    w5 = int(5 / 0.1)    # 5 seconds
    w30 = int(30 / 0.1)  # 30 seconds
    w60 = int(60 / 0.1)  # 60 seconds
    w300 = int(300 / 0.1)  # 5 minutes

    s = pd.Series(gpu_power_mw)
    out["pm_5s"] = s.rolling(w5, min_periods=1).mean()
    out["ps_5s"] = s.rolling(w5, min_periods=1).std().fillna(0)
    out["pm_30s"] = s.rolling(w30, min_periods=1).mean()
    out["ps_30s"] = s.rolling(w30, min_periods=1).std().fillna(0)
    out["pm_60s"] = s.rolling(w60, min_periods=1).mean()
    out["ps_60s"] = s.rolling(w60, min_periods=1).std().fillna(0)
    out["pwr_rate"] = s.diff().fillna(0) / 0.1

    return out


def prepare(cfg: Config, data_dir: str = None,
            max_jobs: int = 50, stride: int = 1,
            force_download: bool = False
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Full pipeline: download → stitch → feature-engineer → X/Y/S/D."""
    if data_dir is None:
        data_dir = "/kaggle/working/energivanu_prod/real_data"
    os.makedirs(data_dir, exist_ok=True)

    # --- Check cache ---
    cache_path = f"{data_dir}/dataset.pkl"
    if not force_download and os.path.exists(cache_path):
        print("  Loading cached real dataset...")
        return pickle.load(open(cache_path, "rb"))

    # --- List & download ---
    print("  Listing GPU job files on S3...")
    try:
        s3_files = list_gpu_files(200)
    except RuntimeError as e:
        print(f"  S3 access failed: {e}")
        raise

    print(f"  Found {len(s3_files)} job files. Downloading up to {max_jobs}...")
    local_files = download_jobs(s3_files, f"{data_dir}/gpu_csv", max_jobs)

    # --- Stitch ---
    print("  Stitching job traces...")
    df = stitch_jobs(local_files)
    if len(df) < 1000:
        raise RuntimeError(f"Too few samples ({len(df)}) after stitching")

    # --- Features ---
    print(f"  Adding features ({len(df)} rows)...")
    feat = add_features(df, cfg)

    # --- Downsample via stride to match synthetic resolution ---
    cols = [c for c in feat.columns]
    F_full = feat[cols].values.astype(np.float32)
    F = F_full[::stride]
    T = F[:, 0].copy()
    print(f"  Downsampled: stride={stride}, {len(F_full)}→{len(F)} samples ({len(F_full)*0.1/3600:.1f}h → {len(F)*5/3600:.1f}h)")

    lb = cfg.model.lookback
    hz = cfg.model.horizon

    X, Y, S, D = [], [], [], []
    for i in range(lb, len(F) - hz):
        window = F[i - lb:i]
        mu = window.mean(0, keepdims=True)
        sd = window.std(0, keepdims=True) + 1e-8
        X.append((window - mu) / sd)
        future = T[i:i + hz]
        Y.append(future)
        mx = future.max()
        S.append(2 if mx >= cfg.signal.critical_mw
                 else 1 if mx >= cfg.signal.warning_mw else 0)
        D.append(1 if future[-1] > future[0] else 0)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    S = np.array(S, dtype=np.int64)
    D = np.array(D, dtype=np.int64)

    print(f"  X: {X.shape} | Y: {Y.shape} | Samples: {len(X)}")
    cfg.model.num_features = X.shape[2]

    # --- Cache ---
    pickle.dump((X, Y, S, D), open(cache_path, "wb"))
    print(f"  Cached to {cache_path}")
    return X, Y, S, D
