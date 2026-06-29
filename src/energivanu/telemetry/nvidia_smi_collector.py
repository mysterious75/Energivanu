# SPDX-License-Identifier: AGPL-3.0-or-later
"""
NVIDIA-SMI Telemetry Collector
================================
Collects GPU power, temperature, utilization, and clock data via
``nvidia-smi`` XML output.  Stores data in SQLite and/or CSV, maintains a
rolling window buffer, and extracts features compatible with
:mod:`energivanu.data`.

Features (15, matching ``data.py``):
    0.  facility_mw          — total GPU power scaled to facility level
    1.  power_roc            — first derivative of power (MW/s)
    2.  power_roc2           — second derivative of power
    3.  power_roll_mean      — rolling mean of power
    4.  power_roll_std       — rolling std of power
    5.  gpu_avg_power_norm   — avg GPU power / 700 W
    6.  gpu_max_power_norm   — max GPU power / 700 W
    7.  gpu_avg_temp_norm    — avg temperature / 100 °C
    8.  gpu_max_temp_norm    — max temperature / 100 °C
    9.  gpu_avg_util_norm    — avg SM utilization / 100
    10. gpu_avg_mem_util_norm — avg memory utilization / 100
    11. cpu_util_est_norm    — estimated CPU utilization / 100
    12. hour_sin             — cyclical hour encoding (sin)
    13. hour_cos             — cyclical hour encoding (cos)
    14. is_allreduce         — heuristic all-reduce detection flag

Usage::

    from energivanu.telemetry import NvidiaSmiCollector

    collector = NvidiaSmiCollector()
    collector.start()
    # ... later ...
    features = collector.get_feature_vector()
    collector.stop()
"""

from __future__ import annotations

import csv
import math
import sqlite3
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from ..logging_config import get_logger, timed

logger = get_logger("telemetry")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GPU_TDP_W = 700.0  # H100 TDP for normalization
_NUM_SMI_FEATURES = 15

# SQL for table creation
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS gpu_telemetry (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    unix_ts         REAL    NOT NULL,
    gpu_id          INTEGER NOT NULL,
    power_w         REAL,
    temp_c          REAL,
    util_pct        REAL,
    mem_util_pct    REAL,
    sm_clock_mhz    REAL,
    mem_clock_mhz   REAL
);
CREATE INDEX IF NOT EXISTS idx_telemetry_ts ON gpu_telemetry(unix_ts);
CREATE INDEX IF NOT EXISTS idx_telemetry_gpu ON gpu_telemetry(gpu_id);
"""

_INSERT_SQL = """
INSERT INTO gpu_telemetry
    (timestamp, unix_ts, gpu_id, power_w, temp_c, util_pct,
     mem_util_pct, sm_clock_mhz, mem_clock_mhz)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

_CSV_HEADERS = [
    "timestamp", "unix_ts", "gpu_id", "power_w", "temp_c",
    "util_pct", "mem_util_pct", "sm_clock_mhz", "mem_clock_mhz",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GpuSample:
    """A single GPU telemetry sample."""
    gpu_id: int
    power_w: float
    temp_c: float
    util_pct: float
    mem_util_pct: float
    sm_clock_mhz: float
    mem_clock_mhz: float
    timestamp: str = ""
    unix_ts: float = 0.0

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if self.unix_ts == 0.0:
            self.unix_ts = time.time()


@dataclass
class AggregatedSample:
    """Aggregated metrics across all GPUs at a single point in time."""
    timestamp: str
    unix_ts: float
    node_power_w: float
    gpu_avg_power_w: float
    gpu_max_power_w: float
    gpu_std_power_w: float
    gpu_avg_temp_c: float
    gpu_max_temp_c: float
    gpu_avg_util_pct: float
    gpu_avg_mem_util_pct: float
    gpu_samples: List[GpuSample] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_nvidia_smi_xml(xml_output: str) -> List[GpuSample]:
    """
    Parse ``nvidia-smi -q -x`` XML output into :class:`GpuSample` objects.

    Args:
        xml_output: Raw XML string from nvidia-smi.

    Returns:
        List of per-GPU samples.

    Raises:
        ValueError: If XML parsing fails.
    """
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(xml_output)
    except ET.ParseError as exc:
        raise ValueError(f"Failed to parse nvidia-smi XML: {exc}") from exc

    now_ts = time.time()
    now_str = datetime.now(timezone.utc).isoformat()
    samples: List[GpuSample] = []

    for gpu_elem in root.findall(".//gpu"):
        gpu_id_str = gpu_elem.findtext("minor_number", "0")
        try:
            gpu_id = int(gpu_id_str)
        except (ValueError, TypeError):
            gpu_id = 0

        # Power
        power_str = gpu_elem.findtext(".//power_readings/power_draw", "N/A")
        power_w = _parse_float(power_str.replace(" W", "").replace("N/A", "0"))

        # Temperature
        temp_str = gpu_elem.findtext(".//temperature/gpu_temp", "N/A")
        temp_c = _parse_float(temp_str.replace(" C", "").replace("N/A", "0"))

        # GPU utilization
        util_str = gpu_elem.findtext(".//utilization/gpu_util", "N/A")
        util_pct = _parse_float(util_str.replace(" %", "").replace("N/A", "0"))

        # Memory utilization
        mem_util_str = gpu_elem.findtext(".//utilization/memory_util", "N/A")
        mem_util_pct = _parse_float(mem_util_str.replace(" %", "").replace("N/A", "0"))

        # SM clock
        sm_clock_str = gpu_elem.findtext(".//clocks/graphics_clock", "N/A")
        sm_clock_mhz = _parse_float(sm_clock_str.replace(" MHz", "").replace("N/A", "0"))

        # Memory clock
        mem_clock_str = gpu_elem.findtext(".//clocks/mem_clock", "N/A")
        mem_clock_mhz = _parse_float(mem_clock_str.replace(" MHz", "").replace("N/A", "0"))

        samples.append(GpuSample(
            gpu_id=gpu_id,
            power_w=power_w,
            temp_c=temp_c,
            util_pct=util_pct,
            mem_util_pct=mem_util_pct,
            sm_clock_mhz=sm_clock_mhz,
            mem_clock_mhz=mem_clock_mhz,
            timestamp=now_str,
            unix_ts=now_ts,
        ))

    return samples


def _parse_float(s: str) -> float:
    """Safely parse a float, returning 0.0 on failure."""
    try:
        return float(s.strip())
    except (ValueError, TypeError, AttributeError):
        return 0.0


# ---------------------------------------------------------------------------
# Simulation mode
# ---------------------------------------------------------------------------

class _SimulatedGpuSource:
    """
    Generates synthetic GPU telemetry for environments without real GPUs.

    Simulates compute/all-reduce cycles typical of distributed LLM training.
    """

    def __init__(self, num_gpus: int = 8, noise_std: float = 5.0):
        self.num_gpus = num_gpus
        self.noise_std = noise_std
        self._step = 0
        self._compute_power = 500.0  # W during compute
        self._allreduce_power = 200.0  # W during all-reduce
        self._cycle_len = 10  # steps per cycle

    def collect(self) -> List[GpuSample]:
        """Generate one round of simulated samples."""
        now_str = datetime.now(timezone.utc).isoformat()
        now_ts = time.time()
        pos = self._step % self._cycle_len
        is_allreduce = pos == self._cycle_len - 1

        samples: List[GpuSample] = []
        for gpu_id in range(self.num_gpus):
            base_power = self._allreduce_power if is_allreduce else self._compute_power
            power_w = base_power + np.random.normal(0, self.noise_std)
            power_w = max(50.0, min(700.0, power_w))

            base_temp = 75.0 if not is_allreduce else 65.0
            temp_c = base_temp + np.random.normal(0, 2.0)
            temp_c = max(30.0, min(95.0, temp_c))

            util_pct = (90.0 if not is_allreduce else 30.0) + np.random.normal(0, 5.0)
            util_pct = max(0.0, min(100.0, util_pct))

            mem_util_pct = (20.0 if is_allreduce else 80.0) + np.random.normal(0, 3.0)
            mem_util_pct = max(0.0, min(100.0, mem_util_pct))

            sm_clock = 1980.0 + np.random.normal(0, 30.0)
            mem_clock = 2619.0 + np.random.normal(0, 10.0)

            samples.append(GpuSample(
                gpu_id=gpu_id,
                power_w=round(power_w, 2),
                temp_c=round(temp_c, 1),
                util_pct=round(util_pct, 1),
                mem_util_pct=round(mem_util_pct, 1),
                sm_clock_mhz=round(sm_clock, 0),
                mem_clock_mhz=round(mem_clock, 0),
                timestamp=now_str,
                unix_ts=now_ts,
            ))

        self._step += 1
        return samples


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class NvidiaSmiCollector:
    """
    Thread-safe GPU telemetry collector using nvidia-smi.

    Collects power, temperature, utilization, and clock data at a configurable
    interval.  Stores to SQLite and/or CSV.  Maintains a rolling window buffer
    and extracts 15-feature vectors compatible with the Energivanu model.

    Args:
        collection_interval_s: Seconds between collection rounds.
        gpu_ids: List of GPU IDs to monitor.  ``None`` = all GPUs.
        storage_backend: ``"sqlite"``, ``"csv"``, or ``"both"``.
        sqlite_path: Path to SQLite database file.
        csv_path: Path to CSV output file.
        rolling_window_size: Number of samples to keep in the rolling buffer.
        simulation_mode: If ``True``, use simulated GPU data (no real GPU needed).
        simulation_num_gpus: Number of GPUs to simulate.
        num_gpus_facility: Total GPUs in facility for MW scaling.
        gpus_per_node: GPUs per node for scaling.
    """

    def __init__(
        self,
        collection_interval_s: float = 1.0,
        gpu_ids: Optional[List[int]] = None,
        storage_backend: str = "sqlite",
        sqlite_path: str = "data/telemetry.db",
        csv_path: str = "data/telemetry.csv",
        rolling_window_size: int = 3600,
        simulation_mode: bool = False,
        simulation_num_gpus: int = 8,
        num_gpus_facility: int = 200_000,
        gpus_per_node: int = 8,
    ):
        self.collection_interval_s = collection_interval_s
        self.gpu_ids = gpu_ids
        self.storage_backend = storage_backend
        self.sqlite_path = sqlite_path
        self.csv_path = csv_path
        self.rolling_window_size = rolling_window_size
        self.simulation_mode = simulation_mode
        self.simulation_num_gpus = simulation_num_gpus
        self.num_gpus_facility = num_gpus_facility
        self.gpus_per_node = gpus_per_node

        # Internal state
        self._buffer: Deque[AggregatedSample] = deque(maxlen=rolling_window_size)
        self._raw_buffer: Deque[List[GpuSample]] = deque(maxlen=rolling_window_size)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._collection_count: int = 0
        self._error_count: int = 0

        # Storage
        self._sqlite_conn: Optional[sqlite3.Connection] = None
        self._csv_file: Optional[Any] = None
        self._csv_writer: Optional[csv.writer] = None

        # Simulation source
        self._sim_source: Optional[_SimulatedGpuSource] = None

        # Detect real GPU availability
        if not simulation_mode:
            self.simulation_mode = not self._has_nvidia_smi()
            if self.simulation_mode:
                logger.warning(
                    "nvidia-smi not available, falling back to simulation mode"
                )

        if self.simulation_mode:
            self._sim_source = _SimulatedGpuSource(
                num_gpus=self.simulation_num_gpus
            )
            logger.info(
                "telemetry collector created (simulation mode)",
                extra={"num_gpus": self.simulation_num_gpus},
            )
        else:
            logger.info(
                "telemetry collector created",
                extra={"gpu_ids": self.gpu_ids, "interval_s": self.collection_interval_s},
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Start background telemetry collection.

        Raises:
            RuntimeError: If the collector is already running.
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Collector is already running")

        self._init_storage()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._collection_loop,
            name="telemetry-collector",
            daemon=True,
        )
        self._thread.start()
        logger.info("telemetry collection started")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the background collection thread and close storage."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        self._close_storage()
        logger.info(
            "telemetry collection stopped",
            extra={
                "total_collections": self._collection_count,
                "errors": self._error_count,
            },
        )

    def is_running(self) -> bool:
        """Return ``True`` if the collection thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def __enter__(self) -> "NvidiaSmiCollector":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def get_latest(self) -> Optional[AggregatedSample]:
        """Return the most recent aggregated sample, or ``None``."""
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def get_buffer(self) -> List[AggregatedSample]:
        """Return a snapshot of the rolling buffer (oldest first)."""
        with self._lock:
            return list(self._buffer)

    def get_raw_buffer(self) -> List[List[GpuSample]]:
        """Return raw per-GPU samples for the rolling window."""
        with self._lock:
            return list(self._raw_buffer)

    @timed("telemetry.feature_extraction")
    def get_feature_vector(
        self,
        num_gpus_facility: Optional[int] = None,
        gpus_per_node: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """
        Extract a 15-feature vector from the current rolling buffer.

        The features match the format expected by
        :func:`energivanu.data.create_sequences`.

        Args:
            num_gpus_facility: Override for facility GPU count.
            gpus_per_node: Override for GPUs per node.

        Returns:
            ``(15,)`` numpy array, or ``None`` if insufficient data.
        """
        n_fac = num_gpus_facility or self.num_gpus_facility
        n_node = gpus_per_node or self.gpus_per_node

        with self._lock:
            if len(self._buffer) < 2:
                return None
            buffer = list(self._buffer)

        # Extract time series
        node_powers = np.array([s.node_power_w for s in buffer])
        avg_powers = np.array([s.gpu_avg_power_w for s in buffer])
        max_powers = np.array([s.gpu_max_power_w for s in buffer])
        avg_temps = np.array([s.gpu_avg_temp_c for s in buffer])
        max_temps = np.array([s.gpu_max_temp_c for s in buffer])
        avg_utils = np.array([s.gpu_avg_util_pct for s in buffer])
        avg_mem_utils = np.array([s.gpu_avg_mem_util_pct for s in buffer])

        # Scale to facility MW
        scale = n_fac / n_node / 1e6
        facility_mw = node_powers * scale
        current_mw = float(facility_mw[-1])

        # Derivatives
        power_roc = float(np.diff(facility_mw[-2:])[0]) if len(facility_mw) >= 2 else 0.0
        if len(facility_mw) >= 3:
            roc_series = np.diff(facility_mw)
            power_roc2 = float(np.diff(roc_series[-2:])[0])
        else:
            power_roc2 = 0.0

        # Rolling stats
        window = min(250, len(facility_mw))
        roll_mean = float(np.mean(facility_mw[-window:]))
        roll_std = float(np.std(facility_mw[-window:]))

        # Time encoding
        now = datetime.now(timezone.utc)
        hour_sin = math.sin(2 * math.pi * now.hour / 24)
        hour_cos = math.cos(2 * math.pi * now.hour / 24)

        # All-reduce heuristic: high util + low mem util
        latest = buffer[-1]
        is_allreduce = 1.0 if (
            latest.gpu_avg_util_pct > 80 and latest.gpu_avg_mem_util_pct < 30
        ) else 0.0

        features = np.array([
            current_mw,
            power_roc,
            power_roc2,
            roll_mean,
            roll_std,
            float(avg_powers[-1]) / _GPU_TDP_W,
            float(max_powers[-1]) / _GPU_TDP_W,
            float(avg_temps[-1]) / 100.0,
            float(max_temps[-1]) / 100.0,
            float(avg_utils[-1]) / 100.0,
            float(avg_mem_utils[-1]) / 100.0,
            0.4,  # CPU util estimate (no direct measurement)
            hour_sin,
            hour_cos,
            is_allreduce,
        ], dtype=np.float32)

        return features

    def get_feature_sequence(self, seq_len: int = 30) -> Optional[np.ndarray]:
        """
        Extract a sequence of feature vectors for model input.

        Args:
            seq_len: Number of time steps.

        Returns:
            ``(seq_len, 15)`` numpy array, or ``None`` if insufficient data.
        """
        with self._lock:
            if len(self._buffer) < seq_len:
                return None
            buffer = list(self._buffer)[-seq_len:]

        n_fac = self.num_gpus_facility
        n_node = self.gpus_per_node
        scale = n_fac / n_node / 1e6

        node_powers = np.array([s.node_power_w for s in buffer])
        facility_mw = node_powers * scale

        avg_powers = np.array([s.gpu_avg_power_w for s in buffer])
        max_powers = np.array([s.gpu_max_power_w for s in buffer])
        avg_temps = np.array([s.gpu_avg_temp_c for s in buffer])
        max_temps = np.array([s.gpu_max_temp_c for s in buffer])
        avg_utils = np.array([s.gpu_avg_util_pct for s in buffer])
        avg_mem_utils = np.array([s.gpu_avg_mem_util_pct for s in buffer])

        # Derivatives
        power_roc = np.diff(facility_mw, prepend=facility_mw[0])
        power_roc2 = np.diff(power_roc, prepend=power_roc[0])

        # Rolling stats (per-step)
        roll_mean = np.convolve(facility_mw, np.ones(5) / 5, mode="same")
        roll_std_arr = np.array([
            float(np.std(facility_mw[max(0, i - 4):i + 1])) for i in range(len(facility_mw))
        ])

        # Time encoding — use buffer timestamps
        hour_sins = []
        hour_coss = []
        for s in buffer:
            try:
                dt = datetime.fromisoformat(s.timestamp)
            except (ValueError, AttributeError):
                dt = datetime.now(timezone.utc)
            h = dt.hour + dt.minute / 60.0
            hour_sins.append(math.sin(2 * math.pi * h / 24))
            hour_coss.append(math.cos(2 * math.pi * h / 24))

        # All-reduce heuristic
        is_ar = ((avg_utils > 80) & (avg_mem_utils < 30)).astype(float)

        seq = np.column_stack([
            facility_mw,
            power_roc,
            power_roc2,
            roll_mean,
            roll_std_arr,
            avg_powers / _GPU_TDP_W,
            max_powers / _GPU_TDP_W,
            avg_temps / 100.0,
            max_temps / 100.0,
            avg_utils / 100.0,
            avg_mem_utils / 100.0,
            np.full(len(buffer), 0.4),  # CPU util estimate
            np.array(hour_sins),
            np.array(hour_coss),
            is_ar,
        ]).astype(np.float32)

        return seq

    def get_stats(self) -> Dict[str, Any]:
        """Return collection statistics."""
        with self._lock:
            return {
                "buffer_size": len(self._buffer),
                "buffer_max": self.rolling_window_size,
                "total_collections": self._collection_count,
                "error_count": self._error_count,
                "is_running": self.is_running(),
                "simulation_mode": self.simulation_mode,
                "collection_interval_s": self.collection_interval_s,
            }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _has_nvidia_smi() -> bool:
        """Check if nvidia-smi is available on the system."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return False

    def _init_storage(self) -> None:
        """Initialize SQLite and/or CSV storage."""
        if self.storage_backend in ("sqlite", "both"):
            db_path = Path(self.sqlite_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._sqlite_conn = sqlite3.connect(
                str(db_path),
                check_same_thread=False,
            )
            self._sqlite_conn.execute("PRAGMA journal_mode=WAL")
            self._sqlite_conn.executescript(_CREATE_TABLE_SQL)
            self._sqlite_conn.commit()
            logger.info("sqlite storage initialized", extra={"path": str(db_path)})

        if self.storage_backend in ("csv", "both"):
            csv_path = Path(self.csv_path)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not csv_path.exists()
            self._csv_file = open(str(csv_path), "a", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._csv_file)
            if write_header:
                self._csv_writer.writerow(_CSV_HEADERS)
                self._csv_file.flush()
            logger.info("csv storage initialized", extra={"path": str(csv_path)})

    def _close_storage(self) -> None:
        """Flush and close storage handles."""
        if self._sqlite_conn is not None:
            self._sqlite_conn.close()
            self._sqlite_conn = None
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    def _collection_loop(self) -> None:
        """Background collection loop (runs in a daemon thread)."""
        logger.info("collection loop started")
        while not self._stop_event.is_set():
            try:
                self._collect_once()
            except Exception as exc:
                self._error_count += 1
                logger.error(
                    "collection error",
                    extra={"error": str(exc), "error_count": self._error_count},
                    exc_info=True,
                )
            self._stop_event.wait(self.collection_interval_s)
        logger.info("collection loop exiting")

    def _collect_once(self) -> None:
        """Execute a single collection round."""
        # Gather samples
        if self.simulation_mode and self._sim_source is not None:
            samples = self._sim_source.collect()
        else:
            samples = self._collect_nvidia_smi()

        if not samples:
            return

        # Filter by gpu_ids if specified
        if self.gpu_ids is not None:
            samples = [s for s in samples if s.gpu_id in self.gpu_ids]

        if not samples:
            return

        # Aggregate
        agg = self._aggregate(samples)

        # Store
        self._store_samples(samples)
        self._store_aggregate(agg)

        # Update buffer
        with self._lock:
            self._raw_buffer.append(samples)
            self._buffer.append(agg)
            self._collection_count += 1

    def _collect_nvidia_smi(self) -> List[GpuSample]:
        """Run nvidia-smi and parse the output."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "-q", "-x"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                logger.warning(
                    "nvidia-smi returned non-zero",
                    extra={"returncode": result.returncode, "stderr": result.stderr[:500]},
                )
                return []
            return parse_nvidia_smi_xml(result.stdout)
        except subprocess.TimeoutExpired:
            logger.warning("nvidia-smi timed out")
            return []
        except FileNotFoundError:
            logger.error("nvidia-smi binary not found")
            return []

    @staticmethod
    def _aggregate(samples: List[GpuSample]) -> AggregatedSample:
        """Aggregate per-GPU samples into a single node-level sample."""
        powers = [s.power_w for s in samples]
        temps = [s.temp_c for s in samples]
        utils = [s.util_pct for s in samples]
        mem_utils = [s.mem_util_pct for s in samples]

        return AggregatedSample(
            timestamp=samples[0].timestamp,
            unix_ts=samples[0].unix_ts,
            node_power_w=sum(powers),
            gpu_avg_power_w=float(np.mean(powers)),
            gpu_max_power_w=float(np.max(powers)),
            gpu_std_power_w=float(np.std(powers)),
            gpu_avg_temp_c=float(np.mean(temps)),
            gpu_max_temp_c=float(np.max(temps)),
            gpu_avg_util_pct=float(np.mean(utils)),
            gpu_avg_mem_util_pct=float(np.mean(mem_utils)),
            gpu_samples=samples,
        )

    def _store_samples(self, samples: List[GpuSample]) -> None:
        """Write raw samples to SQLite and/or CSV."""
        if self._sqlite_conn is not None:
            rows = [
                (s.timestamp, s.unix_ts, s.gpu_id, s.power_w, s.temp_c,
                 s.util_pct, s.mem_util_pct, s.sm_clock_mhz, s.mem_clock_mhz)
                for s in samples
            ]
            try:
                self._sqlite_conn.executemany(_INSERT_SQL, rows)
                self._sqlite_conn.commit()
            except sqlite3.Error as exc:
                logger.error("sqlite write error", extra={"error": str(exc)})

        if self._csv_writer is not None:
            for s in samples:
                self._csv_writer.writerow([
                    s.timestamp, s.unix_ts, s.gpu_id, s.power_w, s.temp_c,
                    s.util_pct, s.mem_util_pct, s.sm_clock_mhz, s.mem_clock_mhz,
                ])
            if self._csv_file is not None:
                self._csv_file.flush()

    def _store_aggregate(self, agg: AggregatedSample) -> None:
        """Log aggregated sample at DEBUG level."""
        logger.debug(
            "aggregated sample",
            extra={
                "node_power_w": round(agg.node_power_w, 1),
                "gpu_avg_power_w": round(agg.gpu_avg_power_w, 1),
                "gpu_avg_temp_c": round(agg.gpu_avg_temp_c, 1),
                "gpu_avg_util_pct": round(agg.gpu_avg_util_pct, 1),
            },
        )


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def create_collector_from_config(config: Any) -> NvidiaSmiCollector:
    """
    Build a :class:`NvidiaSmiCollector` from an
    :class:`energivanu.config.EnergivanuConfig` instance.

    Args:
        config: The global config object.

    Returns:
        Configured collector instance.
    """
    tc = config.telemetry
    return NvidiaSmiCollector(
        collection_interval_s=tc.collection_interval_s,
        gpu_ids=list(tc.gpu_ids),
        storage_backend=tc.storage_backend,
        sqlite_path=tc.sqlite_path,
        csv_path=tc.csv_path,
        rolling_window_size=tc.rolling_window_size,
        simulation_mode=tc.simulation_mode,
        simulation_num_gpus=tc.simulation_num_gpus,
    )
