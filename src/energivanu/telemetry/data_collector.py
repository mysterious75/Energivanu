# SPDX-License-Identifier: AGPL-3.0-or-later
"""
High-Level Data Collection Orchestrator
========================================
Orchestrates GPU telemetry collection with three operational modes:

- **quick** (5 min) — rapid sanity check or debug session
- **standard** (1 hr) — typical training data collection
- **marathon** (8 hr+) — long-running production data gathering

Uses :class:`~energivanu.telemetry.nvidia_smi_collector.NvidiaSmiCollector`
for GPU metrics and :class:`~energivanu.telemetry.codecarbon_tracker.EnergivanuCarbonTracker`
for energy/cost tracking.

Usage::

    from energivanu.telemetry.data_collector import DataCollector, CollectionMode

    # Quick 5-minute test
    collector = DataCollector(mode=CollectionMode.QUICK)
    collector.run()

    # Standard 1-hour collection with custom output
    collector = DataCollector(
        mode=CollectionMode.STANDARD,
        output_dir="data/my_collection",
    )
    collector.run()

    # Marathon mode with custom duration
    collector = DataCollector(
        mode=CollectionMode.MARATHON,
        duration_hours=12.0,
    )
    collector.run()

CLI usage::

    python -m energivanu.telemetry.data_collector quick
    python -m energivanu.telemetry.data_collector standard --output-dir data/run1
    python -m energivanu.telemetry.data_collector marathon --duration 12
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import get_config
from ..logging_config import get_logger, timed
from .codecarbon_tracker import EnergivanuCarbonTracker
from .nvidia_smi_collector import NvidiaSmiCollector

logger = get_logger("telemetry")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

class CollectionMode(Enum):
    """Collection duration presets."""
    QUICK = "quick"
    STANDARD = "standard"
    MARATHON = "marathon"
    CUSTOM = "custom"


_MODE_DURATIONS: Dict[CollectionMode, float] = {
    CollectionMode.QUICK: 5 / 60,       # 5 minutes → hours
    CollectionMode.STANDARD: 1.0,        # 1 hour
    CollectionMode.MARATHON: 8.0,        # 8 hours
}


# ---------------------------------------------------------------------------
# Collection result
# ---------------------------------------------------------------------------

@dataclass
class CollectionResult:
    """Summary of a completed data collection run."""
    mode: str
    duration_requested_hours: float
    duration_actual_hours: float
    total_samples: int
    output_csv: str
    output_db: Optional[str]
    energy_kwh: float
    cost_usd: float
    emissions_kg: float
    errors: int
    simulation_mode: bool
    start_time: str
    end_time: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dict for JSON serialization."""
        return {
            "mode": self.mode,
            "duration_requested_hours": self.duration_requested_hours,
            "duration_actual_hours": self.duration_actual_hours,
            "total_samples": self.total_samples,
            "output_csv": self.output_csv,
            "output_db": self.output_db,
            "energy_kwh": self.energy_kwh,
            "cost_usd": self.cost_usd,
            "emissions_kg": self.emissions_kg,
            "errors": self.errors,
            "simulation_mode": self.simulation_mode,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class DataCollector:
    """
    High-level orchestrator for GPU telemetry collection.

    Manages the lifecycle of :class:`NvidiaSmiCollector` and
    :class:`EnergivanuCarbonTracker`, handles graceful shutdown on
    SIGINT/SIGTERM, and produces a summary report on completion.

    Args:
        mode: Collection mode (quick/standard/marathon/custom).
        duration_hours: Override duration in hours (used with CUSTOM mode
            or to override the preset for the chosen mode).
        output_dir: Directory for output files (CSV, SQLite, summary JSON).
        collection_interval_s: Seconds between telemetry samples.
        simulation_mode: Force simulation mode (no real GPU required).
        electricity_rate: Electricity rate in $/kWh for cost estimation.
    """

    def __init__(
        self,
        mode: CollectionMode = CollectionMode.STANDARD,
        duration_hours: Optional[float] = None,
        output_dir: str = "data/collections",
        collection_interval_s: float = 1.0,
        simulation_mode: Optional[bool] = None,
        electricity_rate: float = 0.12,
    ):
        self.mode = mode
        self.output_dir = Path(output_dir)

        # Determine duration
        if duration_hours is not None:
            self.duration_hours = duration_hours
        elif mode in _MODE_DURATIONS:
            self.duration_hours = _MODE_DURATIONS[mode]
        else:
            self.duration_hours = 1.0

        self.collection_interval_s = collection_interval_s
        self.electricity_rate = electricity_rate

        # Load config for defaults
        try:
            cfg = get_config()
            self._sim_mode = (
                simulation_mode if simulation_mode is not None
                else cfg.telemetry.simulation_mode
            )
            self._num_gpus = cfg.telemetry.simulation_num_gpus
            self._storage_backend = cfg.telemetry.storage_backend
        except Exception:
            self._sim_mode = simulation_mode if simulation_mode is not None else True
            self._num_gpus = 8
            self._storage_backend = "both"

        # Internal components (created on run)
        self._nvidia_collector: Optional[NvidiaSmiCollector] = None
        self._carbon_tracker: Optional[EnergivanuCarbonTracker] = None
        self._shutdown_requested: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @timed("collector.run")
    def run(self) -> CollectionResult:
        """
        Execute a full collection run.

        Installs signal handlers for graceful shutdown, starts telemetry
        collection and energy tracking, waits for the target duration
        (or until interrupted), and returns a summary report.

        Returns:
            :class:`CollectionResult` with statistics and output paths.
        """
        # Prepare output directory
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        csv_path = str(run_dir / "telemetry.csv")
        db_path = (
            str(run_dir / "telemetry.db")
            if self._storage_backend in ("sqlite", "both") else None
        )
        summary_path = str(run_dir / "summary.json")

        logger.info(
            "collection run starting",
            extra={
                "mode": self.mode.value,
                "duration_hours": self.duration_hours,
                "output_dir": str(run_dir),
                "simulation_mode": self._sim_mode,
            },
        )

        # Install signal handlers
        self._shutdown_requested = False
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)

        def _handle_shutdown(signum: int, frame: Any) -> None:
            logger.info("shutdown signal received (signal %d)", signum)
            self._shutdown_requested = True

        signal.signal(signal.SIGINT, _handle_shutdown)
        signal.signal(signal.SIGTERM, _handle_shutdown)

        start_time = datetime.now(timezone.utc)
        start_ts = time.time()

        try:
            # Create components
            self._nvidia_collector = NvidiaSmiCollector(
                collection_interval_s=self.collection_interval_s,
                storage_backend=self._storage_backend,
                csv_path=csv_path,
                sqlite_path=db_path or "data/telemetry.db",
                simulation_mode=self._sim_mode,
                simulation_num_gpus=self._num_gpus,
            )

            self._carbon_tracker = EnergivanuCarbonTracker(
                electricity_rate_usd_per_kwh=self.electricity_rate,
                output_dir=str(run_dir / "carbon"),
            )

            # Start collection
            self._nvidia_collector.start()
            self._carbon_tracker.start_training_tracking()

            # Wait loop
            duration_s = self.duration_hours * 3600
            elapsed = 0.0
            progress_interval = max(60, duration_s / 20)  # ~20 progress logs

            logger.info(
                "collecting telemetry",
                extra={
                    "target_duration_s": duration_s,
                    "interval_s": self.collection_interval_s,
                },
            )

            while elapsed < duration_s and not self._shutdown_requested:
                sleep_time = min(
                    self.collection_interval_s,
                    duration_s - elapsed,
                    progress_interval,
                )
                time.sleep(sleep_time)
                elapsed = time.time() - start_ts

                # Periodic progress log
                if elapsed % progress_interval < self.collection_interval_s * 2:
                    stats = self._nvidia_collector.get_stats()
                    logger.info(
                        "collection progress",
                        extra={
                            "elapsed_min": round(elapsed / 60, 1),
                            "target_min": round(duration_s / 60, 1),
                            "samples": stats["total_collections"],
                            "errors": stats["error_count"],
                        },
                    )

        except Exception as exc:
            logger.error(
                "collection run failed",
                extra={"error": str(exc)},
                exc_info=True,
            )
            raise
        finally:
            # Restore signal handlers
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)

            # Stop components
            if self._nvidia_collector is not None:
                self._nvidia_collector.stop()
            if self._carbon_tracker is not None:
                self._carbon_tracker.stop_training_tracking()

        end_time = datetime.now(timezone.utc)
        elapsed_hours = (time.time() - start_ts) / 3600

        # Gather stats
        nvidia_stats = self._nvidia_collector.get_stats() if self._nvidia_collector else {}
        carbon_stats: Dict[str, Any] = {}
        if self._carbon_tracker:
            try:
                carbon_stats = self._carbon_tracker.stop_training_tracking()
            except Exception:
                pass

        result = CollectionResult(
            mode=self.mode.value,
            duration_requested_hours=self.duration_hours,
            duration_actual_hours=round(elapsed_hours, 4),
            total_samples=nvidia_stats.get("total_collections", 0),
            output_csv=csv_path,
            output_db=db_path,
            energy_kwh=carbon_stats.get("total_energy_kwh", 0.0),
            cost_usd=carbon_stats.get("total_cost_usd", 0.0),
            emissions_kg=carbon_stats.get("total_emissions_kg", 0.0),
            errors=nvidia_stats.get("error_count", 0),
            simulation_mode=self._sim_mode,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
        )

        # Save summary
        self._save_summary(result, summary_path)

        logger.info(
            "collection run complete",
            extra=result.to_dict(),
        )

        return result

    def stop(self) -> None:
        """Request graceful shutdown of a running collection."""
        self._shutdown_requested = True
        logger.info("stop requested")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _save_summary(result: CollectionResult, path: str) -> None:
        """Write collection summary to JSON."""
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(result.to_dict(), fh, indent=2, default=str)
            logger.info("summary saved", extra={"path": str(p)})
        except Exception as exc:
            logger.error("failed to save summary", extra={"error": str(exc)})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="energivanu-collect",
        description="Energivanu GPU telemetry data collector",
    )
    subparsers = parser.add_subparsers(dest="command", help="Collection mode")

    # Shared arguments
    def _add_common_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--output-dir", "-o",
            default="data/collections",
            help="Output directory (default: data/collections)",
        )
        p.add_argument(
            "--interval", "-i",
            type=float, default=1.0,
            help="Collection interval in seconds (default: 1.0)",
        )
        p.add_argument(
            "--simulation", "-s",
            action="store_true",
            default=False,
            help="Force simulation mode (no real GPU required)",
        )
        p.add_argument(
            "--rate",
            type=float, default=0.12,
            help="Electricity rate in $/kWh (default: 0.12)",
        )

    # Quick mode
    quick_parser = subparsers.add_parser(
        "quick", help="5-minute quick collection",
    )
    _add_common_args(quick_parser)

    # Standard mode
    standard_parser = subparsers.add_parser(
        "standard", help="1-hour standard collection",
    )
    _add_common_args(standard_parser)

    # Marathon mode
    marathon_parser = subparsers.add_parser(
        "marathon", help="8+ hour marathon collection",
    )
    _add_common_args(marathon_parser)
    marathon_parser.add_argument(
        "--duration", "-d",
        type=float, default=8.0,
        help="Duration in hours (default: 8.0)",
    )

    # Custom mode
    custom_parser = subparsers.add_parser(
        "custom", help="Custom duration collection",
    )
    _add_common_args(custom_parser)
    custom_parser.add_argument(
        "--duration", "-d",
        type=float, required=True,
        help="Duration in hours (required)",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    CLI entry point for the data collector.

    Args:
        argv: Command-line arguments.  ``None`` uses ``sys.argv[1:]``.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    mode = CollectionMode(args.command)
    duration = getattr(args, "duration", None)

    collector = DataCollector(
        mode=mode,
        duration_hours=duration,
        output_dir=args.output_dir,
        collection_interval_s=args.interval,
        simulation_mode=args.simulation,
        electricity_rate=args.rate,
    )

    try:
        result = collector.run()
        print("\n✅ Collection complete!")
        print(f"   Mode:     {result.mode}")
        print(f"   Duration: {result.duration_actual_hours:.2f} h")
        print(f"   Samples:  {result.total_samples}")
        print(f"   Energy:   {result.energy_kwh:.4f} kWh")
        print(f"   Cost:     ${result.cost_usd:.4f}")
        print(f"   CSV:      {result.output_csv}")
        if result.output_db:
            print(f"   DB:       {result.output_db}")
        return 0
    except KeyboardInterrupt:
        print("\n⏹ Collection interrupted by user")
        return 130
    except Exception as exc:
        logger.error("collection failed", extra={"error": str(exc)}, exc_info=True)
        print(f"\n❌ Collection failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
