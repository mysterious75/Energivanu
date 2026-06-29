# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Data Quality Validator
======================
Comprehensive data quality checks for GPU telemetry datasets.

Checks include:

- **NaN detection** — count and percentage of missing values per column
- **Range validation** — flag values outside expected physical bounds
- **Outlier detection** — IQR-based outlier identification
- **Gap detection** — find missing intervals in time-series data
- **Statistical summary** — mean, std, min, max, percentiles per feature

Works with both CSV file paths and pandas DataFrames directly.

Usage::

    from energivanu.data.validator import DataQualityValidator

    validator = DataQualityValidator()
    report = validator.validate("data/telemetry.csv")
    print(report["nan_summary"])
    print(report["outlier_summary"])

    # Or validate a DataFrame directly
    import pandas as pd
    df = pd.read_csv("data/telemetry.csv")
    report = validator.validate(df)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from energivanu.logging_config import get_logger, timed

logger: logging.Logger = get_logger("data.validator")


# ---------------------------------------------------------------------------
# Default feature ranges (physical bounds for GPU telemetry)
# ---------------------------------------------------------------------------

# Maps column name patterns → (min, max, unit)
DEFAULT_FEATURE_RANGES: Dict[str, Tuple[float, float, str]] = {
    # Power features (watts for node-level, MW for facility-level)
    "facility_mw": (0.0, 500.0, "MW"),
    "node_power_W": (0.0, 15000.0, "W"),
    "gpu_avg_power": (0.0, 800.0, "W"),
    "gpu_max_power": (0.0, 800.0, "W"),
    "power_roc": (-500.0, 500.0, "MW/s"),
    "power_roc2": (-200.0, 200.0, "MW/s²"),
    "power_roll_mean": (0.0, 500.0, "MW"),
    "power_roll_std": (0.0, 100.0, "MW"),
    # Temperature (Celsius, normalised)
    "gpu_avg_temp": (0.0, 110.0, "°C"),
    "gpu_max_temp": (0.0, 110.0, "°C"),
    "gpu_avg_temp_norm": (0.0, 1.1, "normalised"),
    "gpu_max_temp_norm": (0.0, 1.1, "normalised"),
    # Utilisation (percent, normalised)
    "gpu_avg_util": (0.0, 100.0, "%"),
    "gpu_avg_util_norm": (0.0, 1.0, "normalised"),
    "gpu_avg_mem_util": (0.0, 100.0, "%"),
    "gpu_avg_mem_util_norm": (0.0, 1.0, "normalised"),
    "cpu_utilization_percent": (0.0, 100.0, "%"),
    # Cyclical features
    "hour_sin": (-1.0, 1.0, "cyclical"),
    "hour_cos": (-1.0, 1.0, "cyclical"),
    "minute_sin": (-1.0, 1.0, "cyclical"),
    # Binary features
    "is_allreduce": (0.0, 1.0, "binary"),
    # Power ratio (normalised)
    "gpu_avg_power_norm": (0.0, 1.2, "normalised"),
    "gpu_max_power_norm": (0.0, 1.2, "normalised"),
    "gpu_std_power": (0.0, 200.0, "W"),
}


@dataclass
class ValidationReport:
    """
    Structured data quality report.

    Attributes:
        is_valid: ``True`` if no critical issues were found.
        total_rows: Number of rows in the dataset.
        total_columns: Number of columns.
        nan_summary: Per-column NaN counts and percentages.
        range_violations: Columns with values outside expected ranges.
        outlier_summary: Per-column outlier counts via IQR method.
        gap_summary: Detected gaps in time-series (if timestamp column present).
        stats: Descriptive statistics per numeric column.
        warnings: Human-readable warning strings.
        errors: Human-readable error strings (critical issues).
    """

    is_valid: bool = True
    total_rows: int = 0
    total_columns: int = 0
    nan_summary: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    range_violations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    outlier_summary: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    gap_summary: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to a plain dictionary for JSON serialisation."""
        return {
            "is_valid": self.is_valid,
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "nan_summary": self.nan_summary,
            "range_violations": self.range_violations,
            "outlier_summary": self.outlier_summary,
            "gap_summary": self.gap_summary,
            "stats": self.stats,
            "warnings": self.warnings,
            "errors": self.errors,
        }


class DataQualityValidator:
    """
    Validate GPU telemetry data for quality issues.

    Args:
        feature_ranges: Custom ``(min, max, unit)`` mapping.  Falls back to
            :data:`DEFAULT_FEATURE_RANGES` for any column not listed.
        iqr_multiplier: IQR multiplier for outlier detection (default 1.5).
        nan_threshold: Fraction of NaN values above which a column is flagged
            as critical (default 0.05 = 5 %).
        gap_threshold_s: Minimum gap in seconds to flag as a time-series gap
            (default 60).

    Example::

        validator = DataQualityValidator(iqr_multiplier=2.0)
        report = validator.validate("data/telemetry.csv")
        if not report.is_valid:
            for err in report.errors:
                print(f"ERROR: {err}")
    """

    def __init__(
        self,
        feature_ranges: Optional[Dict[str, Tuple[float, float, str]]] = None,
        iqr_multiplier: float = 1.5,
        nan_threshold: float = 0.05,
        gap_threshold_s: float = 60.0,
    ) -> None:
        self._ranges = {**DEFAULT_FEATURE_RANGES, **(feature_ranges or {})}
        self._iqr_mult = iqr_multiplier
        self._nan_thresh = nan_threshold
        self._gap_thresh_s = gap_threshold_s

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @timed("data.validator.validate")
    def validate(
        self,
        source: Union[str, Path, pd.DataFrame],
        timestamp_col: str = "timestamp",
    ) -> ValidationReport:
        """
        Run all quality checks on *source*.

        Args:
            source: Path to a CSV file **or** a :class:`~pandas.DataFrame`.
            timestamp_col: Name of the timestamp column for gap detection.

        Returns:
            A :class:`ValidationReport` with all findings.
        """
        df = self._load(source)
        report = ValidationReport(total_rows=len(df), total_columns=len(df.columns))

        logger.info(
            "starting validation",
            extra={"rows": len(df), "columns": len(df.columns)},
        )

        self._check_nan(df, report)
        self._check_ranges(df, report)
        self._check_outliers(df, report)
        self._check_gaps(df, report, timestamp_col)
        self._compute_stats(df, report)

        # Overall verdict
        if report.errors:
            report.is_valid = False
            logger.warning(
                "validation FAILED",
                extra={"errors": len(report.errors), "warnings": len(report.warnings)},
            )
        else:
            logger.info(
                "validation PASSED",
                extra={"warnings": len(report.warnings)},
            )

        return report

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    @staticmethod
    def _load(source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """Load data from CSV path or return DataFrame as-is."""
        if isinstance(source, pd.DataFrame):
            return source.copy()

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        logger.info("loading CSV", extra={"path": str(path)})
        return pd.read_csv(path)

    def _check_nan(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """Detect NaN / missing values per column."""
        nan_counts = df.isnull().sum()
        nan_dict: Dict[str, Dict[str, Any]] = {}

        for col in df.columns:
            count = int(nan_counts[col])
            if count == 0:
                continue
            pct = count / len(df) * 100
            entry: Dict[str, Any] = {
                "count": count,
                "percent": round(pct, 4),
            }
            nan_dict[col] = entry

            if pct / 100 >= self._nan_thresh:
                report.errors.append(
                    f"Column '{col}' has {pct:.2f}% missing values "
                    f"(threshold: {self._nan_thresh * 100:.1f}%)"
                )
            elif pct > 0:
                report.warnings.append(
                    f"Column '{col}' has {pct:.4f}% missing values"
                )

        report.nan_summary = nan_dict
        logger.debug("nan check complete", extra={"columns_with_nan": len(nan_dict)})

    def _check_ranges(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """Flag values outside expected physical bounds."""
        violations: Dict[str, Dict[str, Any]] = {}

        for col in df.columns:
            if col not in self._ranges:
                continue
            if not np.issubdtype(df[col].dtype, np.number):
                continue

            lo, hi, unit = self._ranges[col]
            series = df[col].dropna()
            if series.empty:
                continue

            below = int((series < lo).sum())
            above = int((series > hi).sum())
            if below == 0 and above == 0:
                continue

            entry: Dict[str, Any] = {
                "expected_min": lo,
                "expected_max": hi,
                "unit": unit,
                "actual_min": round(float(series.min()), 6),
                "actual_max": round(float(series.max()), 6),
                "below_count": below,
                "above_count": above,
            }
            violations[col] = entry

            total_violations = below + above
            pct = total_violations / len(series) * 100
            if pct > 1.0:
                report.errors.append(
                    f"Column '{col}': {total_violations} values outside "
                    f"[{lo}, {hi}] {unit} ({pct:.2f}%)"
                )
            else:
                report.warnings.append(
                    f"Column '{col}': {total_violations} values outside "
                    f"[{lo}, {hi}] {unit} ({pct:.4f}%)"
                )

        report.range_violations = violations
        logger.debug("range check complete", extra={"violations": len(violations)})

    def _check_outliers(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """Detect outliers using the IQR method."""
        outlier_dict: Dict[str, Dict[str, Any]] = {}

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 10:
                continue

            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            if iqr == 0:
                continue

            lower_fence = q1 - self._iqr_mult * iqr
            upper_fence = q3 + self._iqr_mult * iqr

            outlier_mask = (series < lower_fence) | (series > upper_fence)
            count = int(outlier_mask.sum())
            if count == 0:
                continue

            pct = count / len(series) * 100
            entry: Dict[str, Any] = {
                "count": count,
                "percent": round(pct, 4),
                "q1": round(q1, 6),
                "q3": round(q3, 6),
                "iqr": round(iqr, 6),
                "lower_fence": round(lower_fence, 6),
                "upper_fence": round(upper_fence, 6),
            }
            outlier_dict[col] = entry

            if pct > 5.0:
                report.warnings.append(
                    f"Column '{col}': {count} outliers ({pct:.2f}%) "
                    f"outside [{lower_fence:.4f}, {upper_fence:.4f}]"
                )

        report.outlier_summary = outlier_dict
        logger.debug("outlier check complete", extra={"columns_with_outliers": len(outlier_dict)})

    def _check_gaps(
        self,
        df: pd.DataFrame,
        report: ValidationReport,
        timestamp_col: str,
    ) -> None:
        """Detect gaps in time-series data."""
        if timestamp_col not in df.columns:
            report.gap_summary = {"detected": False, "reason": "no timestamp column"}
            return

        try:
            ts = pd.to_datetime(df[timestamp_col], errors="coerce")
        except Exception as exc:
            report.gap_summary = {"detected": False, "reason": f"parse error: {exc}"}
            return

        ts = ts.dropna().sort_values().reset_index(drop=True)
        if len(ts) < 2:
            report.gap_summary = {"detected": False, "reason": "insufficient data"}
            return

        diffs = ts.diff().dt.total_seconds().iloc[1:]
        median_interval = float(diffs.median())
        gap_mask = diffs > self._gap_thresh_s

        gaps: List[Dict[str, Any]] = []
        if gap_mask.any():
            gap_indices = diffs[gap_mask].index.tolist()
            for idx in gap_indices:
                gap_entry = {
                    "start": str(ts.iloc[idx - 1]),
                    "end": str(ts.iloc[idx]),
                    "duration_s": round(float(diffs.iloc[idx - 1 if idx > 0 else 0]), 2),
                }
                gaps.append(gap_entry)

            report.gap_summary = {
                "detected": True,
                "gap_count": len(gaps),
                "median_interval_s": round(median_interval, 4),
                "threshold_s": self._gap_thresh_s,
                "gaps": gaps[:50],  # cap to first 50 for readability
            }
            report.warnings.append(
                f"Detected {len(gaps)} time-series gaps "
                f"(>{self._gap_thresh_s}s, median interval: {median_interval:.2f}s)"
            )
        else:
            report.gap_summary = {
                "detected": False,
                "median_interval_s": round(median_interval, 4),
                "threshold_s": self._gap_thresh_s,
            }

        logger.debug("gap check complete", extra={"gaps_found": len(gaps)})

    def _compute_stats(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """Compute descriptive statistics for each numeric column."""
        stats_dict: Dict[str, Dict[str, float]] = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                continue

            stats_dict[col] = {
                "count": int(series.count()),
                "mean": round(float(series.mean()), 6),
                "std": round(float(series.std()), 6),
                "min": round(float(series.min()), 6),
                "p25": round(float(series.quantile(0.25)), 6),
                "p50": round(float(series.quantile(0.50)), 6),
                "p75": round(float(series.quantile(0.75)), 6),
                "max": round(float(series.max()), 6),
                "skew": round(float(series.skew()), 6),
                "kurtosis": round(float(series.kurtosis()), 6),
            }

        report.stats = stats_dict
        logger.debug("stats computed", extra={"columns": len(stats_dict)})
