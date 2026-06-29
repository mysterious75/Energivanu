# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Data Provenance & Lineage Tracking
====================================
Track the origin, licence, collection date, and integrity hash of every
dataset used in the Energivanu pipeline.

Provenance records are stored as JSON sidecar files alongside the data:

    data/telemetry.csv          ← the data
    data/telemetry.csv.prov.json  ← provenance metadata

This ensures provenance survives file moves, copies, and version control.

Usage::

    from energivanu.data.provenance import (
        register_dataset,
        verify_provenance,
        list_datasets,
    )

    # Register a new dataset
    register_dataset(
        path="data/alibaba_gpu_trace/pai_sensor_table.csv",
        source="alibaba",
        licence="CC BY 4.0",
        collection_date="2020-07-01",
        citation="Weng et al., NSDI '22",
    )

    # Verify integrity before training
    result = verify_provenance("data/alibaba_gpu_trace/pai_sensor_table.csv")
    if result["hash_match"]:
        print("Dataset integrity verified")

    # List all registered datasets
    for ds in list_datasets("data/"):
        print(f"{ds['path']} — {ds['source']} ({ds['licence']})")
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from energivanu.logging_config import get_logger, timed

logger: logging.Logger = get_logger("data.provenance")

# Sidecar file suffix
_PROV_SUFFIX = ".prov.json"

# Known data sources
KNOWN_SOURCES = {
    "alibaba": "Alibaba GPU Trace 2020 (CC BY 4.0)",
    "york": "York University H100 (CC BY-NC-ND 4.0 — research only)",
    "kaggle": "Kaggle / self-collected (public domain)",
    "synthetic": "Synthetically generated",
    "google": "Google Cluster Trace (Apache 2.0)",
    "mit": "MIT Supercloud (CC BY-NC-ND 4.0 — research only)",
    "custom": "Custom collection",
}


@dataclass
class ProvenanceRecord:
    """
    A single provenance record for a dataset file.

    Attributes:
        path: Absolute or relative path to the dataset file.
        source: Data source identifier (alibaba/york/kaggle/synthetic/etc.).
        licence: Licence string (e.g. "CC BY 4.0").
        collection_date: ISO-8601 date when data was collected.
        hash_sha256: SHA-256 hex digest of the file contents.
        hash_algorithm: Hash algorithm used (always "sha256").
        registered_at: ISO-8601 timestamp of when provenance was registered.
        file_size_bytes: Size of the data file in bytes.
        citation: BibTeX or plain-text citation (optional).
        notes: Free-form notes (optional).
        is_commercial_safe: Whether this data may enter commercial pipelines.
        tags: Arbitrary key-value tags for filtering.
    """

    path: str
    source: str
    licence: str
    collection_date: str
    hash_sha256: str
    hash_algorithm: str = "sha256"
    registered_at: str = ""
    file_size_bytes: int = 0
    citation: Optional[str] = None
    notes: Optional[str] = None
    is_commercial_safe: bool = True
    tags: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def _compute_hash(path: Path, chunk_size: int = 8192) -> str:
    """
    Compute the SHA-256 hex digest of a file.

    Args:
        path: Path to the file.
        chunk_size: Read buffer size in bytes.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


# ---------------------------------------------------------------------------
# Sidecar I/O
# ---------------------------------------------------------------------------

def _sidecar_path(data_path: Path) -> Path:
    """Return the provenance sidecar path for a data file."""
    return data_path.with_suffix(data_path.suffix + _PROV_SUFFIX)


def _write_sidecar(record: ProvenanceRecord) -> Path:
    """Write a provenance record to its sidecar file."""
    sidecar = _sidecar_path(Path(record.path))
    sidecar.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(record)
    # Remove None values for cleaner JSON
    data = {k: v for k, v in data.items() if v is not None}

    sidecar.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    logger.debug("wrote sidecar", extra={"path": str(sidecar)})
    return sidecar


def _read_sidecar(data_path: Path) -> Optional[ProvenanceRecord]:
    """Read a provenance sidecar file, or return None if absent."""
    sidecar = _sidecar_path(data_path)
    if not sidecar.exists():
        return None

    try:
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
        return ProvenanceRecord(**raw)
    except (json.JSONDecodeError, TypeError, KeyError) as exc:
        logger.warning(
            "corrupt sidecar file",
            extra={"path": str(sidecar), "error": str(exc)},
        )
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@timed("data.provenance.register")
def register_dataset(
    path: str,
    source: str,
    licence: str,
    collection_date: str,
    citation: Optional[str] = None,
    notes: Optional[str] = None,
    is_commercial_safe: Optional[bool] = None,
    tags: Optional[Dict[str, str]] = None,
) -> ProvenanceRecord:
    """
    Register a dataset's provenance and write the sidecar file.

    The file's SHA-256 hash is computed automatically.

    Args:
        path: Path to the dataset file (relative to project root or absolute).
        source: Data source identifier.  One of the keys in
            :data:`KNOWN_SOURCES` (alibaba, york, kaggle, synthetic, etc.).
        licence: Licence string (e.g. ``"CC BY 4.0"``).
        collection_date: ISO-8601 date string (e.g. ``"2020-07-01"``).
        citation: Optional citation text.
        notes: Optional free-form notes.
        is_commercial_safe: Override automatic safety detection.
            If ``None``, determined from the *source*.
        tags: Arbitrary tags for filtering.

    Returns:
        The written :class:`ProvenanceRecord`.

    Raises:
        FileNotFoundError: If the data file does not exist.

    Example::

        register_dataset(
            path="data/alibaba/pai_sensor_table.csv",
            source="alibaba",
            licence="CC BY 4.0",
            collection_date="2020-07-01",
            citation="Weng et al., NSDI '22",
        )
    """
    data_path = Path(path).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    # Determine commercial safety
    if is_commercial_safe is None:
        nc_sources = {"york", "mit"}
        is_commercial_safe = source.lower() not in nc_sources

    file_hash = _compute_hash(data_path)

    record = ProvenanceRecord(
        path=str(data_path),
        source=source.lower(),
        licence=licence,
        collection_date=collection_date,
        hash_sha256=file_hash,
        registered_at=datetime.now(timezone.utc).isoformat(),
        file_size_bytes=data_path.stat().st_size,
        citation=citation,
        notes=notes,
        is_commercial_safe=is_commercial_safe,
        tags=tags or {},
    )

    _write_sidecar(record)

    logger.info(
        "dataset registered",
        extra={
            "path": str(data_path),
            "source": source,
            "licence": licence,
            "commercial_safe": is_commercial_safe,
            "hash": file_hash[:16] + "...",
        },
    )
    return record


@timed("data.provenance.verify")
def verify_provenance(path: str) -> Dict[str, Any]:
    """
    Verify a dataset's provenance and integrity.

    Checks:
    1. Sidecar file exists.
    2. SHA-256 hash matches the current file contents.

    Args:
        path: Path to the dataset file.

    Returns:
        A dict with keys:

        - ``registered`` (bool) — whether a sidecar exists
        - ``hash_match`` (bool) — whether the hash matches
        - ``record`` (dict or None) — the provenance record
        - ``warnings`` (list[str]) — any issues found

    Example::

        result = verify_provenance("data/telemetry.csv")
        if not result["registered"]:
            print("WARNING: unregistered dataset!")
        elif not result["hash_match"]:
            print("WARNING: file has been modified since registration!")
    """
    data_path = Path(path).resolve()
    result: Dict[str, Any] = {
        "registered": False,
        "hash_match": False,
        "record": None,
        "warnings": [],
    }

    record = _read_sidecar(data_path)
    if record is None:
        result["warnings"].append(f"No provenance sidecar found for {path}")
        logger.warning("no provenance", extra={"path": str(data_path)})
        return result

    result["registered"] = True
    result["record"] = asdict(record)

    if not data_path.exists():
        result["warnings"].append("Data file does not exist (may have been moved)")
        return result

    current_hash = _compute_hash(data_path)
    result["hash_match"] = (current_hash == record.hash_sha256)

    if not result["hash_match"]:
        result["warnings"].append(
            f"Hash mismatch: expected {record.hash_sha256[:16]}..., "
            f"got {current_hash[:16]}..."
        )
        logger.warning(
            "hash mismatch",
            extra={"path": str(data_path), "expected": record.hash_sha256[:16]},
        )

    if not record.is_commercial_safe:
        result["warnings"].append(
            f"Dataset is NOT commercial-safe (source: {record.source}, "
            f"licence: {record.licence})"
        )

    logger.info(
        "provenance verified",
        extra={
            "path": str(data_path),
            "hash_match": result["hash_match"],
            "commercial_safe": record.is_commercial_safe,
        },
    )
    return result


@timed("data.provenance.list")
def list_datasets(
    root: str = "data/",
    source_filter: Optional[str] = None,
    commercial_only: bool = False,
) -> List[Dict[str, Any]]:
    """
    List all registered datasets under a directory.

    Args:
        root: Directory to scan for sidecar files.
        source_filter: If set, only return datasets from this source.
        commercial_only: If ``True``, only return commercial-safe datasets.

    Returns:
        List of provenance record dicts.

    Example::

        # List all commercial-safe Alibaba datasets
        for ds in list_datasets("data/", source_filter="alibaba", commercial_only=True):
            print(f"{ds['path']} — {ds['licence']}")
    """
    root_path = Path(root).resolve()
    if not root_path.exists():
        logger.warning("root directory does not exist", extra={"root": str(root_path)})
        return []

    datasets: List[Dict[str, Any]] = []
    for sidecar in sorted(root_path.rglob(f"*{_PROV_SUFFIX}")):
        try:
            raw = json.loads(sidecar.read_text(encoding="utf-8"))
            record = ProvenanceRecord(**raw)
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning(
                "skipping corrupt sidecar",
                extra={"path": str(sidecar), "error": str(exc)},
            )
            continue

        # Apply filters
        if source_filter and record.source != source_filter.lower():
            continue
        if commercial_only and not record.is_commercial_safe:
            continue

        datasets.append(asdict(record))

    logger.info(
        "listed datasets",
        extra={"root": str(root_path), "count": len(datasets)},
    )
    return datasets
