#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
NC-Licence Compliance Scanner
==============================
Scan the repository for non-commercial (NC) licensed data contamination.

Checks performed:

1. **File headers** — Look for NC licence text (CC BY-NC, CC BY-NC-ND, etc.)
2. **Data file references** — Detect York University / MIT Supercloud dataset refs
3. **Import statements** — Flag packages that are NC-only

Exit codes:
- ``0`` — No violations found.
- ``1`` — Violations detected (details printed to stdout).
- ``2`` — Scanner error (missing dependency, etc.)

Usage::

    # Scan entire repo
    python scripts/check_compliance.py

    # Scan specific directory
    python scripts/check_compliance.py --root src/

    # Scan only staged files (for pre-commit hook)
    python scripts/check_compliance.py --staged

    # JSON output
    python scripts/check_compliance.py --format json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Patterns that indicate NC-licensed content
_NC_LICENSE_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"CC\s+BY-NC", re.IGNORECASE),
    re.compile(r"Creative\s+Commons.*NonCommercial", re.IGNORECASE),
    re.compile(r"CC-BY-NC", re.IGNORECASE),
    re.compile(r"Attribution-NonCommercial", re.IGNORECASE),
    re.compile(r"NonCommercial\s+4\.0", re.IGNORECASE),
    re.compile(r"by-nc/4\.0", re.IGNORECASE),
]

# Known NC-only dataset identifiers
_NC_DATASET_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"York\s+University.*H100", re.IGNORECASE),
    re.compile(r"MIT\s+Supercloud", re.IGNORECASE),
    re.compile(r"MIT\s+Datacenter\s+Challenge", re.IGNORECASE),
    re.compile(r"FigShare.*York", re.IGNORECASE),
    re.compile(r"CC.BY.NC.ND", re.IGNORECASE),
    re.compile(r"High-resolution.AI.Data.Center.*FigShare", re.IGNORECASE),
]

# File extensions to scan for licence text
_TEXT_EXTENSIONS: Set[str] = {
    ".py", ".md", ".txt", ".rst", ".yaml", ".yml", ".toml",
    ".cfg", ".ini", ".json", ".csv", ".tsv", ".sh", ".bash",
    ".html", ".css", ".js", ".ts",
}

# Data file extensions to scan for dataset references
_DATA_EXTENSIONS: Set[str] = {
    ".csv", ".tsv", ".parquet", ".json", ".yaml", ".yml",
    ".h5", ".hdf5", ".pkl", ".pickle", ".npy", ".npz",
}

# Directories to always skip
_SKIP_DIRS: Set[str] = {
    ".git", "__pycache__", ".mypy_cache", ".pytest_cache",
    "node_modules", ".venv", "venv", "env", ".eggs",
    "*.egg-info", "build", "dist",
}

# Files/directories that are documentation-only and may legitimately
# reference NC datasets for informational/warning purposes.
# These are NOT data contamination — they describe what NOT to use.
_EXCLUDE_PATHS: Set[str] = {
    "scripts/check_compliance.py",  # The scanner itself
    "README.md",
    "MASTER_STRATEGY.md",
    "MODEL_CARD.md",
    "PROJECT_STATUS.md",
    "TECHNICAL_DOCUMENTATION.md",
    "VERIFICATION_REPORT.md",
    "DEEP_DIVE_ANALYSIS.md",
    "ZERO_BUDGET_MASTER_PLAN.md",
    "docs/LEGAL_FAQ.md",
    "data/README.md",
    "option-2-open-license/",
    "option-3-dual-strategy/",
    ".github/workflows/compliance.yml",
    "magazine/LEGAL_COMPLIANCE.md",
    "magazine/build_magazine.py",
    "magazine/build_docx.py",
    "models/results.json",
    "config/data_sources.yaml",
    "index.html",
    "docs/index.html",
    "kaggle/03_full_pipeline.py",
    "src/energivanu/data/provenance.py",
    "src/energivanu/data.py",
    "src/energivanu/data/h100_processor.py",
    "src/energivanu/data/alibaba_processor.py",
}

# NC-only Python packages (hypothetical — extend as needed)
_NC_PACKAGES: Set[str] = {
    # Add any known NC-only packages here
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Violation:
    """A single compliance violation."""
    file: str
    line: int
    category: str  # "nc_license", "nc_dataset_ref", "nc_package"
    message: str
    matched_text: str


@dataclass
class ComplianceReport:
    """Aggregated compliance scan results."""
    violations: List[Violation] = field(default_factory=list)
    files_scanned: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return len(self.violations) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_clean": self.is_clean,
            "files_scanned": self.files_scanned,
            "violation_count": len(self.violations),
            "violations": [
                {
                    "file": v.file,
                    "line": v.line,
                    "category": v.category,
                    "message": v.message,
                    "matched_text": v.matched_text,
                }
                for v in self.violations
            ],
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

class ComplianceScanner:
    """
    Scan a directory tree for NC-licence contamination.

    Args:
        root: Root directory to scan.
        skip_dirs: Additional directory names to skip.
        extra_nc_patterns: Additional regex patterns for NC licence detection.

    Example::

        scanner = ComplianceScanner(root=Path("."))
        report = scanner.scan()
        if not report.is_clean:
            for v in report.violations:
                print(f"{v.file}:{v.line} — {v.message}")
    """

    def __init__(
        self,
        root: Path,
        skip_dirs: Optional[Set[str]] = None,
        extra_nc_patterns: Optional[List[str]] = None,
    ) -> None:
        self._root = root.resolve()
        self._skip = _SKIP_DIRS | (skip_dirs or set())
        self._nc_patterns = list(_NC_LICENSE_PATTERNS)
        if extra_nc_patterns:
            for pat in extra_nc_patterns:
                self._nc_patterns.append(re.compile(pat, re.IGNORECASE))

        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s: %(message)s",
        )
        self._logger = logging.getLogger("compliance")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self, file_list: Optional[Sequence[str]] = None) -> ComplianceReport:
        """
        Run a full compliance scan.

        Args:
            file_list: If provided, scan only these files (relative to root).
                Otherwise, walk the entire tree.

        Returns:
            A :class:`ComplianceReport`.
        """
        report = ComplianceReport()

        if file_list is not None:
            paths = [self._root / f for f in file_list]
        else:
            paths = list(self._iter_files())

        self._logger.info("scanning %d files under %s", len(paths), self._root)

        for path in paths:
            if not path.is_file():
                continue
            report.files_scanned += 1
            try:
                self._scan_file(path, report)
            except Exception as exc:
                report.errors.append(f"Error scanning {path}: {exc}")

        self._logger.info(
            "scan complete: %d files, %d violations",
            report.files_scanned,
            len(report.violations),
        )
        return report

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _iter_files(self):
        """Yield all scannable files under root, skipping excluded dirs."""
        for path in sorted(self._root.rglob("*")):
            # Skip excluded directories
            if any(part in self._skip for part in path.parts):
                continue
            # Skip excluded paths (documentation files that legitimately
            # reference NC datasets for informational purposes)
            rel = str(path.relative_to(self._root))
            if any(rel.startswith(excl) or rel == excl for excl in _EXCLUDE_PATHS):
                continue
            if path.is_file():
                yield path

    def _scan_file(self, path: Path, report: ComplianceReport) -> None:
        """Run all checks on a single file."""
        suffix = path.suffix.lower()

        # Text files — check headers for NC licence text
        if suffix in _TEXT_EXTENSIONS:
            self._check_text_file(path, report)

        # Data files — check for NC dataset references
        if suffix in _DATA_EXTENSIONS:
            self._check_data_file(path, report)

        # Python files — check imports
        if suffix == ".py":
            self._check_imports(path, report)

    def _check_text_file(self, path: Path, report: ComplianceReport) -> None:
        """Check a text file for NC licence patterns."""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return

        lines = text.splitlines()
        for lineno, line in enumerate(lines, start=1):
            for pattern in self._nc_patterns:
                match = pattern.search(line)
                if match:
                    report.violations.append(Violation(
                        file=str(path.relative_to(self._root)),
                        line=lineno,
                        category="nc_license",
                        message=f"NC licence pattern found: '{match.group()}'",
                        matched_text=match.group(),
                    ))

        # Also check for NC dataset references in text files
        for lineno, line in enumerate(lines, start=1):
            for pattern in _NC_DATASET_PATTERNS:
                match = pattern.search(line)
                if match:
                    report.violations.append(Violation(
                        file=str(path.relative_to(self._root)),
                        line=lineno,
                        category="nc_dataset_ref",
                        message=f"NC dataset reference found: '{match.group()}'",
                        matched_text=match.group(),
                    ))

    def _check_data_file(self, path: Path, report: ComplianceReport) -> None:
        """Check a data file for NC dataset identifiers."""
        try:
            # Read first 1000 lines max for large data files
            text = path.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()[:1000]
        except OSError:
            return

        for lineno, line in enumerate(lines, start=1):
            for pattern in _NC_DATASET_PATTERNS:
                match = pattern.search(line)
                if match:
                    report.violations.append(Violation(
                        file=str(path.relative_to(self._root)),
                        line=lineno,
                        category="nc_dataset_ref",
                        message=f"NC dataset reference in data file: '{match.group()}'",
                        matched_text=match.group(),
                    ))

    def _check_imports(self, path: Path, report: ComplianceReport) -> None:
        """Check Python imports for NC-only packages."""
        if not _NC_PACKAGES:
            return

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return

        import_re = re.compile(
            r"^(?:from|import)\s+([\w.]+)", re.MULTILINE
        )

        for match in import_re.finditer(text):
            pkg = match.group(1).split(".")[0]
            if pkg in _NC_PACKAGES:
                lineno = text[:match.start()].count("\n") + 1
                report.violations.append(Violation(
                    file=str(path.relative_to(self._root)),
                    line=lineno,
                    category="nc_package",
                    message=f"Import of NC-only package: '{pkg}'",
                    matched_text=match.group(0),
                ))


# ---------------------------------------------------------------------------
# Staged-files helper (for pre-commit hooks)
# ---------------------------------------------------------------------------

def get_staged_files(root: Path) -> List[str]:
    """
    Return a list of files currently staged in git.

    Args:
        root: Repository root (must be a git repo).

    Returns:
        List of staged file paths relative to *root*.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        return [line for line in result.stdout.splitlines() if line.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logging.getLogger("compliance").warning(
            "Could not get staged files: %s", exc
        )
        return []


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(report: ComplianceReport, fmt: str = "text") -> None:
    """Print the compliance report to stdout."""
    if fmt == "json":
        print(json.dumps(report.to_dict(), indent=2))
        return

    # Text format
    print("=" * 60)
    print("  ENERGIVANU COMPLIANCE REPORT")
    print("=" * 60)
    print(f"  Files scanned:  {report.files_scanned}")
    print(f"  Violations:     {len(report.violations)}")
    print(f"  Errors:         {len(report.errors)}")
    print("=" * 60)

    if report.is_clean:
        print("\n  ✅ ALL CLEAR — No NC-licensed data contamination found.\n")
    else:
        print(f"\n  ❌ VIOLATIONS FOUND: {len(report.violations)}\n")
        for v in report.violations:
            print(f"  [{v.category}] {v.file}:{v.line}")
            print(f"    {v.message}")
            print(f"    matched: '{v.matched_text}'")
            print()

    if report.errors:
        print("  ⚠️  Scanner errors:")
        for err in report.errors:
            print(f"    - {err}")
        print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    CLI entry point.

    Returns:
        Exit code (0 = clean, 1 = violations, 2 = error).
    """
    parser = argparse.ArgumentParser(
        description="Scan repository for NC-licensed data contamination."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Root directory to scan (default: current directory).",
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        help="Scan only git staged files.",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format.",
    )
    parser.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="Exit with code 1 even on warnings (not just violations).",
    )

    args = parser.parse_args(argv)
    root: Path = args.root.resolve()

    if not root.is_dir():
        print(f"Error: root directory does not exist: {root}", file=sys.stderr)
        return 2

    scanner = ComplianceScanner(root=root)

    file_list: Optional[List[str]] = None
    if args.staged:
        file_list = get_staged_files(root)
        if not file_list:
            print("No staged files found.", file=sys.stderr)
            return 0

    try:
        report = scanner.scan(file_list=file_list)
    except Exception as exc:
        print(f"Scanner error: {exc}", file=sys.stderr)
        return 2

    print_report(report, fmt=args.format)

    if not report.is_clean:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
