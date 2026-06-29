# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Energivanu Logging Configuration
=================================
Structured logging with JSON format, log rotation, and performance decorators.

Usage::

    from energivanu.logging_config import setup_logging, get_logger

    setup_logging()                          # configure once at startup
    logger = get_logger("model")
    logger.info("model loaded", extra={"parameters": 1_200_000})

    @timed("model.inference")
    def predict(x):
        ...

Modules should call :func:`get_logger` with a component name to get a
pre-configured child logger.  All loggers respect the global logging config.
"""

from __future__ import annotations

import functools
import json
import logging
import logging.handlers
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Optional, TypeVar

# ---------------------------------------------------------------------------
# JSON Formatter
# ---------------------------------------------------------------------------

class JSONFormatter(logging.Formatter):
    """
    Emit each log record as a single JSON line.

    Fields: ``timestamp``, ``level``, ``logger``, ``message``, plus any
    ``extra`` fields attached to the record.
    """

    _RESERVED = {
        "name", "msg", "args", "created", "relativeCreated",
        "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "pathname", "filename", "module", "levelno", "levelname",
        "msecs", "thread", "threadName", "processName", "process",
        "message", "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Attach exception info
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Attach extra fields
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                log_entry[key] = value

        return json.dumps(log_entry, default=str, ensure_ascii=False)


class ReadableFormatter(logging.Formatter):
    """Human-friendly formatter for console output."""

    COLORS = {
        "DEBUG": "\033[36m",     # cyan
        "INFO": "\033[32m",      # green
        "WARNING": "\033[33m",   # yellow
        "ERROR": "\033[31m",     # red
        "CRITICAL": "\033[35m",  # magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_color: bool = True):
        super().__init__()
        self._use_color = use_color and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        level = record.levelname
        if self._use_color:
            color = self.COLORS.get(level, "")
            level = f"{color}{level}{self.RESET}"

        msg = record.getMessage()
        parts = [f"{ts} | {level:8s} | {record.name} | {msg}"]

        if record.exc_info and record.exc_info[0] is not None:
            parts.append(self.formatException(record.exc_info))

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

_INITIALIZED = False
_LOG_DIR: Optional[Path] = None


def setup_logging(
    level: str = "INFO",
    fmt: str = "json",
    log_dir: str = "logs",
    max_bytes: int = 10_485_760,
    backup_count: int = 5,
    console_output: bool = True,
    config: Optional[Any] = None,
) -> None:
    """
    Configure the root ``energivanu`` logger.

    Args:
        level: Minimum log level (DEBUG/INFO/WARNING/ERROR/CRITICAL).
        fmt: ``"json"`` for structured JSON, ``"readable"`` for human-friendly.
        log_dir: Directory for log files.
        max_bytes: Maximum size per log file before rotation.
        backup_count: Number of rotated backups to keep.
        console_output: Whether to also log to stderr.
        config: Optional :class:`energivanu.config.LoggingConfig` instance.
            When provided, overrides all other parameters.
    """
    global _INITIALIZED, _LOG_DIR

    # Allow a LoggingConfig dataclass to drive everything
    if config is not None:
        level = getattr(config, "level", level)
        fmt = getattr(config, "format", fmt)
        log_dir = getattr(config, "log_dir", log_dir)
        max_bytes = getattr(config, "max_bytes", max_bytes)
        backup_count = getattr(config, "backup_count", backup_count)
        console_output = getattr(config, "console_output", console_output)

    _LOG_DIR = Path(log_dir)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger("energivanu")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    root_logger.handlers.clear()

    # File handler — always JSON, with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(_LOG_DIR / "energivanu.log"),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%S%z"))
    root_logger.addHandler(file_handler)

    # Per-component file handlers
    _COMPONENT_FILES = {
        "model": "model.log",
        "mpc": "mpc.log",
        "telemetry": "telemetry.log",
        "api": "api.log",
        "optimizer": "optimizer.log",
        "scheduler": "scheduler.log",
    }
    for component, filename in _COMPONENT_FILES.items():
        comp_handler = logging.handlers.RotatingFileHandler(
            filename=str(_LOG_DIR / filename),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        comp_handler.setLevel(logging.DEBUG)
        comp_handler.setFormatter(JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%S%z"))
        comp_logger = logging.getLogger(f"energivanu.{component}")
        comp_logger.addHandler(comp_handler)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        if fmt == "json":
            console_handler.setFormatter(JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%S%z"))
        else:
            console_handler.setFormatter(ReadableFormatter())
        root_logger.addHandler(console_handler)

    _INITIALIZED = True
    root_logger.info(
        "logging initialized",
        extra={"level": level, "format": fmt, "log_dir": str(_LOG_DIR)},
    )


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the ``energivanu`` namespace.

    If :func:`setup_logging` has not been called yet, a basic fallback
    configuration is applied so logging never silently drops messages.

    Args:
        name: Component name (e.g. ``"model"``, ``"mpc"``, ``"telemetry"``).

    Returns:
        A :class:`logging.Logger` instance.
    """
    if not _INITIALIZED:
        setup_logging()

    full_name = name if name.startswith("energivanu.") else f"energivanu.{name}"
    return logging.getLogger(full_name)


# ---------------------------------------------------------------------------
# Performance logging
# ---------------------------------------------------------------------------

F = TypeVar("F", bound=Callable[..., Any])


def timed(
    operation: str,
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
) -> Callable[[F], F]:
    """
    Decorator that logs the execution time of a function.

    Args:
        operation: Human-readable operation name (e.g. ``"model.inference"``).
        logger: Logger to use.  If ``None``, uses ``energivanu.perf``.
        level: Log level for the timing message.

    Example::

        @timed("model.forward")
        def forward(x):
            return model(x)
    """
    if logger is None:
        logger = get_logger("perf")

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.log(
                    level,
                    "operation completed",
                    extra={
                        "operation": operation,
                        "elapsed_ms": round(elapsed_ms, 3),
                        "status": "success",
                    },
                )
                return result
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.log(
                    level,
                    "operation failed",
                    extra={
                        "operation": operation,
                        "elapsed_ms": round(elapsed_ms, 3),
                        "status": "error",
                        "error": str(exc),
                    },
                )
                raise

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.log(
                    level,
                    "operation completed",
                    extra={
                        "operation": operation,
                        "elapsed_ms": round(elapsed_ms, 3),
                        "status": "success",
                    },
                )
                return result
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.log(
                    level,
                    "operation failed",
                    extra={
                        "operation": operation,
                        "elapsed_ms": round(elapsed_ms, 3),
                        "status": "error",
                        "error": str(exc),
                    },
                )
                raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return wrapper  # type: ignore[return-value]

    return decorator


@contextmanager
def log_timing(
    operation: str,
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that records elapsed time.

    Args:
        operation: Human-readable operation name.
        logger: Logger to use.  If ``None``, uses ``energivanu.perf``.
        level: Log level for the timing message.

    Example::

        with log_timing("data.load") as ctx:
            df = load_data()
        print(ctx["elapsed_ms"])
    """
    if logger is None:
        logger = get_logger("perf")

    ctx: Dict[str, Any] = {"operation": operation, "elapsed_ms": 0.0}
    start = time.perf_counter()
    try:
        yield ctx
    except Exception as exc:
        ctx["status"] = "error"
        ctx["error"] = str(exc)
        raise
    finally:
        ctx["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 3)
        ctx.setdefault("status", "success")
        logger.log(level, "timing", extra=ctx)


# ---------------------------------------------------------------------------
# Helpers for migrating from print()
# ---------------------------------------------------------------------------

def replace_print_with_logging(module_name: str = "energivanu") -> None:
    """
    Monkey-patch ``builtins.print`` so that calls from *module_name* are
    redirected to the logger at WARNING level.

    **Use sparingly** — this is a migration helper, not a production pattern.
    """
    import builtins

    _original_print = builtins.print
    _logger = get_logger(module_name)

    def _patched_print(*args: Any, **kwargs: Any) -> None:
        message = " ".join(str(a) for a in args)
        # Only redirect calls that look like Energivanu logging
        caller_frame = sys._getframe(1)
        caller_module = caller_frame.f_globals.get("__name__", "")
        if module_name in caller_module:
            _logger.warning(message, extra={"source": "print_migration"})
        else:
            _original_print(*args, **kwargs)

    builtins.print = _patched_print  # type: ignore[assignment]
