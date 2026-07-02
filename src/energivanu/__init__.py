# SPDX-License-Identifier: AGPL-3.0-or-later
"""Energivanu — ML-based power prediction for AI data centers."""

__version__ = "0.1.0"

# Lazy imports — avoid requiring torch at package import time
# This allows non-ML modules (bess, grid, mpc, optimizer, scheduler)
# to work without PyTorch installed.

def __getattr__(name):
    _lazy_map = {
        "EnergivanuPEB": ".model",
        "load_model": ".model",
        "RealH100Dataset": ".data",
        "build_dataloaders": ".data",
        "MPCController": ".mpc",
        "PeakShavingOptimizer": ".optimizer",
        "PhaseStaggeringScheduler": ".scheduler",
        "PyBaMMBattery": ".bess",
        "BatteryState": ".bess",
        "BESSModbusServer": ".bess",
        "OpenADRVEN": ".grid",
        "GridEvent": ".grid",
        "GridSignalLevel": ".grid",
        "ERCOTSCEDClient": ".grid",
        "SCEDSignal": ".grid",
    }
    _lazy_map.update({
        "setup_distributed": ".distributed",
        "cleanup_distributed": ".distributed",
        "is_main_process": ".distributed",
        "get_device": ".distributed",
        "save_checkpoint": ".distributed",
    })
    if name in _lazy_map:
        import importlib
        mod = importlib.import_module(_lazy_map[name], __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "EnergivanuPEB",
    "load_model",
    "RealH100Dataset",
    "build_dataloaders",
    "MPCController",
    "PeakShavingOptimizer",
    "PhaseStaggeringScheduler",
    "PyBaMMBattery",
    "BatteryState",
    "BESSModbusServer",
    "OpenADRVEN",
    "GridEvent",
    "GridSignalLevel",
    "ERCOTSCEDClient",
    "SCEDSignal",
]
