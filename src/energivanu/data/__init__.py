# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Energivanu Data Package
========================
Data loading, processing, and dataset construction for the PEB model.

Components:

- :mod:`h100_processor` — Real H100 Data Processor (York University format)
- :mod:`alibaba_processor` — Alibaba GPU Trace 2020 data processor
- :mod:`validator` — Data quality validation
- :mod:`provenance` — Data lineage tracking
- :mod:`cluster_merger` — Multi-node telemetry merger for cluster-scale training
"""

# Lazy imports to avoid requiring torch at package import time
def __getattr__(name):
    if name in ("RealH100Dataset", "build_dataloaders", "create_sequences",
                "load_node_data", "scale_to_facility"):
        from .h100_processor import (  # noqa: F401
            RealH100Dataset,  # noqa: F401
            build_dataloaders,  # noqa: F401
            create_sequences,  # noqa: F401
            load_node_data,  # noqa: F401
            scale_to_facility,  # noqa: F401
        )
        return locals()[name]
    elif name == "AlibabaTraceProcessor":
        from .alibaba_processor import AlibabaTraceProcessor
        return AlibabaTraceProcessor
    elif name == "ClusterMerger":
        from .cluster_merger import ClusterMerger
        return ClusterMerger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "RealH100Dataset",
    "build_dataloaders",
    "create_sequences",
    "load_node_data",
    "scale_to_facility",
    "AlibabaTraceProcessor",
    "ClusterMerger",
]
