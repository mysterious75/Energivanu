# SPDX-License-Identifier: AGPL-3.0-or-later
# =============================================================================
# Energivanu Telemetry Collector — Docker Image
# =============================================================================
# Use this image to deploy the collector on containerized GPU clusters
# (Kubernetes, Slurm with enroot/Charliecloud, etc.).
#
# Build:
#   docker build -t energivanu-collector:latest -f scripts/cluster/Dockerfile .
#
# Run (standalone):
#   docker run --gpus all -v /mnt/shared:/data energivanu-collector:latest \
#       standard --output /data/node_$(hostname).csv
#
# Run (Kubernetes DaemonSet):
#   kubectl apply -f scripts/cluster/k8s-daemonset.yaml
#
# The collector writes per-GPU telemetry CSV to the mounted volume.
# After collection, merge all node CSVs:
#   python -m energivanu.data.cluster_merger /mnt/shared/node_*.csv -o merged.npz
# =============================================================================

FROM python:3.11-slim

LABEL maintainer="Energivanu Team <support@voraprotocol.com>"
LABEL description="GPU telemetry collector for Energivanu PEB model"
LABEL version="1.0"

# Install NVIDIA tools (nvidia-smi is needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    nvidia-utils-535 \
    && rm -rf /var/lib/apt/lists/*

# Install energivanu
COPY . /tmp/energivanu
RUN pip install --no-cache-dir /tmp/energivanu && \
    rm -rf /tmp/energivanu

# Default entry point
ENTRYPOINT ["python", "/usr/local/bin/collect_data.py"]

# Default: standard 1-hour collection, output to /data/node_<hostname>.csv
CMD ["standard", "--output", "/data/node_$(hostname -s).csv"]
