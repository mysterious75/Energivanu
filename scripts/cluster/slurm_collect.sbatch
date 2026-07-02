#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# =============================================================================
# Energivanu Telemetry Collection — SLURM Job
# =============================================================================
# Collect GPU telemetry across all nodes in a SLURM allocation.
#
# Submit:
#   sbatch --nodes=32 --ntasks-per-node=1 scripts/cluster/slurm_collect.sbatch
#
# This launches one collector per node, writing to a shared directory.
# After the job completes, merge the data:
#   python -m energivanu.data.cluster_merger \
#       /shared/energivanu/telemetry_<JOBID>/node_*.csv \
#       -o /shared/energivanu/telemetry_<JOBID>/merged.npz
#
# Then fine-tune:
#   torchrun --nproc_per_node=8 -m energivanu.train_real_cluster \
#       --data /shared/energivanu/telemetry_<JOBID>/merged.npz \
#       --checkpoint models/checkpoints/commercial_best.pt
# =============================================================================

#SBATCH --job-name=energivanu-collect
#SBATCH --output=/shared/energivanu/logs/collect_%j.out
#SBATCH --error=/shared/energivanu/logs/collect_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --exclusive

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COLLECT_MODE="standard"       # quick | standard | marathon | custom
COLLECT_DURATION=""           # minutes (only for custom mode)
COLLECT_INTERVAL=""           # seconds (optional)
BASE_DIR="/shared/energivanu/telemetry_${SLURM_JOB_ID}"

# ---------------------------------------------------------------------------
# Node-level collection
# ---------------------------------------------------------------------------
NODE_NAME=$(hostname -s)
OUTPUT_CSV="${BASE_DIR}/node_${NODE_NAME}.csv"

mkdir -p "${BASE_DIR}"

ARGS="--mode ${COLLECT_MODE} --output ${OUTPUT_CSV}"
[[ -n "$COLLECT_DURATION" ]] && ARGS="${ARGS} --duration ${COLLECT_DURATION}"
[[ -n "$COLLECT_INTERVAL" ]] && ARGS="${ARGS} --interval ${COLLECT_INTERVAL}"

echo "[$(date '+%H:%M:%S')] Node ${NODE_NAME}: starting collection"
echo "  Output: ${OUTPUT_CSV}"
echo "  Args:   ${ARGS}"

python scripts/collect_data.py ${ARGS}

echo "[$(date '+%H:%M:%S')] Node ${NODE_NAME}: collection complete"
wc -l "${OUTPUT_CSV}"
