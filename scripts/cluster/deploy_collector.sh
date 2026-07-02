#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# =============================================================================
# Energivanu Cluster Collector Deployment
# =============================================================================
# Deploy telemetry collector to all nodes in a cluster via SSH.
#
# Prerequisites:
#   - energivanu installed on each node (`pip install energivanu`)
#   - passwordless SSH to all nodes
#   - a shared output directory (NFS, Lustre, etc.)
#
# Usage:
#   # Deploy 1-hour standard collection on 3 nodes
#   ./scripts/cluster/deploy_collector.sh \
#       --nodes node1.example.com,node2.example.com,node3.example.com \
#       --output /shared/data/collection_2026-07-02 \
#       --mode standard
#
#   # Quick 5-min test across all nodes in a hostfile
#   ./scripts/cluster/deploy_collector.sh \
#       --hostfile ~/hosts.txt \
#       --mode quick
#
#   # Custom 30-minute collection
#   ./scripts/cluster/deploy_collector.sh \
#       --nodes node1,node2,node3 \
#       --mode custom --duration 30 --interval 2 \
#       --output /nfs/energivanu/data/run3
#
# After collection completes on all nodes, merge the data:
#   python -m energivanu.data.cluster_merger \
#       /shared/data/run3/node_*.csv \
#       -o /shared/data/run3/merged.npz
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODE="standard"
OUTPUT_DIR="./cluster_collection_$(date +%Y%m%d_%H%M%S)"
DURATION=""
INTERVAL=""
NODES=""
HOSTFILE=""
COLLECT_SCRIPT="collect_data.py"
MAX_PARALLEL=16  # don't SSH into more than this at once

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
usage() {
    cat <<EOF
Usage: $0 [options]

Deploy Energivanu telemetry collector across cluster nodes via SSH.

Options:
  --nodes N1,N2,...    Comma-separated list of node hostnames
  --hostfile FILE      File with one node hostname per line
  --mode MODE          Collection mode: quick|standard|marathon|custom (default: standard)
  --duration MIN       Duration in minutes (required for custom mode)
  --interval SEC       Sampling interval in seconds (default: auto)
  --output DIR         Output directory on shared storage (default: ./cluster_...)
  --max-parallel N     Max concurrent SSH sessions (default: 16)
  --help               Show this message

Examples:
  $0 --nodes node1,node2,node3 --mode standard --output /nfs/telemetry
  $0 --hostfile ~/hosts.txt --mode quick
EOF
    exit 1
}

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nodes)       NODES="$2"; shift 2 ;;
        --hostfile)    HOSTFILE="$2"; shift 2 ;;
        --mode)        MODE="$2"; shift 2 ;;
        --duration)    DURATION="$2"; shift 2 ;;
        --interval)    INTERVAL="$2"; shift 2 ;;
        --output)      OUTPUT_DIR="$2"; shift 2 ;;
        --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
        --help)        usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ---------------------------------------------------------------------------
# Collect node list
# ---------------------------------------------------------------------------
NODE_ARRAY=()

if [[ -n "$NODES" ]]; then
    IFS=',' read -ra NODE_ARRAY <<< "$NODES"
elif [[ -n "$HOSTFILE" ]]; then
    if [[ ! -f "$HOSTFILE" ]]; then
        echo "ERROR: Hostfile not found: $HOSTFILE"
        exit 1
    fi
    mapfile -t NODE_ARRAY < "$HOSTFILE"
else
    echo "ERROR: Either --nodes or --hostfile is required."
    exit 1
fi

NUM_NODES=${#NODE_ARRAY[@]}
echo "Nodes: ${NUM_NODES} (${NODE_ARRAY[*]})"
echo "Mode:  ${MODE}"
echo "Output: ${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Build remote command
# ---------------------------------------------------------------------------
REMOTE_ARGS="--mode ${MODE} --output \"${OUTPUT_DIR}/\$(hostname -s).csv\""

if [[ -n "$DURATION" ]]; then
    REMOTE_ARGS="${REMOTE_ARGS} --duration ${DURATION}"
fi
if [[ -n "$INTERVAL" ]]; then
    REMOTE_ARGS="${REMOTE_ARGS} --interval ${INTERVAL}"
fi

REMOTE_CMD="mkdir -p \"${OUTPUT_DIR}\" && python \"${COLLECT_SCRIPT}\" ${REMOTE_ARGS}"
echo "Remote command: ${REMOTE_CMD}"

# ---------------------------------------------------------------------------
# Deploy in parallel
# ---------------------------------------------------------------------------
echo ""
echo "Deploying collector to ${NUM_NODES} nodes..."
echo "========================================"

pids=()
failures=0

for node in "${NODE_ARRAY[@]}"; do
    # Limit parallelism
    while [[ ${#pids[@]} -ge $MAX_PARALLEL ]]; do
        for i in "${!pids[@]}"; do
            if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                wait "${pids[$i]}" || ((failures++))
                unset "pids[$i]"
            fi
        done
        sleep 0.5
        pids=("${pids[@]}")  # re-index
    done

    echo "[$(date '+%H:%M:%S')] Deploying to ${node}..."
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
        "${node}" "${REMOTE_CMD}" &
    pids+=($!)
done

# Wait for remaining
for pid in "${pids[@]}"; do
    wait "$pid" || ((failures++))
done

echo ""
echo "========================================"
echo "Collection complete: ${NUM_NODES} nodes, ${failures} failures"
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "To merge telemetry into training data:"
echo "  python -m energivanu.data.cluster_merger \\"
echo "      ${OUTPUT_DIR}/node_*.csv \\"
echo "      -o ${OUTPUT_DIR}/merged.npz"
echo ""
echo "To fine-tune the model:"
echo "  torchrun --nproc_per_node=N -m energivanu.train_real_cluster \\"
echo "      --data ${OUTPUT_DIR}/merged.npz \\"
echo "      --checkpoint models/checkpoints/commercial_best.pt"
echo ""
