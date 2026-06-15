#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Elastic Expert Parallelism (ePLB) on the unified vLLM backend.
#
# Brings up a single head-node backend with the Ray DP backend so the live
# `scale_elastic_ep` control can spin DP-worker Ray actors up/down without a
# pod restart. Elastic EP requires `--data-parallel-backend ray` (vLLM asserts
# nnodes == 1 for the ray backend); for multi-node, join secondary nodes to the
# Ray cluster out-of-band with `ray start --address=<head>` before launching —
# the head node's engine schedules new DP ranks onto whichever cluster nodes
# have free GPUs. This is distinct from `--headless` (mp-backend multi-node).
#
# Scale at runtime (control surface is served on DYN_SYSTEM_PORT):
#   curl -X POST localhost:${DYN_SYSTEM_PORT:-8081}/engine/control/scale_elastic_ep \
#        -H 'Content-Type: application/json' \
#        -d '{"new_data_parallel_size": 4}'

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh" # print_launch_banner, wait_any_exit

# ---- Tunables (override via env vars) ----
# Small MoE that exercises expert parallelism; override for larger models.
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
DP_SIZE="${DP_SIZE:-2}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
SYSTEM_PORT="${DYN_SYSTEM_PORT:-8081}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --dp-size)
            DP_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model <name>     Model (default: $MODEL)"
            echo "  --dp-size <n>      Initial data-parallel size (default: $DP_SIZE)"
            echo "  -h, --help         Show this help message"
            echo ""
            echo "Any additional options are passed through to unified_main."
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

print_launch_banner "Launching Elastic EP / ePLB (unified, DP=$DP_SIZE)" "$MODEL" "$HTTP_PORT"

# Elastic EP requires vLLM's Ray DP backend (--data-parallel-backend ray), but
# the vLLM runtime image does not ship ray. Ensure it is importable.
python3 -c 'import ray' 2>/dev/null || { echo "ray not found; installing (required for elastic EP)..."; pip install -q ray; }

# run ingress
python -m dynamo.frontend --router-mode kv &

# Elastic-EP head node on the unified backend.
# --data-parallel-backend ray is required for scale_elastic_ep.
# --enforce-eager is for quick deployment; remove for production.
DYN_SYSTEM_PORT="$SYSTEM_PORT" \
python3 -m dynamo.vllm.unified_main \
    --model "$MODEL" \
    --data-parallel-backend ray \
    --data-parallel-size "$DP_SIZE" \
    --enable-expert-parallel \
    --enable-eplb \
    --enable-elastic-ep \
    --enforce-eager \
    "${EXTRA_ARGS[@]}" &

echo "All workers starting. (press Ctrl+C to stop)..."
# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
