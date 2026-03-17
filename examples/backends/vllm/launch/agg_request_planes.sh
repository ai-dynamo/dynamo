#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

# Parse command-line arguments for request plane mode
REQUEST_PLANE="tcp"  # Default to TCP

while [[ $# -gt 0 ]]; do
    case $1 in
        --tcp)
            REQUEST_PLANE="tcp"
            shift
            ;;
        --http)
            REQUEST_PLANE="http"
            shift
            ;;
        --nats)
            REQUEST_PLANE="nats"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--tcp|--http|--nats]"
            echo "  --tcp   Use TCP request plane (default)"
            echo "  --http  Use HTTP/2 request plane"
            echo "  --nats  Use NATS request plane"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

MODEL="Qwen/Qwen3-0.6B"

# ---- Tunable (override via env vars) ----
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"

# Set the request plane mode
export DYN_REQUEST_PLANE=$REQUEST_PLANE
echo "Using request plane mode: $REQUEST_PLANE"

GPU_MEM_FRACTION=$(build_gpu_mem_args vllm --model "$MODEL" --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_CONCURRENT_SEQS")

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Aggregated Serving + Request Planes (1 GPU)" "$MODEL" "$HTTP_PORT"

python -m dynamo.frontend &

DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
DYN_HEALTH_CHECK_ENABLED=true \
    python -m dynamo.vllm --model "$MODEL" --enforce-eager \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_CONCURRENT_SEQS" \
    ${GPU_MEM_FRACTION:+--gpu-memory-utilization "$GPU_MEM_FRACTION"} &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
