#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated serving on a single GPU.

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../../common/gpu_utils.sh"   # build_vllm_gpu_mem_args
source "$SCRIPT_DIR/../../../../common/launch_utils.sh" # print_launch_banner, wait_any_exit

export VLLM_TARGET_DEVICE=xpu

# Default model
MODEL="Qwen/Qwen3-0.6B"

# Parse command line arguments
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# ---- Tunable (override via env vars) ----
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"

GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Aggregated Serving (1 GPU)" "$MODEL" "$HTTP_PORT"

# run ingress
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python -m dynamo.frontend &

# run worker
# --enforce-eager is added for quick deployment. for production use, need to remove this flag
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    python -m dynamo.vllm --model "$MODEL" --enforce-eager \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_CONCURRENT_SEQS" \
    --block-size "${BLOCK_SIZE:-64}" \
    $GPU_MEM_ARGS "${EXTRA_ARGS[@]}" &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
