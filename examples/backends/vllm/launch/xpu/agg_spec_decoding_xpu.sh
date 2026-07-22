#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../../common/launch_utils.sh"

export VLLM_TARGET_DEVICE=xpu

MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
BLOCK_SIZE=64
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Speculative Decoding (1 GPU)" "$MODEL" "$HTTP_PORT"

# ---------------------------
# 1. Frontend (Ingress)
# ---------------------------
python -m dynamo.frontend --http-port="$HTTP_PORT" &


# ---------------------------
# 2. Speculative Main Worker
# ---------------------------
# This runs the main model with EAGLE as the draft model for speculative decoding
# TODO: use build_vllm_gpu_mem_args to measure VRAM instead of hardcoded fractions
DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=8081 \
ZE_AFFINITY_MASK=0 python -m dynamo.vllm \
    --model "$MODEL" \
    --enforce-eager \
    --max-model-len "$MAX_MODEL_LEN" \
    # --block-size $BLOCK_SIZE \
    --speculative_config '{
        "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
        "draft_tensor_parallel_size": 1,
        "num_speculative_tokens": 2,
        "method": "eagle3"
    }' \
    --gpu-memory-utilization 0.8 &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
