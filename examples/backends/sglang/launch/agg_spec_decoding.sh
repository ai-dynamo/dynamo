#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"  # build_sglang_gpu_mem_args

# When the serve test sets requested_sglang_kv_tokens, this emits a
# --max-total-tokens cap so VRAM is bounded (GPU-size-independent). Empty for
# manual runs.
GPU_MEM_ARGS=$(build_sglang_gpu_mem_args)

MODEL="Qwen/Qwen3-8B"
DRAFT_MODEL="Tengyunw/qwen3_8b_eagle3"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Speculative Decoding (1 GPU)" "$MODEL" "$HTTP_PORT"

# ---------------------------
# 1. Frontend (Ingress)
# ---------------------------
python3 -m dynamo.frontend &


# ---------------------------
# 2. Speculative Main Worker
# ---------------------------
# This runs the main model with an EAGLE3 draft model for speculative decoding.
# --enable-metrics exposes sglang:spec_* metrics on the worker system port.
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
python3 -m dynamo.sglang \
    --model-path "$MODEL" \
    --served-model-name "$MODEL" \
    --page-size 16 \
    --tp 1 \
    --trust-remote-code \
    --enable-metrics \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path "$DRAFT_MODEL" \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    $GPU_MEM_ARGS "$@" &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
