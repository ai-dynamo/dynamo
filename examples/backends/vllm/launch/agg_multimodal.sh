#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated multimodal serving with standard Dynamo preprocessing
#
# Architecture: Single-worker PD (Prefill-Decode)
# - Frontend: Rust OpenAIPreprocessor handles image URLs (HTTP and data:// base64)
# - Worker: Standard vLLM worker with vision model support
#
# For EPD (Encode-Prefill-Decode) architecture with dedicated encoding worker,
# see agg_multimodal_epd.sh

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

# Default values
MODEL_NAME="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"

# Parse command line arguments
# Extra arguments are passed through to the vLLM worker
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME=$2
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] [-- EXTRA_VLLM_ARGS]"
            echo "Options:"
            echo "  --model <model_name>   Specify the VLM model to use (default: $MODEL_NAME)"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Any additional arguments are passed through to the vLLM worker."
            echo "Example: $0 --model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 --dyn-tool-call-parser hermes"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner --multimodal "Launching Aggregated Multimodal Serving" "$MODEL_NAME" "$HTTP_PORT"

# Use TCP transport (instead of default NATS)
# TCP is preferred for multimodal workloads because it overcomes:
# - NATS default 1MB max payload limit (multimodal base64 images can exceed this)
export DYN_REQUEST_PLANE=tcp

# Start frontend with Rust OpenAIPreprocessor
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python -m dynamo.frontend &

# ---- Per-model defaults ----
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"
MODEL_EXTRA_ARGS=""
case "$MODEL_NAME" in
    meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8)
        MAX_MODEL_LEN="${MAX_MODEL_LEN:-108960}"
        MODEL_EXTRA_ARGS="--tensor-parallel-size=8" ;;
esac

GPU_MEM_ARGS=$(build_gpu_mem_args vllm)

# Start vLLM worker with vision model
# --enforce-eager: Quick deployment (remove for production)
# Extra args from command line come last to allow overrides
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    python -m dynamo.vllm --enable-multimodal --model $MODEL_NAME \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_CONCURRENT_SEQS" \
    $GPU_MEM_ARGS \
    $MODEL_EXTRA_ARGS \
    "${EXTRA_ARGS[@]}"

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
