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

# Default values
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

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

# Use TCP transport (instead of default NATS)
# TCP is preferred for multimodal workloads because it overcomes:
# - NATS default 1MB max payload limit (multimodal base64 images can exceed this)
export DYN_REQUEST_PLANE=tcp

# Start frontend with Rust OpenAIPreprocessor
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python -m dynamo.frontend &

# Configure GPU memory optimization for specific models (if no extra args override)
MODEL_SPECIFIC_ARGS=""
if [[ "$MODEL_NAME" == "Qwen/Qwen2.5-VL-7B-Instruct" ]]; then
    MODEL_SPECIFIC_ARGS="--gpu-memory-utilization 0.85 --max-model-len 4096"
elif [[ "$MODEL_NAME" == "llava-hf/llava-1.5-7b-hf" ]]; then
    MODEL_SPECIFIC_ARGS="--gpu-memory-utilization 0.85 --max-model-len 4096"
fi

# Start vLLM worker with vision model
# Multimodal data (images) are decoded in the backend worker using ImageLoader
# --enforce-eager: Quick deployment (remove for production)
# --connector none: No KV transfer needed for aggregated serving
# Extra args from command line come last to allow overrides
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    python -m dynamo.vllm --enable-multimodal --model $MODEL_NAME --connector none $MODEL_SPECIFIC_ARGS "${EXTRA_ARGS[@]}"

# Wait for all background processes to complete
wait


