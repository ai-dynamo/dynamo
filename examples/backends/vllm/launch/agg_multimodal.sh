#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME=$2
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model <model_name> Specify the VLM model to use (default: $MODEL_NAME)"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Start frontend with Rust OpenAIPreprocessor
python -m dynamo.frontend --http-port=8000 &

# Configure GPU memory optimization for specific models
EXTRA_ARGS=""
if [[ "$MODEL_NAME" == "Qwen/Qwen2.5-VL-7B-Instruct" ]]; then
    EXTRA_ARGS="--gpu-memory-utilization 0.85 --max-model-len 2048"
fi

# Start vLLM worker with vision model
# Multimodal data (images) are decoded in the backend worker using ImageLoader
# --enforce-eager: Quick deployment (remove for production)
# --connector none: No KV transfer needed for aggregated serving
DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=8081 \
    python -m dynamo.vllm --model $MODEL_NAME --enforce-eager --connector none $EXTRA_ARGS

# Wait for all background processes to complete
wait


