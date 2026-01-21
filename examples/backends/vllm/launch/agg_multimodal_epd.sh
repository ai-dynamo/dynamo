#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# EPD (Encode-Prefill-Decode) multimodal deployment
#
# Architecture: 3-component disaggregation
# - Processor: Python-based preprocessor (bypasses Rust OpenAIPreprocessor)
# - Encode Worker: Dedicated vision encoder that extracts image embeddings
# - PD Worker: Standard prefill/decode worker that receives embeddings via NIXL
#
# Benefits: Decouples encoding from inference, enables independent scaling
# For standard single-worker deployment, see agg_multimodal.sh

set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Default values
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
SINGLE_GPU=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME=$2
            shift 2
            ;;
        --single-gpu)
            SINGLE_GPU=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model <model_name> Specify the model to use (default: $MODEL_NAME)"
            echo "  --single-gpu         Run both encode and PD workers on GPU 0 (for pre-merge CI)"
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

# Start frontend (HTTP endpoint)
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python -m dynamo.frontend &

# Set GPU memory utilization and model length based on deployment mode
# Single-GPU mode: Both workers share GPU 0, so use reduced memory settings
# Multi-GPU mode: Each worker gets its own GPU, so use higher memory settings
EXTRA_ARGS=""
if [[ "$SINGLE_GPU" == "true" ]]; then
    EXTRA_ARGS="--gpu-memory-utilization 0.5 --enforce-eager --max-model-len 30426"
else
    # Multi-GPU mode: standard memory settings
    EXTRA_ARGS="--gpu-memory-utilization 0.85 --max-model-len 34096"
fi

# Start processor (Python-based preprocessing, handles prompt templating)
python -m dynamo.vllm --multimodal-processor --enable-multimodal --model $MODEL_NAME &

# run E/P/D workers
# Use single GPU (GPU 0) for pre-merge CI, otherwise use GPU 0 for encode and GPU 1 for PD
if [[ "$SINGLE_GPU" == "true" ]]; then
    # Single GPU mode: both workers share GPU 0 with reduced memory
    CUDA_VISIBLE_DEVICES=0 python -m dynamo.vllm --multimodal-encode-worker --enable-multimodal --model $MODEL_NAME $EXTRA_ARGS &
    CUDA_VISIBLE_DEVICES=0 python -m dynamo.vllm --multimodal-worker --enable-multimodal --enable-mm-embeds --model $MODEL_NAME $EXTRA_ARGS &
else
    CUDA_VISIBLE_DEVICES=0 python -m dynamo.vllm --multimodal-encode-worker --enable-multimodal --model $MODEL_NAME $EXTRA_ARGS &
    CUDA_VISIBLE_DEVICES=1 python -m dynamo.vllm --multimodal-worker --enable-multimodal --enable-mm-embeds --model $MODEL_NAME $EXTRA_ARGS &
fi

# Wait for all background processes to complete
wait
