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
MODEL_NAME="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
ENCODER_COUNT=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME=$2
            shift 2
            ;;
        --encoder-count)
            ENCODER_COUNT=$2
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model <model_name> Specify the model to use (default: $MODEL_NAME)"
            echo "  --encoder-count <n>  Specify the number of encode workers to run (default: $ENCODER_COUNT)"
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

# Set max model length based on model name
MAX_MODEL_LEN="30426"
EXTRA_ARGS="--gpu-memory-utilization 0.8 --max-model-len $MAX_MODEL_LEN --no-enable-prefix-caching"

# Start processor (Python-based preprocessing, handles prompt templating)
python -m dynamo.vllm --multimodal-processor --enable-multimodal --model $MODEL_NAME &

# run E/P/D workers
CUDA_VISIBLE_DEVICES=0 python -m dynamo.vllm --multimodal-worker --enable-multimodal --enable-mm-embeds --model $MODEL_NAME $EXTRA_ARGS &
i=1
while [ $i -le $ENCODER_COUNT ]; do
    CUDA_VISIBLE_DEVICES=$((i)) python -m dynamo.vllm --multimodal-encode-worker --enable-multimodal --model $MODEL_NAME $EXTRA_ARGS &
    i=$((i+1))
done
# Wait for all background processes to complete
wait
