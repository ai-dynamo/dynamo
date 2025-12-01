#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated multimodal serving with tool calling support
#
# Architecture: Single-worker PD (Prefill-Decode) with tool call parser
# - Frontend: Rust OpenAIPreprocessor handles image URLs (HTTP and data:// base64)
# - Worker: vLLM worker with vision model and tool calling support

set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Default values
MODEL_NAME="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
MAX_MODEL_LEN="10000"
TOOL_PARSER="hermes"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME=$2
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN=$2
            shift 2
            ;;
        --tool-parser)
            TOOL_PARSER=$2
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model <model_name>        Specify the VLM model to use (default: $MODEL_NAME)"
            echo "  --max-model-len <length>    Maximum model length (default: $MAX_MODEL_LEN)"
            echo "  --tool-parser <parser>      Tool call parser (default: $TOOL_PARSER)"
            echo "  -h, --help                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
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

# Start vLLM worker with vision model and tool calling
# --enable-multimodal: Enable multimodal support for vision models
# --dyn-tool-call-parser: Enable tool calling with specified parser
# --enforce-eager: Quick deployment (remove for production)
# --connector none: No KV transfer needed for aggregated serving
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    python3 -m dynamo.vllm \
    --model $MODEL_NAME \
    --max-model-len $MAX_MODEL_LEN \
    --dyn-tool-call-parser $TOOL_PARSER \
    --enable-multimodal \
    --enforce-eager \
    --connector none

# Wait for all background processes to complete
wait

