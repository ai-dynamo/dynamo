#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch script for testing vLLM-Omni integration with text-to-text generation
# This script starts an aggregated (frontend + omni worker) deployment for testing
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Default model - Qwen2.5-Omni-7B for text-to-text
MODEL="${MODEL:-Qwen/Qwen2.5-Omni-7B}"

# Stage config path - use single-stage LLM config for text-to-text
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
STAGE_CONFIG="${STAGE_CONFIG:-$SCRIPT_DIR/stage_configs/single_stage_llm.yaml}"

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

echo "=========================================="
echo "Starting vLLM-Omni Worker (Text-to-Text)"
echo "Model: $MODEL"
echo "Stage Config: $STAGE_CONFIG"
echo "=========================================="

# Run ingress (frontend)
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
echo "Starting frontend on port ${DYN_HTTP_PORT:-8000}..."
python -m dynamo.frontend &
FRONTEND_PID=$!

# Wait a bit for frontend to start
sleep 2

echo "Starting Omni worker..."
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    python -m dynamo.vllm \
    --model "$MODEL" \
    --omni \
    --stage-configs-path "$STAGE_CONFIG" \
    --connector none \
    "${EXTRA_ARGS[@]}"
