#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

# Setup cleanup trap
cleanup() {
    echo "Cleaning up background processes..."
    kill $FRONTEND_PID 2>/dev/null || true
    wait $FRONTEND_PID 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# Model configuration
MODEL_PATH="inclusionAI/LLaDA2.0-mini-preview"

# Diffusion algorithm configuration
DLLM_ALGORITHM="${DLLM_ALGORITHM:-LowConfidence}"
DLLM_ALGORITHM_CONFIG="${DLLM_ALGORITHM_CONFIG:-}"  # Optional: path to YAML config file

# Dynamo configuration
NAMESPACE="${NAMESPACE:-dynamo}"
COMPONENT="${COMPONENT:-backend}"
ENDPOINT="${ENDPOINT:-generate}"
HTTP_PORT="${HTTP_PORT:-8001}"
TP_SIZE="${TP_SIZE:-1}"

echo "=========================================="
echo "Launching Diffusion LM Worker (LLaDA2.0)"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Namespace: $NAMESPACE"
echo "Component: $COMPONENT"
echo "Frontend Port: $HTTP_PORT"
echo "TP Size: $TP_SIZE"
echo "Diffusion Algorithm: ${DLLM_ALGORITHM:-LowConfidence}"
echo "Algorithm Config: ${DLLM_ALGORITHM_CONFIG:-default}"
echo "=========================================="

# Launch frontend (OpenAI-compatible API server)
echo "Starting Dynamo Frontend on port $HTTP_PORT..."
python -m dynamo.frontend \
    --http-port "$HTTP_PORT" &

FRONTEND_PID=$!

# Wait for frontend to start
sleep 2

# Launch diffusion worker
echo "Starting Diffusion LM Worker..."

# Build the command with required arguments
export CUDA_VISIBLE_DEVICES=0
CMD="python -m dynamo.sglang \
    --model-path $MODEL_PATH \
    --tp-size $TP_SIZE \
    --skip-tokenizer-init \
    --trust-remote-code \
    --endpoint dyn://${NAMESPACE}.${COMPONENT}.${ENDPOINT} \
    --enable-metrics \
    --disable-cuda-graph \
    --disable-overlap-schedule \
    --attention-backend triton \
    --dllm-algorithm $DLLM_ALGORITHM"

# Add optional algorithm config if provided
if [ -n "$DLLM_ALGORITHM_CONFIG" ]; then
    CMD="$CMD --dllm-algorithm-config $DLLM_ALGORITHM_CONFIG"
fi

# Execute the command
eval $CMD