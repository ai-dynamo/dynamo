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
MODEL_PATH="${MODEL_PATH:-inclusionAI/LLaDA2.0-mini}"
TP_SIZE="${TP_SIZE:-1}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}"  # Diffusion LMs typically run with low concurrency

# Dynamo configuration
NAMESPACE="${NAMESPACE:-dynamo}"
COMPONENT="${COMPONENT:-backend}"
ENDPOINT="${ENDPOINT:-generate}"
STORE_KV="${STORE_KV:-file}"
HTTP_PORT="${HTTP_PORT:-8001}"

echo "=========================================="
echo "Launching Diffusion LM Worker (LLaDA2.0)"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "TP Size: $TP_SIZE"
echo "Max Running Requests: $MAX_RUNNING_REQUESTS"
echo "Namespace: $NAMESPACE"
echo "Component: $COMPONENT"
echo "Frontend Port: $HTTP_PORT"
echo ""
echo "NOTE: Diffusion algorithm is HARDCODED to 'LowConfidence'"
echo "TODO: Add CLI flags for --dllm-algorithm and --dllm-algorithm-config"
echo "=========================================="

# Launch frontend (OpenAI-compatible API server)
echo "Starting Dynamo Frontend on port $HTTP_PORT..."
python -m dynamo.frontend \
    --http-port "$HTTP_PORT" \
    --store-kv "$STORE_KV" &
FRONTEND_PID=$!

# Wait for frontend to start
sleep 2

# Launch diffusion worker
echo "Starting Diffusion LM Worker..."
python -m dynamo.sglang \
    --model-path "$MODEL_PATH" \
    --tp-size "$TP_SIZE" \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --skip-tokenizer-init \
    --trust-remote-code \
    --endpoint "dyn://${NAMESPACE}.${COMPONENT}.${ENDPOINT}" \
    --store-kv "$STORE_KV" \
    --enable-metrics \
    --disable-cuda-graph \
    --attention-backend triton \
    --disable-overlap-schedule \
    --diffusion-worker
