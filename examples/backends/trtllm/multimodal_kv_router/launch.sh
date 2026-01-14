#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch script for Dynamo TRT-LLM Multimodal KV Router Example
#
# This script launches multiple TRT-LLM workers on a single GPU (H100 80GB can fit 2x Qwen2-VL-2B)
# and a custom API frontend with MM-aware KV routing.
#
# Usage:
#   ./launch.sh                           # Default: 2 workers, Qwen2-VL-2B
#   NUM_WORKERS=3 ./launch.sh             # 3 workers
#   MODEL=Qwen/Qwen2.5-VL-7B ./launch.sh  # Different model

set -e

# Configuration
MODEL="${MODEL:-Qwen/Qwen2-VL-2B-Instruct}"
MODEL_TYPE="${MODEL_TYPE:-qwen2_vl}"
NUM_WORKERS="${NUM_WORKERS:-2}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
GPU_ID="${GPU_ID:-0}"
API_PORT="${API_PORT:-8000}"

# Dynamo configuration
NAMESPACE="${NAMESPACE:-default}"
COMPONENT="${COMPONENT:-trtllm}"
ENDPOINT="${ENDPOINT:-generate}"

# Get Dynamo home directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMO_HOME="${DYNAMO_HOME:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"

echo "=============================================="
echo "Dynamo TRT-LLM Multimodal KV Router Example"
echo "=============================================="
echo "Model: $MODEL"
echo "Model Type: $MODEL_TYPE"
echo "Workers: $NUM_WORKERS"
echo "Block Size: $BLOCK_SIZE"
echo "GPU: $GPU_ID"
echo "API Port: $API_PORT"
echo "Dynamo Home: $DYNAMO_HOME"
echo "=============================================="

# Cleanup function
cleanup() {
    echo "Stopping services..."
    # Kill background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    # Stop NATS if we started it
    if [ "$NATS_STARTED" = "true" ]; then
        docker compose -f "$DYNAMO_HOME/deploy/docker-compose.yml" stop nats 2>/dev/null || true
    fi
    echo "Cleanup complete"
}
trap cleanup EXIT

# Check if NATS is running, start if not
check_nats() {
    if docker ps --format '{{.Names}}' | grep -q nats; then
        echo "NATS is already running"
        return 0
    fi

    echo "Starting NATS server..."
    if [ -f "$DYNAMO_HOME/deploy/docker-compose.yml" ]; then
        docker compose -f "$DYNAMO_HOME/deploy/docker-compose.yml" up -d nats
        NATS_STARTED="true"
        sleep 5
        echo "NATS started"
    else
        echo "ERROR: docker-compose.yml not found at $DYNAMO_HOME/deploy/"
        echo "Please start NATS manually or set DYNAMO_HOME correctly"
        exit 1
    fi
}

# Start NATS
check_nats

# Start TRT-LLM workers
echo ""
echo "Starting $NUM_WORKERS TRT-LLM worker(s)..."
WORKER_PIDS=()

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    echo "  Starting worker $i..."

    CUDA_VISIBLE_DEVICES=$GPU_ID python -m dynamo.trtllm \
        --model "$MODEL" \
        --namespace "$NAMESPACE" \
        --component "$COMPONENT" \
        --endpoint "$ENDPOINT" \
        --kv-block-size "$BLOCK_SIZE" \
        --publish-events-and-metrics \
        --store-kv file \
        2>&1 | sed "s/^/[worker-$i] /" &

    WORKER_PIDS+=($!)

    # Wait between workers to avoid memory allocation conflicts
    if [ $i -lt $((NUM_WORKERS - 1)) ]; then
        echo "  Waiting 30s before starting next worker..."
        sleep 30
    fi
done

echo ""
echo "All workers started. PIDs: ${WORKER_PIDS[*]}"

# Wait for workers to initialize
echo ""
echo "Waiting 60s for workers to fully initialize..."
sleep 60

# Start API frontend
echo ""
echo "Starting API frontend on port $API_PORT..."
python "$SCRIPT_DIR/api.py" \
    --model "$MODEL" \
    --model-type "$MODEL_TYPE" \
    --block-size "$BLOCK_SIZE" \
    --namespace "$NAMESPACE" \
    --component "$COMPONENT" \
    --endpoint "$ENDPOINT" \
    --http-port "$API_PORT"
