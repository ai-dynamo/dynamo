#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Launch script for MM Router Worker with TRT-LLM backend
#
# This script starts:
# 1. TRT-LLM workers (standard, with KV event publishing)
# 2. MM Router Worker (computes mm_hash, routes to best worker)
# 3. Frontend (HTTP ingress)

set -e

# Get the directory where this script is located and navigate to dynamo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "$DYNAMO_ROOT"
echo "Working directory: $(pwd)"

# Configuration
MODEL="${MODEL:-Qwen/Qwen2-VL-2B-Instruct}"
MODEL_TYPE="${MODEL_TYPE:-qwen2_vl}"
NAMESPACE="${NAMESPACE:-default}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
HTTP_PORT="${HTTP_PORT:-8000}"
NUM_WORKERS="${NUM_WORKERS:-1}"

echo "=== MM Router Worker Launch Script ==="
echo "Model: $MODEL"
echo "Model Type: $MODEL_TYPE"
echo "Namespace: $NAMESPACE"
echo "Block Size: $BLOCK_SIZE"
echo "HTTP Port: $HTTP_PORT"
echo "Num Workers: $NUM_WORKERS"
echo ""

# Collect PIDs for cleanup
PIDS=()

cleanup() {
    echo "Cleaning up..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
}
trap cleanup EXIT

# Start TRT-LLM workers
# Use a different served-model-name so Frontend routes to MM Router instead
# Use NATS request plane to match MM Router
echo ""
echo "=== Starting TRT-LLM Workers ==="
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    echo "Starting TRT-LLM worker $i..."
    DYN_REQUEST_PLANE=nats python -m dynamo.trtllm \
        --model-path "$MODEL" \
        --served-model-name "${MODEL}__internal" \
        --endpoint "dyn://${NAMESPACE}.trtllm.generate" \
        --modality multimodal \
        --publish-events-and-metrics \
        --kv-block-size "$BLOCK_SIZE" \
        2>&1 | sed "s/^/[trtllm-$i] /" &
    PIDS+=($!)
done

# Wait for workers to initialize
echo "Waiting for TRT-LLM workers to initialize..."
sleep 15

# Start MM Router Worker
# Use NATS request plane to match Frontend
echo ""
echo "=== Starting MM Router Worker ==="
DYN_REQUEST_PLANE=nats python -m examples.backends.trtllm.mm_router_worker \
    --model "$MODEL" \
    --model-type "$MODEL_TYPE" \
    --namespace "$NAMESPACE" \
    --component mm_router \
    --endpoint generate \
    --downstream-component trtllm \
    --downstream-endpoint generate \
    --block-size "$BLOCK_SIZE" \
    2>&1 | sed "s/^/[mm_router] /" &
PIDS+=($!)

# Wait for router to initialize
echo "Waiting for MM Router to initialize..."
sleep 5

# Start Frontend
# Use NATS request plane to match MM Router
echo ""
echo "=== Starting Frontend ==="
DYN_REQUEST_PLANE=nats python -m dynamo.frontend \
    --http-port "$HTTP_PORT" \
    --router-mode round-robin \
    2>&1 | sed "s/^/[frontend] /" &
PIDS+=($!)

echo ""
echo "=== All services started ==="
echo "Frontend available at http://localhost:$HTTP_PORT"
echo ""
echo "Test with:"
echo "curl http://localhost:$HTTP_PORT/v1/chat/completions \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{"
echo "    \"model\": \"$MODEL\","
echo "    \"messages\": [{"
echo "      \"role\": \"user\","
echo "      \"content\": [{"
echo "        \"type\": \"text\","
echo "        \"text\": \"Describe this image\""
echo "      }, {"
echo "        \"type\": \"image_url\","
echo "        \"image_url\": {\"url\": \"https://example.com/image.jpg\"}"
echo "      }]"
echo "    }],"
echo "    \"max_tokens\": 100"
echo "  }'"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for all background processes
wait
