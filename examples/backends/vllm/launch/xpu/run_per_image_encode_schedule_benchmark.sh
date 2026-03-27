#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run benchmark with PER-IMAGE scheduler only

set -e

MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
DATASET="lmarena-ai/VisionArena-Chat"
NUM_PROMPTS=10
SEED=0
PORT=8000

OUTPUT_DIR="/workspace/benchmark_results_per_image_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Per-Image Scheduler Benchmark"
echo "=========================================="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Requests: $NUM_PROMPTS"
echo "Output: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Cleanup any existing processes
echo "Cleaning up existing processes..."
pkill -9 -f "dynamo.vllm" || true
pkill -9 -f "dynamo.frontend" || true
sleep 5

# Start server with per-image scheduler
echo "Starting server with per-image scheduler..."
export DYN_ENCODER_SCHEDULER="per_image"
export DEVICE_PLATFORM="xpu"
export DYN_HTTP_PORT="$PORT"

bash /workspace/examples/backends/vllm/launch/xpu/dual_encoders_for_epd.sh \
    --model "$MODEL" > "$OUTPUT_DIR/server.log" 2>&1 &

SERVER_PID=$!
echo "Server started (PID: $SERVER_PID)"

# Wait for server
echo "Waiting for server to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "✓ Server is ready!"
        break
    fi
    sleep 2
done

# Check all workers registered
echo "Checking worker registration..."
curl -s http://localhost:$PORT/health | python -m json.tool

# Skip warmup - server needs more time to be ready
# Warmup was failing with "Not Found" errors and causing GIL crashes
echo ""
echo "Skipping warmup, waiting for server to be fully ready..."
sleep 100

# Run benchmark
echo ""
echo "Running benchmark ($NUM_PROMPTS requests)..."
vllm bench serve \
    --model "$MODEL" \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --dataset-name hf \
    --dataset-path "$DATASET" \
    --seed $SEED \
    --num-prompts $NUM_PROMPTS \
    --port $PORT \
    2>&1 | tee "$OUTPUT_DIR/results.txt"

echo ""
echo "=========================================="
echo "✓ Benchmark Complete!"
echo "=========================================="
echo "Results: $OUTPUT_DIR/results.txt"
echo "Logs: $OUTPUT_DIR/server.log"
echo ""
echo "To stop server:"
echo "  pkill -9 -f 'dynamo.vllm'"
echo "  pkill -9 -f 'dynamo.frontend'"
echo ""
