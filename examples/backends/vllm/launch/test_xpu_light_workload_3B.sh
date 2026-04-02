#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Lightweight test configuration for XPU hardware limitations
# Configuration: 1 image @ 480p, lower request rates

set -e

MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
DATASET="random-mm"
NUM_PROMPTS=32  # Reduced from 32
SEED=0
PORT=8000
INPUT_LEN=128
OUTPUT_LEN=128  # Reduced from 256 to lower decode time
NUM_IMAGES=1    # Reduced to 1 image
IMAGE_RESOLUTION="(640, 480, 1)"  # 480p (SD) instead of 1080p
REQUEST_RATES=(0.2 0.5 0.8 1.0 1.2 1.5 2.0)  # Extended rate range for detailed analysis

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE_DIR="/workspace/benchmark_xpu_light_3B_image8_request32_again_${TIMESTAMP}"
mkdir -p "$OUTPUT_BASE_DIR"

echo "=========================================="
echo "XPU-Only E/P/D Benchmark (Light Workload)"
echo "=========================================="
echo "Model: $MODEL"
echo "Configuration: XPU Encoder + XPU Prefill + XPU Decode"
echo "Dataset: $DATASET"
echo "Requests: $NUM_PROMPTS"
echo "Input Length: $INPUT_LEN tokens"
echo "Output Length: $OUTPUT_LEN tokens"
echo "Images per request: $NUM_IMAGES @ 480p (854x480)"
echo "Request rates: ${REQUEST_RATES[*]}"
echo "Output base: $OUTPUT_BASE_DIR"
echo "=========================================="
echo ""

# Cleanup any existing processes
echo "Cleaning up existing processes..."
pkill -9 -f "dynamo.vllm" 2>/dev/null || true
pkill -9 -f "dynamo.frontend" 2>/dev/null || true
sleep 5

# Pre-download model to avoid lock contention
echo ""
echo "Pre-downloading model to avoid lock contention..."
bash /workspace/predownload_model.sh "$MODEL" | tee "$OUTPUT_BASE_DIR/model_download.log"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "✗ Model download failed. Check logs: $OUTPUT_BASE_DIR/model_download.log"
    exit 1
fi
echo ""

# Start server (XPU only - all workers on XPU)
echo "Starting XPU-only server..."
export DEVICE_PLATFORM="xpu"
export DYN_HTTP_PORT="$PORT"
export DYN_ENCODE_WORKER_GPU=5
export DYN_PREFILL_WORKER_GPU=6
export DYN_DECODE_WORKER_GPU=7
# Reduce max-model-len to avoid OOM (10240 requires 1.41 GiB but only 1.37 GiB available)
export DYN_PREFILL_GPU_MEM=0.95
export DYN_DECODE_GPU_MEM=0.95

bash /workspace/examples/backends/vllm/launch/xpu/disagg_multimodal_epd_xpu.sh \
    --model "$MODEL" > "$OUTPUT_BASE_DIR/server.log" 2>&1 &

SERVER_PID=$!
echo "Server started (PID: $SERVER_PID)"

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in {1..120}; do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "✓ Server is ready!"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "✗ Server failed to start within 4 minutes"
        cat "$OUTPUT_BASE_DIR/server.log"
        exit 1
    fi
    sleep 2
done

# Check worker registration
echo "Checking worker registration..."
curl -s http://localhost:$PORT/health | python -m json.tool > "$OUTPUT_BASE_DIR/health_check.json"
cat "$OUTPUT_BASE_DIR/health_check.json"

echo ""
echo "Running warmup (3 requests)..."
vllm bench serve \
    --model "$MODEL" \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --dataset-name random-mm \
    --input-len $INPUT_LEN \
    --output-len $OUTPUT_LEN \
    --random-mm-base-items-per-request $NUM_IMAGES \
    --random-mm-bucket-config "{$IMAGE_RESOLUTION: 1.0}" \
    --seed $SEED \
    --num-prompts 3 \
    --port $PORT \
    > /dev/null 2>&1 || true

sleep 10

# Run benchmarks for each request rate
for rate in "${REQUEST_RATES[@]}"; do
    OUTPUT_DIR="$OUTPUT_BASE_DIR/rate_${rate}"
    mkdir -p "$OUTPUT_DIR"

    echo ""
    echo "=========================================="
    echo "Running benchmark: request-rate=$rate req/s"
    echo "=========================================="

    vllm bench serve \
        --model "$MODEL" \
        --backend openai-chat \
        --endpoint /v1/chat/completions \
        --dataset-name random-mm \
        --input-len $INPUT_LEN \
        --output-len $OUTPUT_LEN \
        --random-mm-base-items-per-request $NUM_IMAGES \
        --random-mm-bucket-config "{$IMAGE_RESOLUTION: 1.0}" \
        --seed $SEED \
        --num-prompts $NUM_PROMPTS \
        --request-rate $rate \
        --port $PORT \
        2>&1 | tee "$OUTPUT_DIR/results.txt"

    echo "✓ Completed rate=$rate, results saved to $OUTPUT_DIR"
    sleep 5
done

echo ""
echo "=========================================="
echo "✓ All Benchmarks Complete!"
echo "=========================================="
echo "Configuration: XPU Encoder + XPU Prefill + XPU Decode"
echo "Results directory: $OUTPUT_BASE_DIR"
echo ""
echo "Individual results:"
for rate in "${REQUEST_RATES[@]}"; do
    echo "  - Rate $rate req/s: $OUTPUT_BASE_DIR/rate_${rate}/results.txt"
done
echo ""
echo "Server logs: $OUTPUT_BASE_DIR/server.log"
echo ""
echo "To stop server:"
echo "  pkill -9 -f 'dynamo.vllm'"
echo "  pkill -9 -f 'dynamo.frontend'"
echo ""
