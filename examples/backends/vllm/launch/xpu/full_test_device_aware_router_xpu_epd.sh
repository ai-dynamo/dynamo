#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Full lifecycle test for Device-Aware Weighted Router with XPU E/P/D
# This script:
#   1. Cleans up any existing processes
#   2. Starts frontend with device-aware-weighted router
#   3. Starts XPU E/P/D server
#   4. Runs benchmarks for multiple request rates with health checks

set -e

MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
DATASET="random-mm"
NUM_PROMPTS=32
SEED=0
PORT=8000
INPUT_LEN=128
OUTPUT_LEN=128
NUM_IMAGES=8
IMAGE_RESOLUTION="(640, 480, 1)"
REQUEST_RATES=(0.2 0.5 0.8 1.0 1.2 1.5 2.0)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE_DIR="/workspace/benchmark_DAR_XPU_EPD_${TIMESTAMP}"
mkdir -p "$OUTPUT_BASE_DIR"

echo "=========================================="
echo "Device-Aware Router XPU E/P/D Full Test"
echo "=========================================="
echo "Model: $MODEL"
echo "Configuration: XPU Encode + XPU Prefill + XPU Decode"
echo "Router Mode: device-aware-weighted"
echo "Dataset: $DATASET"
echo "Requests: $NUM_PROMPTS"
echo "Input Length: $INPUT_LEN tokens"
echo "Output Length: $OUTPUT_LEN tokens"
echo "Images per request: $NUM_IMAGES @ 480p (854x480)"
echo "Request rates: ${REQUEST_RATES[*]}"
echo "Output base: $OUTPUT_BASE_DIR"
echo "=========================================="
echo ""

# Function to check if server is healthy
check_server_health() {
    local max_attempts=30
    local attempt=1

    echo "Checking server health..."
    while [ $attempt -le $max_attempts ]; do
        if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "✓ Server is healthy (attempt $attempt)"
            return 0
        fi
        echo "  Waiting for server... (attempt $attempt/$max_attempts)"
        sleep 2
        attempt=$((attempt + 1))
    done

    echo "✗ Server health check failed after $max_attempts attempts"
    return 1
}

# Function to verify server responds to requests
verify_server_responsive() {
    local max_attempts=5
    local attempt=1

    echo "Verifying server responsiveness with test request..."
    while [ $attempt -le $max_attempts ]; do
        if vllm bench serve \
            --model "$MODEL" \
            --backend openai-chat \
            --endpoint /v1/chat/completions \
            --dataset-name random-mm \
            --input-len $INPUT_LEN \
            --output-len $OUTPUT_LEN \
            --random-mm-base-items-per-request $NUM_IMAGES \
            --random-mm-bucket-config "{$IMAGE_RESOLUTION: 1.0}" \
            --seed $SEED \
            --num-prompts 1 \
            --port $PORT \
            > /dev/null 2>&1; then
            echo "✓ Server responded successfully"
            return 0
        fi
        echo "  Server not responsive yet... (attempt $attempt/$max_attempts)"
        sleep 5
        attempt=$((attempt + 1))
    done

    echo "✗ Server failed to respond after $max_attempts attempts"
    return 1
}

# Cleanup function
cleanup_processes() {
    echo "Cleaning up existing processes..."
    pkill -9 -f "dynamo.vllm" 2>/dev/null || true
    pkill -9 -f "dynamo.frontend" 2>/dev/null || true
    sleep 3
    echo "✓ Cleanup complete"
}

# Initial cleanup
cleanup_processes

# Set environment variables
export DYN_HTTP_PORT="$PORT"
export DEVICE_PLATFORM="xpu"
export DYN_ENCODE_WORKER_GPU=0
export DYN_PREFILL_WORKER_GPU=1
export DYN_DECODE_WORKER_GPU=2

# Start frontend
echo ""
echo "Starting frontend with device-aware-weighted router..."
python -m dynamo.frontend --router-mode device-aware-weighted \
    > "$OUTPUT_BASE_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!
echo "Frontend started (PID: $FRONTEND_PID)"
sleep 5

# Check if frontend is still running
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo "✗ Frontend failed to start. Check logs: $OUTPUT_BASE_DIR/frontend.log"
    exit 1
fi

# Start XPU E/P/D server
echo ""
echo "Starting XPU E/P/D server..."
echo "  Encode Worker: GPU $DYN_ENCODE_WORKER_GPU"
echo "  Prefill Worker: GPU $DYN_PREFILL_WORKER_GPU"
echo "  Decode Worker: GPU $DYN_DECODE_WORKER_GPU"
echo "  Router Mode: device-aware-weighted"
echo "  XPU to CPU Encoder Ratio: 8:1"

DYN_ROUTER_MODE='device-aware-weighted' \
DYN_ENCODER_XPU_TO_CPU_RATIO=8 \
DYN_ENCODE_WORKER_GPU=0 \
DYN_PREFILL_WORKER_GPU=1 \
DYN_DECODE_WORKER_GPU=2 \
DEVICE_PLATFORM='xpu' \
bash examples/backends/vllm/launch/xpu/disagg_multimodal_epd_xpu.sh \
    --model "$MODEL" \
    > "$OUTPUT_BASE_DIR/server.log" 2>&1 &
SERVER_PID=$!
echo "Server started (PID: $SERVER_PID)"

# Wait for server to be ready
echo ""
echo "Waiting for server initialization..."
sleep 30

# Verify server health
if ! check_server_health; then
    echo "✗ Server health check failed. Check logs:"
    echo "  Frontend: $OUTPUT_BASE_DIR/frontend.log"
    echo "  Server: $OUTPUT_BASE_DIR/server.log"
    cleanup_processes
    exit 1
fi

# Verify server is responsive
if ! verify_server_responsive; then
    echo "✗ Server responsiveness check failed. Check logs:"
    echo "  Frontend: $OUTPUT_BASE_DIR/frontend.log"
    echo "  Server: $OUTPUT_BASE_DIR/server.log"
    cleanup_processes
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ Server Ready - Starting Benchmarks"
echo "=========================================="

# Run warmup
echo ""
echo "Running warmup (5 requests)..."
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
    --num-prompts 5 \
    --port $PORT \
    > "$OUTPUT_BASE_DIR/warmup.log" 2>&1

if [ $? -ne 0 ]; then
    echo "✗ Warmup failed. Check log: $OUTPUT_BASE_DIR/warmup.log"
    cleanup_processes
    exit 1
fi
echo "✓ Warmup complete"

sleep 10

# Run benchmarks for each request rate
SUCCESS_COUNT=0
TOTAL_RATES=${#REQUEST_RATES[@]}

for rate in "${REQUEST_RATES[@]}"; do
    OUTPUT_DIR="$OUTPUT_BASE_DIR/rate_${rate}"
    mkdir -p "$OUTPUT_DIR"

    echo ""
    echo "=========================================="
    echo "Benchmark: request-rate=$rate req/s"
    echo "=========================================="

    # Health check before starting benchmark
    if ! check_server_health; then
        echo "✗ Server not healthy before rate=$rate benchmark"
        echo "  Attempting to continue anyway..."
    fi

    # Run mini-warmup between rates
    echo "Running mini-warmup before rate=$rate..."
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
        --num-prompts 2 \
        --port $PORT \
        > "$OUTPUT_DIR/mini_warmup.log" 2>&1

    if [ $? -ne 0 ]; then
        echo "✗ Mini-warmup failed for rate=$rate. Server may be stuck."
        echo "  Check log: $OUTPUT_DIR/mini_warmup.log"
        break
    fi

    sleep 5

    # Run actual benchmark
    echo "Running benchmark for rate=$rate..."
    START_TIME=$(date +%s)

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

    BENCHMARK_EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    if [ $BENCHMARK_EXIT_CODE -eq 0 ]; then
        echo "✓ Completed rate=$rate in ${DURATION}s"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        sleep 15  # Longer sleep after successful benchmark
    else
        echo "✗ Benchmark failed for rate=$rate (exit code: $BENCHMARK_EXIT_CODE)"
        echo "  Duration: ${DURATION}s"
        echo "  Check results: $OUTPUT_DIR/results.txt"
        break
    fi
done

echo ""
echo "=========================================="
echo "Benchmark Summary"
echo "=========================================="
echo "Successful benchmarks: $SUCCESS_COUNT / $TOTAL_RATES"
echo "Results directory: $OUTPUT_BASE_DIR"
echo ""

if [ $SUCCESS_COUNT -eq $TOTAL_RATES ]; then
    echo "✓ ALL BENCHMARKS PASSED!"
    echo ""
    echo "Individual results:"
    for rate in "${REQUEST_RATES[@]}"; do
        if [ -f "$OUTPUT_BASE_DIR/rate_${rate}/results.txt" ]; then
            echo "  - Rate $rate req/s: $OUTPUT_BASE_DIR/rate_${rate}/results.txt"
        fi
    done
else
    echo "✗ Some benchmarks failed or were skipped"
    echo ""
    echo "Completed results:"
    for rate in "${REQUEST_RATES[@]}"; do
        if [ -f "$OUTPUT_BASE_DIR/rate_${rate}/results.txt" ]; then
            echo "  ✓ Rate $rate req/s: $OUTPUT_BASE_DIR/rate_${rate}/results.txt"
        else
            echo "  ✗ Rate $rate req/s: Not completed"
        fi
    done
fi

echo ""
echo "Logs:"
echo "  Frontend: $OUTPUT_BASE_DIR/frontend.log"
echo "  Server: $OUTPUT_BASE_DIR/server.log"
echo "  Warmup: $OUTPUT_BASE_DIR/warmup.log"
echo ""

# Cleanup
echo "Cleaning up processes..."
cleanup_processes

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="

exit 0
