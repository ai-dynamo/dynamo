#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Setup cleanup trap
cleanup() {
    echo "Cleaning up background processes..."
    kill $DYNAMO_PID $PREFILL_PID1 $PREFILL_PID2 $DECODE_PID1 $DECODE_PID2 2>/dev/null || true
    wait $DYNAMO_PID $PREFILL_PID1 $PREFILL_PID2 $DECODE_PID1 $DECODE_PID2 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# Parse command line arguments
ENABLE_OTEL=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --enable-otel)
            ENABLE_OTEL=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --enable-otel        Enable OpenTelemetry tracing"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Note: System metrics are enabled by default on ports:"
            echo "  8081-8082 (prefill workers), 8083-8084 (decode workers)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Enable tracing if requested
TRACE_ARGS=()
if [ "$ENABLE_OTEL" = true ]; then
    export DYN_LOGGING_JSONL=true
    export OTEL_EXPORT_ENABLED=1
    export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT:-http://localhost:4317}
    TRACE_ARGS+=(--enable-trace --otlp-traces-endpoint localhost:4317)
fi

# Start frontend with KV routing
# The frontend will automatically detect prefill workers and activate an internal prefill router
# No standalone prefill router needed - the frontend handles prefill routing internally
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
OTEL_SERVICE_NAME=dynamo-frontend \
python3 -m dynamo.frontend \
    --router-mode kv \
    --router-reset-states &
DYNAMO_PID=$!

# run prefill worker
OTEL_SERVICE_NAME=dynamo-worker-prefill-1 DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
python3 -m dynamo.sglang \
  --model-path Qwen/Qwen3-0.6B \
  --served-model-name Qwen/Qwen3-0.6B \
  --page-size 64 \
  --tp 1 \
  --trust-remote-code \
  --disaggregation-mode prefill \
  --host 0.0.0.0 \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}' \
  --disaggregation-transfer-backend nixl \
  --enable-metrics \
  "${TRACE_ARGS[@]}" &
PREFILL_PID1=$!

# run prefill worker
OTEL_SERVICE_NAME=dynamo-worker-prefill-2 DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
CUDA_VISIBLE_DEVICES=1 python3 -m dynamo.sglang \
  --model-path Qwen/Qwen3-0.6B \
  --served-model-name Qwen/Qwen3-0.6B \
  --page-size 64 \
  --tp 1 \
  --trust-remote-code \
  --disaggregation-mode prefill \
  --host 0.0.0.0 \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5558"}' \
  --disaggregation-transfer-backend nixl \
  --enable-metrics \
  "${TRACE_ARGS[@]}" &
PREFILL_PID2=$!

# run decode worker
OTEL_SERVICE_NAME=dynamo-worker-decode-1 DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT3:-8083} \
CUDA_VISIBLE_DEVICES=3 python3 -m dynamo.sglang \
  --model-path Qwen/Qwen3-0.6B \
  --served-model-name Qwen/Qwen3-0.6B \
  --page-size 64 \
  --tp 1 \
  --trust-remote-code \
  --disaggregation-mode decode \
  --host 0.0.0.0 \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5560"}' \
  --disaggregation-transfer-backend nixl \
  --enable-metrics \
  "${TRACE_ARGS[@]}" &
DECODE_PID1=$!

# run decode worker
OTEL_SERVICE_NAME=dynamo-worker-decode-2 DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT4:-8084} \
CUDA_VISIBLE_DEVICES=2 python3 -m dynamo.sglang \
  --model-path Qwen/Qwen3-0.6B \
  --served-model-name Qwen/Qwen3-0.6B \
  --page-size 64 \
  --tp 1 \
  --trust-remote-code \
  --disaggregation-mode decode \
  --host 0.0.0.0 \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5559"}' \
  --disaggregation-transfer-backend nixl \
  --enable-metrics \
  "${TRACE_ARGS[@]}" &
DECODE_PID2=$!

# Wait for any worker to exit (keeps script running)
wait
