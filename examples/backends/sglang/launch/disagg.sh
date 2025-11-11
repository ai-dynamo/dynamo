#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Setup cleanup trap
cleanup() {
    echo "Cleaning up background processes..."
    kill $DYNAMO_PID $PREFILL_PID 2>/dev/null || true
    wait $DYNAMO_PID $PREFILL_PID 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# Parse command line arguments
ENABLE_OTEL=false
ENABLE_METRICS=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --enable-otel)
            ENABLE_OTEL=true
            shift
            ;;
        --enable-metrics)
            ENABLE_METRICS=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --enable-otel        Enable OpenTelemetry tracing"
            echo "  --enable-metrics     Enable system metrics server"
            echo "  -h, --help           Show this help message"
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
if [ "$ENABLE_OTEL" = true ]; then
    export DYN_LOGGING_JSONL=true
    export OTEL_EXPORT_ENABLED=1
    export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT:-http://localhost:4317}
fi

# Set up metrics ports if requested
METRICS_ARGS=""
FRONTEND_METRICS_PORT=""
PREFILL_METRICS_PORT=""
DECODE_METRICS_PORT=""
if [ "$ENABLE_METRICS" = true ]; then
    FRONTEND_METRICS_PORT="8080"
    PREFILL_METRICS_PORT="8081"
    DECODE_METRICS_PORT="8082"
    METRICS_ARGS="--enable-metrics"
fi

# Enable metrics for OTEL (uses same ports as metrics flag)
if [ "$ENABLE_OTEL" = true ]; then
    PREFILL_METRICS_PORT="${PREFILL_METRICS_PORT:-8081}"
    DECODE_METRICS_PORT="${DECODE_METRICS_PORT:-8082}"
fi

# run ingress
OTEL_SERVICE_NAME=dynamo-frontend DYN_SYSTEM_PORT=$FRONTEND_METRICS_PORT \
python3 -m dynamo.frontend --http-port=8000 &
DYNAMO_PID=$!

# run prefill worker
OTEL_SERVICE_NAME=dynamo-worker-prefill DYN_SYSTEM_PORT=$PREFILL_METRICS_PORT \
python3 -m dynamo.sglang \
  --model-path Qwen/Qwen3-0.6B \
  --served-model-name Qwen/Qwen3-0.6B \
  --page-size 16 \
  --tp 1 \
  --trust-remote-code \
  --disaggregation-mode prefill \
  --disaggregation-bootstrap-port 12345 \
  --host 0.0.0.0 \
  --disaggregation-transfer-backend nixl \
  $METRICS_ARGS &
PREFILL_PID=$!

# run decode worker
OTEL_SERVICE_NAME=dynamo-worker-decode DYN_SYSTEM_PORT=$DECODE_METRICS_PORT \
CUDA_VISIBLE_DEVICES=1 python3 -m dynamo.sglang \
  --model-path Qwen/Qwen3-0.6B \
  --served-model-name Qwen/Qwen3-0.6B \
  --page-size 16 \
  --tp 1 \
  --trust-remote-code \
  --disaggregation-mode decode \
  --disaggregation-bootstrap-port 12345 \
  --host 0.0.0.0 \
  --disaggregation-transfer-backend nixl \
  $METRICS_ARGS
