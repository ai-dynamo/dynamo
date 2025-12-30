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
            echo "Note: System metrics are enabled by default on ports 8081 (prefill), 8082 (decode)"
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

# run ingress
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
OTEL_SERVICE_NAME=dynamo-frontend \
python3 -m dynamo.frontend &
DYNAMO_PID=$!

#AssertionError: Prefill round robin balance is required when dp size > 1. Please make sure that the prefill instance is launched with `--load-balance-method round_robin` and `--prefill-round-robin-balance` is set for decode server.

# run prefill worker
# Use DYN_SYSTEM_PORT1/2 instead of *_PREFILL/*_DECODE env names so test
# harnesses can set one simple pair for disaggregated deployments.
OTEL_SERVICE_NAME=dynamo-worker-prefill DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
python3 -m dynamo.sglang \
  --model-path silence09/DeepSeek-R1-Small-2layers \
  --served-model-name silence09/DeepSeek-R1-Small-2layers \
  --page-size 16 \
  --tp 2 --dp-size 2 --enable-dp-attention \
  --load-balance-method round_robin \
  --trust-remote-code \
  --disaggregation-mode prefill \
  --disaggregation-bootstrap-port 12345 \
  --host 0.0.0.0 \
  --port 40000 \
  --disaggregation-transfer-backend nixl \
  --enable-metrics \
  "${TRACE_ARGS[@]}" &
PREFILL_PID=$!

# run decode worker
OTEL_SERVICE_NAME=dynamo-worker-decode DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
CUDA_VISIBLE_DEVICES=2,3 python3 -m dynamo.sglang \
  --model-path silence09/DeepSeek-R1-Small-2layers \
  --served-model-name silence09/DeepSeek-R1-Small-2layers \
  --page-size 16 \
  --prefill-round-robin-balance \
  --tp 2 --dp-size 2 --enable-dp-attention \
  --trust-remote-code \
  --disaggregation-mode decode \
  --disaggregation-bootstrap-port 12345 \
  --host 0.0.0.0 \
  --disaggregation-transfer-backend nixl \
  --enable-metrics \
  "${TRACE_ARGS[@]}"
