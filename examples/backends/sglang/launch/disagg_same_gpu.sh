#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Parse command line arguments
GPU_MEM_FRACTION="0.45"
ENABLE_OTEL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --enable-otel)
            ENABLE_OTEL=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] [GPU_MEM_FRACTION]"
            echo "Options:"
            echo "  --enable-otel        Enable OpenTelemetry tracing"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Arguments:"
            echo "  GPU_MEM_FRACTION     Fraction of GPU memory to use per worker (default: 0.45)"
            echo ""
            echo "Note: System metrics are enabled by default on ports 8080 (frontend), 8081 (prefill), 8082 (decode)"
            exit 0
            ;;
        *)
            # Treat any other argument as GPU_MEM_FRACTION
            GPU_MEM_FRACTION="$1"
            shift
            ;;
    esac
done

# Check GPU memory before starting disaggregated mode on single GPU
FREE_GPU_GB=$(python3 -c "import torch; print(torch.cuda.mem_get_info()[0]/1024**3)" 2>/dev/null)
if [ $? -ne 0 ]; then
  echo "Error: Failed to check GPU memory. Is PyTorch with CUDA available?"
  exit 1
fi

REQUIRED_GB=16
# Use Python for floating-point comparison to avoid bc dependency
if python3 -c "import sys; sys.exit(0 if float('$FREE_GPU_GB') >= $REQUIRED_GB else 1)"; then
  echo "GPU memory check passed: ${FREE_GPU_GB}GB available (required: ${REQUIRED_GB}GB)"
else
  echo "Error: Insufficient GPU memory. Required: ${REQUIRED_GB}GB, Available: ${FREE_GPU_GB}GB"
  echo "Please free up GPU memory before running disaggregated mode on single GPU."
  exit 1
fi

# Setup cleanup trap
cleanup() {
    echo "Cleaning up background processes..."
    kill $DYNAMO_PID $PREFILL_PID 2>/dev/null || true
    wait $DYNAMO_PID $PREFILL_PID 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# Enable tracing if requested
if [ "$ENABLE_OTEL" = true ]; then
    export DYN_LOGGING_JSONL=true
    export OTEL_EXPORT_ENABLED=1
    export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT:-http://localhost:4317}
fi

# run ingress with KV router mode for disaggregated setup
OTEL_SERVICE_NAME=dynamo-frontend DYN_SYSTEM_PORT=8080 \
python3 -m dynamo.frontend --router-mode kv --http-port=8000 &
DYNAMO_PID=$!

# run prefill worker with metrics on port 8081
OTEL_SERVICE_NAME=dynamo-worker-prefill DYN_SYSTEM_PORT=8081 \
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
  --mem-fraction-static ${GPU_MEM_FRACTION} \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 4096 \
  --enable-memory-saver \
  --delete-ckpt-after-loading \
  --max-running-requests 2 \
  --enable-metrics &
PREFILL_PID=$!

# Wait for prefill worker to initialize before starting decode worker
# This prevents both workers from competing for GPU memory simultaneously, which can cause OOM.
# The prefill worker needs time to:
# 1. Load model weights and allocate its memory fraction
# 2. Initialize KV cache with --delete-ckpt-after-loading to free checkpoint memory
# 3. Register with NATS service discovery so decode worker can find it
echo "Waiting for prefill worker to initialize..."
sleep 5

# run decode worker with metrics on port 8082 (foreground)
OTEL_SERVICE_NAME=dynamo-worker-decode DYN_SYSTEM_PORT=8082 \
python3 -m dynamo.sglang \
  --model-path Qwen/Qwen3-0.6B \
  --served-model-name Qwen/Qwen3-0.6B \
  --page-size 16 \
  --tp 1 \
  --trust-remote-code \
  --disaggregation-mode decode \
  --disaggregation-bootstrap-port 12345 \
  --host 0.0.0.0 \
  --disaggregation-transfer-backend nixl \
  --mem-fraction-static ${GPU_MEM_FRACTION} \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 4096 \
  --enable-memory-saver \
  --delete-ckpt-after-loading \
  --max-running-requests 2 \
  --enable-metrics

