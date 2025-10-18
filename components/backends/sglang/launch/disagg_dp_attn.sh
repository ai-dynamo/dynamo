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
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Enable tracing if requested
if [ "$ENABLE_OTEL" = true ]; then
    export DYN_LOGGING_JSONL=true
    export OTEL_EXPORT_ENABLED=1
    export OTEL_EXPORT_ENDPOINT=http://localhost:4317
    export DYN_SYSTEM_ENABLED=true
    export DYN_SYSTEM_PORT=8081
fi

# run ingress
OTEL_SERVICE_NAME=dynamo-frontend python3 -m dynamo.frontend --http-port=8000 &
DYNAMO_PID=$!

# run prefill worker
OTEL_SERVICE_NAME=dynamo-worker-prefill python3 -m dynamo.sglang \
  --model-path silence09/DeepSeek-R1-Small-2layers \
  --served-model-name silence09/DeepSeek-R1-Small-2layers \
  --tp 2 \
  --dp-size 2 \
  --page-size 16 \
  --enable-dp-attention \
  --trust-remote-code \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend nixl \
  --load-balance-method round_robin \
  --port 30000 &
PREFILL_PID=$!

# run decode worker
OTEL_SERVICE_NAME=dynamo-worker-decode CUDA_VISIBLE_DEVICES=2,3 python3 -m dynamo.sglang \
  --model-path silence09/DeepSeek-R1-Small-2layers \
  --served-model-name silence09/DeepSeek-R1-Small-2layers \
  --tp 2 \
  --dp-size 2 \
  --page-size 16 \
  --enable-dp-attention \
  --trust-remote-code \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend nixl \
  --prefill-round-robin-balance \
  --port 31000