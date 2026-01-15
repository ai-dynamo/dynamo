#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Disaggregated mode on single GPU - for testing only
# Both prefill and decode workers share the same GPU with reduced memory

# Check GPU memory availability
FREE_GPU_GB=$(python3 -c "import torch; print(torch.cuda.mem_get_info()[0]/1024**3)" 2>/dev/null)
if [ $? -ne 0 ]; then
    echo "Error: Failed to check GPU memory. Is PyTorch with CUDA available?"
    exit 1
fi

REQUIRED_GB=16
# Use bash arithmetic instead of bc to avoid external dependency
FREE_GPU_INT=$(python3 -c "print(int(float('$FREE_GPU_GB')))" 2>/dev/null)
if [ $? -ne 0 ]; then
    echo "Error: Failed to parse GPU memory value."
    exit 1
fi

if (( FREE_GPU_INT < REQUIRED_GB )); then
    echo "Error: Insufficient GPU memory. Required: ${REQUIRED_GB}GB, Available: ${FREE_GPU_GB}GB"
    echo "Please free up GPU memory before running disaggregated mode on single GPU."
    exit 1
fi

echo "GPU memory check passed: ${FREE_GPU_GB}GB available (required: ${REQUIRED_GB}GB)"

# Environment variables with defaults
export DYNAMO_HOME=${DYNAMO_HOME:-"/workspace"}
export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-0.6B"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"Qwen/Qwen3-0.6B"}
export PREFILL_ENGINE_ARGS=${PREFILL_ENGINE_ARGS:-"$DYNAMO_HOME/tests/serve/trtllm/engine_configs/qwen3/prefill.yaml"}
export DECODE_ENGINE_ARGS=${DECODE_ENGINE_ARGS:-"$DYNAMO_HOME/tests/serve/trtllm/engine_configs/qwen3/decode.yaml"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}
export MODALITY=${MODALITY:-"text"}

# Setup cleanup trap
cleanup() {
    echo "Cleaning up background processes..."
    kill $DYNAMO_PID $PREFILL_PID 2>/dev/null || true
    wait $DYNAMO_PID $PREFILL_PID 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

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
    TRACE_ARGS+=(--override-engine-args "{\"return_perf_metrics\": true, \"otlp_traces_endpoint\": \"${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT}\" }")
fi
# run frontend
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
OTEL_SERVICE_NAME=dynamo-frontend \
python3 -m dynamo.frontend &
DYNAMO_PID=$!

# run prefill worker (shares GPU with decode)
OTEL_SERVICE_NAME=dynamo-worker-prefill \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args  "$PREFILL_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --publish-events-and-metrics \
  --disaggregation-mode prefill \
  "${TRACE_ARGS[@]}" &
PREFILL_PID=$!

# run decode worker (shares GPU with prefill)
OTEL_SERVICE_NAME=dynamo-worker-decode \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args  "$DECODE_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --publish-events-and-metrics \
  --disaggregation-mode decode \
  "${TRACE_ARGS[@]}"

