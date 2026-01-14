#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Launch script for Meituan LongCat models with tool calling support
# Requires 8xH100 (80GB) for the 560B MoE model

# Setup cleanup trap
cleanup() {
    echo "Cleaning up background processes..."
    kill $DYNAMO_PID 2>/dev/null || true
    wait $DYNAMO_PID 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# Default values
MODEL="meituan-longcat/LongCat-Flash-Chat"
TP_SIZE=8
ENABLE_OTEL=false
ENABLE_THINKING=false

# Parse command line arguments
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL="$2"
            shift 2
            ;;
        --tp)
            TP_SIZE="$2"
            shift 2
            ;;
        --enable-otel)
            ENABLE_OTEL=true
            shift
            ;;
        --enable-thinking)
            ENABLE_THINKING=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Launch Meituan LongCat model with tool calling support"
            echo ""
            echo "Options:"
            echo "  --model-path <name>  Specify model (default: $MODEL)"
            echo "  --tp <size>          Tensor parallel size (default: $TP_SIZE)"
            echo "  --enable-otel        Enable OpenTelemetry tracing"
            echo "  --enable-thinking    Enable reasoning parser for LongCat-Flash-Thinking"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Run LongCat-Flash-Chat with tool calling"
            echo "  $0"
            echo ""
            echo "  # Run LongCat-Flash-Thinking with reasoning"
            echo "  $0 --model-path meituan-longcat/LongCat-Flash-Thinking --enable-thinking"
            echo ""
            echo "Additional SGLang/Dynamo flags can be passed and will be forwarded"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
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

# Build reasoning parser args if thinking mode enabled
REASONING_ARGS=()
if [ "$ENABLE_THINKING" = true ]; then
    REASONING_ARGS+=(--dyn-reasoning-parser longcat)
fi

echo "Starting LongCat deployment..."
echo "  Model: $MODEL"
echo "  Tensor Parallel: $TP_SIZE"
echo "  Tool Call Parser: longcat"
if [ "$ENABLE_THINKING" = true ]; then
    echo "  Reasoning Parser: longcat"
fi
echo ""

# Run frontend
OTEL_SERVICE_NAME=dynamo-frontend \
python3 -m dynamo.frontend &
DYNAMO_PID=$!

# Wait for frontend to start
sleep 2

# Run worker with LongCat tool calling enabled
OTEL_SERVICE_NAME=dynamo-worker DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
python3 -m dynamo.sglang \
  --model-path "$MODEL" \
  --served-model-name "$MODEL" \
  --tp "$TP_SIZE" \
  --trust-remote-code \
  --mem-fraction-static 0.85 \
  --dyn-tool-call-parser longcat \
  "${REASONING_ARGS[@]}" \
  --enable-metrics \
  "${TRACE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
