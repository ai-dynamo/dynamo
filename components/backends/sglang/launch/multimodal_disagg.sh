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

# Default values
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
CHAT_TEMPLATE="qwen2-vl"
PROVIDED_CHAT_TEMPLATE=""
ENABLE_OTEL=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME=$2
            shift 2
            ;;
        --chat-template)
            PROVIDED_CHAT_TEMPLATE=$2
            shift 2
            ;;
        --enable-otel)
            ENABLE_OTEL=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model <model_name> Specify the model to use (default: $MODEL_NAME)"
            echo "  --chat-template <template> Specify the SGLang chat template to use (default: $CHAT_TEMPLATE)"
            echo "  --enable-otel        Enable OpenTelemetry tracing"
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

# Set CHAT_TEMPLATE if provided
if [[ -n "$PROVIDED_CHAT_TEMPLATE" ]]; then
    CHAT_TEMPLATE="$PROVIDED_CHAT_TEMPLATE"
fi

# Enable tracing if requested
if [ "$ENABLE_OTEL" = true ]; then
    export DYN_LOGGING_JSONL=true
    export OTEL_EXPORT_ENABLED=1
    export OTEL_EXPORT_ENDPOINT=http://localhost:4317
    export DYN_SYSTEM_ENABLED=true
    export DYN_SYSTEM_PORT=8081
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_BACKEND_DIR="$SCRIPT_DIR/src"


# run ingress
export OTEL_SERVICE_NAME=dynamo-frontend
python3 -m dynamo.frontend --http-port=8000 &
DYNAMO_PID=$!

# run SGLang multimodal processor
OTEL_SERVICE_NAME=dynamo-multimodal-processor python3 -m dynamo.sglang --multimodal-processor --model-path "$MODEL_NAME" --chat-template "$CHAT_TEMPLATE" &

# run SGLang multimodal encode worker
OTEL_SERVICE_NAME=dynamo-multimodal-encode CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.sglang --multimodal-encode-worker --model-path "$MODEL_NAME" --chat-template "$CHAT_TEMPLATE" &

# run SGLang multimodal prefill worker
# TODO: Remove disable-radix-cache once the issue is fixed.
# See https://github.com/sgl-project/sglang/pull/11203.
OTEL_SERVICE_NAME=dynamo-multimodal-worker-prefill CUDA_VISIBLE_DEVICES=1 python3 -m dynamo.sglang \
  --multimodal-worker \
  --model-path "$MODEL_NAME" \
  --page-size 16 \
  --tp 1 \
  --trust-remote-code \
  --skip-tokenizer-init \
  --disaggregation-mode prefill \
  --disaggregation-bootstrap-port 12345 \
  --host 0.0.0.0 \
  --disable-radix-cache \
  --disaggregation-transfer-backend nixl &

# run SGLang multimodal decode worker
OTEL_SERVICE_NAME=dynamo-multimodal-worker-decode CUDA_VISIBLE_DEVICES=2 python3 -m dynamo.sglang \
  --multimodal-worker \
  --model-path "$MODEL_NAME" \
  --page-size 16 \
  --tp 1 \
  --trust-remote-code \
  --skip-tokenizer-init \
  --disaggregation-mode decode \
  --disaggregation-bootstrap-port 12345 \
  --host 0.0.0.0 \
  --disaggregation-transfer-backend nixl &

# Wait for all background processes to complete
wait