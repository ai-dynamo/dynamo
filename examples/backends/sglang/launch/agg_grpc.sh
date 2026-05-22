#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated serving via the Rust bridge to SGLang's native gRPC server.
# Three independent processes: dynamo.frontend, sglang.launch_server, and
# dynamo.sglang_grpc (the bridge worker). Mirrors agg.sh's CLI surface.
# GPUs: 1

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

# Default values
MODEL="Qwen/Qwen3-0.6B"
ENABLE_OTEL=false
SGLANG_GRPC_PORT="${SGLANG_GRPC_PORT:-40000}"
SGLANG_HTTP_PORT="${SGLANG_HTTP_PORT:-30000}"

# Parse command line arguments
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL="$2"
            shift 2
            ;;
        --enable-otel)
            ENABLE_OTEL=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model-path <name>  Specify model (default: $MODEL)"
            echo "  --enable-otel        Enable OpenTelemetry tracing"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Additional SGLang flags can be passed and will be forwarded to sglang.launch_server"
            echo "Note: System metrics are enabled by default on port 8081 (worker)"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Enable tracing if requested
if [ "$ENABLE_OTEL" = true ]; then
    export DYN_LOGGING_JSONL=true
    export OTEL_EXPORT_ENABLED=1
    export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT:-http://localhost:4317}
fi

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Aggregated Serving (gRPC bridge)" "$MODEL" "$HTTP_PORT"

# dynamo.frontend's preprocessor needs a local model dir for the tokenizer.
FRONTEND_MODEL_PATH="$(resolve_local_model_dir "$MODEL")"

# run ingress
OTEL_SERVICE_NAME=dynamo-frontend \
python3 -m dynamo.frontend \
    --model-name "$MODEL" \
    --model-path "$FRONTEND_MODEL_PATH" &

# run sglang with the native gRPC server, tokenizer disabled (dynamo.frontend
# is the sole tokenizer/detokenizer; the wire format carries token IDs).
SGLANG_ENABLE_GRPC=1 \
python3 -m sglang.launch_server \
    --enable-grpc \
    --grpc-port "$SGLANG_GRPC_PORT" \
    --port "$SGLANG_HTTP_PORT" \
    --model-path "$MODEL" \
    --tp 1 \
    --trust-remote-code \
    --skip-tokenizer-init \
    --enable-metrics \
    "${EXTRA_ARGS[@]}" &

echo "Waiting for SGLang gRPC (:$SGLANG_GRPC_PORT)..."
if ! wait_for_port "$SGLANG_GRPC_PORT" 600; then
    echo "ERROR: SGLang gRPC port :$SGLANG_GRPC_PORT did not open within 600s" >&2
    exit 1
fi

# run bridge worker
OTEL_SERVICE_NAME=dynamo-worker DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
python3 -m dynamo.sglang_grpc \
    --sglang-grpc-endpoint "http://127.0.0.1:$SGLANG_GRPC_PORT" &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
