#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Two aggregated workers behind a KV-aware router.
# GPUs: 2

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

# Parse command line arguments
ENABLE_OTEL=false
APPROX_MODE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --enable-otel)
            ENABLE_OTEL=true
            shift
            ;;
        --approx)
            APPROX_MODE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --enable-otel        Enable OpenTelemetry tracing"
            echo "  --approx             Enable approximate KV routing (no KV events)"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Note: System metrics are enabled by default on ports 8081 (worker-1), 8082 (worker-2)"
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

MODEL="Qwen/Qwen3-0.6B"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Aggregated Router (2 workers)" "$MODEL" "$HTTP_PORT"

# run ingress
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
FRONTEND_ARGS=(--router-mode kv)
if [ "$APPROX_MODE" = true ]; then
    FRONTEND_ARGS+=(--no-kv-events)
fi
OTEL_SERVICE_NAME=dynamo-frontend \
python3 -m dynamo.frontend "${FRONTEND_ARGS[@]}" &

# run worker
# Build KV events args conditionally (only when not in approx mode)
KV_EVENTS_ARGS_1=()
KV_EVENTS_ARGS_2=()
if [ "$APPROX_MODE" = false ]; then
    KV_EVENTS_ARGS_1=(--kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}')
    KV_EVENTS_ARGS_2=(--kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5558"}')
fi

OTEL_SERVICE_NAME=dynamo-worker-1 DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT_WORKER1:-8081} \
python3 -m dynamo.sglang \
  --model-path Qwen/Qwen3-0.6B \
  --served-model-name Qwen/Qwen3-0.6B \
  --page-size 16 \
  --tp 1 \
  --trust-remote-code \
  "${KV_EVENTS_ARGS_1[@]}" \
  --enable-metrics \
  "${TRACE_ARGS[@]}" &

OTEL_SERVICE_NAME=dynamo-worker-2 DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT_WORKER2:-8082} \
CUDA_VISIBLE_DEVICES=1 python3 -m dynamo.sglang \
  --model-path Qwen/Qwen3-0.6B \
  --served-model-name Qwen/Qwen3-0.6B \
  --page-size 16 \
  --tp 1 \
  --trust-remote-code \
  "${KV_EVENTS_ARGS_2[@]}" \
  --enable-metrics \
  "${TRACE_ARGS[@]}" &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
