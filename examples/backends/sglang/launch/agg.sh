#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated serving: single worker handles both prefill and decode.
# GPUs: 1

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"   # build_gpu_mem_args
source "$SCRIPT_DIR/../../../common/launch_utils.sh" # print_launch_banner, wait_any_exit

# Default values
MODEL="zai-org/GLM-4.7-Flash"
ENABLE_OTEL=false
CONTEXT_LENGTH="${CONTEXT_LENGTH:-131072}"

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
            echo "Additional SGLang/Dynamo flags can be passed and will be forwarded"
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
TRACE_ARGS=()
if [ "$ENABLE_OTEL" = true ]; then
    export DYN_LOGGING_JSONL=true
    export OTEL_EXPORT_ENABLED=1
    export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT:-http://localhost:4317}
    TRACE_ARGS+=(--enable-trace --otlp-traces-endpoint localhost:4317)
fi

GPU_MEM_FRACTION=$(build_gpu_mem_args sglang --model "$MODEL" --max-model-len "$CONTEXT_LENGTH")

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Aggregated Serving" "$MODEL" "$HTTP_PORT"

# run ingress
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
OTEL_SERVICE_NAME=dynamo-frontend \
python3 -m dynamo.frontend --router-mode kv --enable-cache-control &


# run worker with metrics enabled
OTEL_SERVICE_NAME=dynamo-worker DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
python3 -m dynamo.sglang \
  --model-path "$MODEL" \
  --served-model-name "$MODEL" \
  --page-size 16 \
  --tp 2 --ep 2 --enable-dp-attention --dp 2 --disable-piecewise-cuda-graph \
  --dyn-tool-call-parser glm47  --dyn-reasoning-parser glm45 --enable-streaming-session \
  --trust-remote-code --moe-dense-tp-size 1 \
  --skip-tokenizer-init \
  --enable-metrics \
  "${TRACE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
