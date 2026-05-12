#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

# Environment variables with defaults
export DYNAMO_HOME=${DYNAMO_HOME:-"/workspace"}
export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-0.6B"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"Qwen/Qwen3-0.6B"}
export PREFILL_ENGINE_ARGS=${PREFILL_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3/prefill.yaml"}
export DECODE_ENGINE_ARGS=${DECODE_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3/decode.yaml"}
export PREFILL_CUDA_VISIBLE_DEVICES=${PREFILL_CUDA_VISIBLE_DEVICES:-"0"}
export DECODE_CUDA_VISIBLE_DEVICES=${DECODE_CUDA_VISIBLE_DEVICES:-"1"}
export MODALITY=${MODALITY:-"text"}
# If you want to use multimodal, set MODALITY to "multimodal"
#export MODALITY=${MODALITY:-"multimodal"}

# Strip --unified via the shared helper, then parse the remaining flags.
# All of this runs BEFORE installing the kill-process-group EXIT trap so
# an early exit (--help / unknown option) doesn't tear down the caller.
pick_worker_module dynamo.trtllm dynamo.trtllm.unified_main "$@"
set -- "${REMAINING_ARGS[@]}"

ENABLE_OTEL=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --enable-otel) ENABLE_OTEL=true; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --enable-otel        Enable OpenTelemetry tracing"
            echo "  --unified            Use the unified backend entry point"
            echo "                       (python -m dynamo.trtllm.unified_main)"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *) echo "Unknown option: $1"; echo "Use --help for usage information"; exit 1 ;;
    esac
done

trap 'echo Cleaning up...; kill 0' EXIT

# Enable tracing if requested
TRACE_ARGS=()
if [ "$ENABLE_OTEL" = true ]; then
    export DYN_LOGGING_JSONL=true
    export OTEL_EXPORT_ENABLED=1
    export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT:-http://localhost:4317}
    TRACE_ARGS+=(--override-engine-args "{\"return_perf_metrics\": true, \"otlp_traces_endpoint\": \"${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT}\" }")
fi

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Disaggregated Serving (2 GPUs)" "$MODEL_PATH" "$HTTP_PORT"

# run frontend
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
OTEL_SERVICE_NAME=dynamo-frontend \
python3 -m dynamo.frontend &

# run prefill worker
OTEL_SERVICE_NAME=dynamo-worker-prefill CUDA_VISIBLE_DEVICES=$PREFILL_CUDA_VISIBLE_DEVICES python3 -m "$WORKER_MODULE" \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args  "$PREFILL_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --disaggregation-mode prefill \
  "${TRACE_ARGS[@]}" &

# run decode worker
OTEL_SERVICE_NAME=dynamo-worker-decode CUDA_VISIBLE_DEVICES=$DECODE_CUDA_VISIBLE_DEVICES python3 -m "$WORKER_MODULE" \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args  "$DECODE_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --disaggregation-mode decode \
  "${TRACE_ARGS[@]}" &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
