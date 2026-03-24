#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Multimodal E/P/D on a SINGLE GPU with OTEL tracing.
# Encode (vision encoder), prefill, and decode workers share one GPU.
# KV cache fractions are set in the engine YAML configs (not overridden here).

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

# Default model — Qwen3-VL-2B is small enough for 3 workers on one GPU
MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"

# Parse command line arguments
ENABLE_OTEL=true
while [[ $# -gt 0 ]]; do
    case $1 in
        --disable-otel)
            ENABLE_OTEL=false
            shift
            ;;
        --model)
            MODEL=$2
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model <model>      Specify the model to use (default: $MODEL)"
            echo "  --disable-otel       Disable OpenTelemetry tracing (enabled by default)"
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

# Environment variables with defaults
# KV cache fractions are set in the engine YAML configs (0.10 for P/D).
# Do NOT override free_gpu_memory_fraction here — the YAML values are
# tuned for 3 workers sharing a single GPU.
export DYNAMO_HOME=${DYNAMO_HOME:-"/workspace"}
export ENCODE_ENGINE_ARGS=${ENCODE_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3-vl-2b-instruct/encode.yaml"}
export PREFILL_ENGINE_ARGS=${PREFILL_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3-vl-2b-instruct/prefill.yaml"}
export DECODE_ENGINE_ARGS=${DECODE_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3-vl-2b-instruct/decode.yaml"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}
export ENCODE_ENDPOINT=${ENCODE_ENDPOINT:-"dyn://dynamo.tensorrt_llm_encode.generate"}
export MODALITY=${MODALITY:-"multimodal"}
export ALLOWED_LOCAL_MEDIA_PATH=${ALLOWED_LOCAL_MEDIA_PATH:-"/tmp"}
export MAX_FILE_SIZE_MB=${MAX_FILE_SIZE_MB:-50}

# Prevent port collisions: the test framework exports DYN_SYSTEM_PORT which all
# child processes would inherit. Unset it so only workers that need it set their own.
unset DYN_SYSTEM_PORT

# Build --override-engine-args JSON for OTEL tracing (P/D workers only).
OVERRIDE_ARGS=()
if [ "$ENABLE_OTEL" = true ]; then
    export DYN_LOGGING_JSONL=true
    export OTEL_EXPORT_ENABLED=1
    export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT:-http://localhost:4317}
    OVERRIDE_ARGS=(--override-engine-args "{\"return_perf_metrics\": true, \"otlp_traces_endpoint\": \"${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT}\"}")
fi

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner --multimodal "Launching Multimodal E/P/D on Same GPU (1 GPU)" "$MODEL" "$HTTP_PORT" \
    "Workers:     3 (encode + prefill + decode)"

# run frontend
OTEL_SERVICE_NAME=dynamo-frontend \
python3 -m dynamo.frontend &

# run encode worker (vision encoder only, no KV cache)
echo "Starting encode worker on GPU $CUDA_VISIBLE_DEVICES..."
OTEL_SERVICE_NAME=dynamo-worker-trtllm-encode \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
python3 -m dynamo.trtllm \
  --model-path "$MODEL" \
  --served-model-name "$MODEL" \
  --extra-engine-args "$ENCODE_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
  --max-file-size-mb "$MAX_FILE_SIZE_MB" \
  --disaggregation-mode encode &

# run prefill worker (shares GPU with encode and decode)
echo "Starting prefill worker on GPU $CUDA_VISIBLE_DEVICES..."
OTEL_SERVICE_NAME=dynamo-worker-trtllm-prefill \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
python3 -m dynamo.trtllm \
  --model-path "$MODEL" \
  --served-model-name "$MODEL" \
  --extra-engine-args "$PREFILL_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --publish-events-and-metrics \
  --disaggregation-mode prefill \
  --encode-endpoint "$ENCODE_ENDPOINT" \
  "${OVERRIDE_ARGS[@]}" &

# run decode worker (shares GPU with encode and prefill)
echo "Starting decode worker on GPU $CUDA_VISIBLE_DEVICES..."
OTEL_SERVICE_NAME=dynamo-worker-trtllm-decode \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT3:-8083} \
python3 -m dynamo.trtllm \
  --model-path "$MODEL" \
  --served-model-name "$MODEL" \
  --extra-engine-args "$DECODE_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --publish-events-and-metrics \
  --disaggregation-mode decode \
  "${OVERRIDE_ARGS[@]}" &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
