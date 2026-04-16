#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Disaggregated prefill/decode on a SINGLE GPU.
# Per-worker VRAM is controlled via absolute KV token caps (not fractions).
# Profiler overrides (_PROFILE_OVERRIDE_TRTLLM_MAX_TOTAL_TOKENS) are handled via
# build_trtllm_override_args_with_mem; standalone runs use MAX_TOTAL_TOKENS.
#
# Measured reference (Qwen/Qwen3-0.6B, RTX 6000 Ada 48 GiB):
#   peak VRAM (nvidia-smi)     : ~6.6 GiB total (both workers)
#   default MAX_TOTAL_TOKENS   : 25000 per worker
#   min tokens (profiled)      : 256 per worker

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"

MODEL="Qwen/Qwen3-0.6B"

# ---- Tunable (override via env vars) ----
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-25000}"

# Environment variables with defaults
export DYNAMO_HOME=${DYNAMO_HOME:-"/workspace"}
export PREFILL_ENGINE_ARGS=${PREFILL_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3/prefill.yaml"}
export DECODE_ENGINE_ARGS=${DECODE_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3/decode.yaml"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}
export MODALITY=${MODALITY:-"text"}

source "$SCRIPT_DIR/../../../common/launch_utils.sh"

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

# Build --override-engine-args JSON.
#
# KV cache control (always absolute caps, never fractions):
#   1. Profiler env var (_PROFILE_OVERRIDE_TRTLLM_MAX_TOTAL_TOKENS or
#      _PROFILE_OVERRIDE_TRTLLM_MAX_GPU_TOTAL_BYTES) via build_trtllm_override_args_with_mem.
#   2. MAX_TOTAL_TOKENS env var (default 25000) for standalone runs.

# Collect non-memory override pairs (otel, etc.)
NON_MEM_PAIRS=""
if [ "$ENABLE_OTEL" = true ]; then
    export DYN_LOGGING_JSONL=true
    export OTEL_EXPORT_ENABLED=1
    export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT:-http://localhost:4317}
    NON_MEM_PAIRS="\"return_perf_metrics\": true, \"otlp_traces_endpoint\": \"${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT}\""
fi

if [[ -n "${_PROFILE_OVERRIDE_TRTLLM_MAX_TOTAL_TOKENS:-}" ]] || [[ -n "${_PROFILE_OVERRIDE_TRTLLM_MAX_GPU_TOTAL_BYTES:-}" ]]; then
    # Profiler provides absolute cap
    BASE_JSON=""
    [[ -n "$NON_MEM_PAIRS" ]] && BASE_JSON="{${NON_MEM_PAIRS}}"
    FINAL_JSON=$(build_trtllm_override_args_with_mem ${BASE_JSON:+--merge-with-json "$BASE_JSON"})
    OVERRIDE_ARGS=(--override-engine-args "$FINAL_JSON")
else
    # No profiler — use absolute token cap from MAX_TOTAL_TOKENS
    OVERRIDE_PAIRS="\"kv_cache_config\": {\"max_tokens\": ${MAX_TOTAL_TOKENS}}"
    if [[ -n "$NON_MEM_PAIRS" ]]; then
        OVERRIDE_PAIRS="${OVERRIDE_PAIRS}, $NON_MEM_PAIRS"
    fi
    OVERRIDE_ARGS=(--override-engine-args "{${OVERRIDE_PAIRS}}")
fi

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Disaggregated on Same GPU with Decode KVBM (1 GPU)" "$MODEL" "$HTTP_PORT" \
    "Workers:     2 (prefill + decode, fraction is per worker)" \
    "Decode KVBM: enabled"

# run frontend
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
OTEL_SERVICE_NAME=dynamo-frontend \
python3 -m dynamo.frontend &

# run prefill worker (shares GPU with decode)
OTEL_SERVICE_NAME=dynamo-worker-prefill \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
python3 -m dynamo.trtllm \
  --model-path "$MODEL" \
  --served-model-name "$MODEL" \
  --extra-engine-args  "$PREFILL_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --publish-events-and-metrics \
  --disaggregation-mode prefill \
  "${OVERRIDE_ARGS[@]}" &

# Wait for prefill worker to load model and allocate KV cache before starting
# decode.  Both workers share one GPU; without this wait they compete for GPU
# memory during model loading, which can cause OOM.
# || true: don't let set -e kill the script on timeout (wait_for_ready returns 1).
PREFILL_SYSTEM_PORT="${DYN_SYSTEM_PORT1:-8081}"
wait_for_ready "http://localhost:${PREFILL_SYSTEM_PORT}/health" 45 || true

# run decode worker (shares GPU with prefill)
OTEL_SERVICE_NAME=dynamo-worker-decode \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS=${DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS:-512} \
DYN_KVBM_KV_EVENTS_ENABLE_CONSOLIDATOR=${DYN_KVBM_KV_EVENTS_ENABLE_CONSOLIDATOR:-false} \
DYN_KVBM_METRICS=${DYN_KVBM_METRICS:-true} \
DYN_KVBM_METRICS_PORT=${DYN_KVBM_METRICS_PORT:-16880} \
python3 -m dynamo.trtllm \
  --model-path "$MODEL" \
  --served-model-name "$MODEL" \
  --extra-engine-args  "$DECODE_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --publish-events-and-metrics \
  --disaggregation-mode decode \
  --connector kvbm \
  "${OVERRIDE_ARGS[@]}" &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
