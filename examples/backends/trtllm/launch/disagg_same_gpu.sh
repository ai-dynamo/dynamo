#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Disaggregated prefill/decode on a SINGLE GPU.
# Per-worker VRAM is controlled via env vars (MAX_SEQ_LEN, MAX_CONCURRENT_SEQS).
# TODO: unify with build_gpu_mem_args once trtllm --override-engine-args JSON
# merging is supported.
#
# NOTE — trtllm fraction semantics differ from vllm/sglang:
#   vllm/sglang:  fraction of TOTAL VRAM  (weights + KV + activations all inside)
#   trtllm:       fraction of FREE  VRAM  (KV cache only, after model load)
# build_gpu_mem_args handles this — see gpu_utils.sh / gpu_utils.md.
#
# Measured reference (Qwen/Qwen3-0.6B, --max-seq-len 4096, RTX 6000 Ada 48 GiB):
#   estimate (from gpu_utils.sh) : ~8.0 GiB per worker (~16.0 GiB total)
#   actual (nvidia-smi)          : ~7.4 GiB per worker (~14.8 GiB total)
#   fraction per worker (free)   : 0.05
#   Overestimating is intentional -- better to pad than OOM.

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"

MODEL="Qwen/Qwen3-0.6B"

# ---- Tunable (override via env vars) ----
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"

# TODO: unify with build_gpu_mem_args once trtllm --override-engine-args JSON
# merging is supported.
GPU_MEM_FRACTION="${GPU_MEM_FRACTION:-}"

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
# Always override free_gpu_memory_fraction so the script controls KV cache size,
# matching how vllm (--gpu-memory-utilization) and sglang (--mem-fraction-static)
# pass memory parameters from the launch script.
OVERRIDE_PAIRS=""
if [[ -n "$GPU_MEM_FRACTION" ]]; then
    OVERRIDE_PAIRS="\"kv_cache_config\": {\"free_gpu_memory_fraction\": ${GPU_MEM_FRACTION}}"
fi
if [ "$ENABLE_OTEL" = true ]; then
    export DYN_LOGGING_JSONL=true
    export OTEL_EXPORT_ENABLED=1
    export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT:-http://localhost:4317}
    OVERRIDE_PAIRS="${OVERRIDE_PAIRS}, \"return_perf_metrics\": true, \"otlp_traces_endpoint\": \"${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT}\""
fi
OVERRIDE_ARGS=(--override-engine-args "{${OVERRIDE_PAIRS}}")

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Disaggregated on Same GPU (1 GPU)" "$MODEL" "$HTTP_PORT" \
    "Workers:     2 (prefill + decode, fraction is per worker)"

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

# run decode worker (shares GPU with prefill)
OTEL_SERVICE_NAME=dynamo-worker-decode \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
python3 -m dynamo.trtllm \
  --model-path "$MODEL" \
  --served-model-name "$MODEL" \
  --extra-engine-args  "$DECODE_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --publish-events-and-metrics \
  --disaggregation-mode decode \
  "${OVERRIDE_ARGS[@]}" &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
