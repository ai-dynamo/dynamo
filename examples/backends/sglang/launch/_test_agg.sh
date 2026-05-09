#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Test-aggregated serving: same shape as agg.sh but with every observability
# / KV-feature flag turned on so a load test against this lights up the full
# Grafana dashboard (Dynamo + SGLang Engine).
#
# Differences from agg.sh:
#   - --enable-hierarchical-cache         (populates HiCache row)
#   - --enable-streaming-session          (populates streaming-session held tokens)
#   - --enable-metrics-for-all-schedulers (per-scheduler metric breakdown)
#   - --enable-mfu-metrics                (model FLOPs utilization)
#   - --mem-fraction-static 0.92          (was implicit ~0.85 default)
#   - --max-running-requests 256          (was unset → conservative auto-pick)
#   - --chunked-prefill-size 8192         (was 4096)
#   - --page-size 32                      (was 16)
#   - cuda graph piecewise compile re-enabled (no --disable-piecewise-cuda-graph)
#   - --enable-trace + --enable-metrics on by default
#
# NOTE on per-pool-type gauges (full / SWA / Mamba):
#   sglang only populates swa_token_usage and mamba_usage on hybrid attention
#   models. For Qwen/Qwen3-0.6B (default) only full_token_usage moves.
#   To exercise SWA: --model-path google/gemma-2-2b-it
#   To exercise Mamba: --model-path ai21labs/Jamba-tiny-dev (or similar)
#
# GPUs: 1

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

# Defaults
MODEL="Qwen/Qwen3-0.6B"
USE_UNIFIED=false
MEM_FRACTION="0.92"
MAX_RUNNING="256"
PAGE_SIZE="32"
CHUNKED_PREFILL="8192"
HICACHE_RATIO="2.0"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL="$2"; shift 2 ;;
        --unified)
            USE_UNIFIED=true; shift ;;
        --mem-fraction-static)
            MEM_FRACTION="$2"; shift 2 ;;
        --max-running-requests)
            MAX_RUNNING="$2"; shift 2 ;;
        --page-size)
            PAGE_SIZE="$2"; shift 2 ;;
        --chunked-prefill-size)
            CHUNKED_PREFILL="$2"; shift 2 ;;
        --hicache-ratio)
            HICACHE_RATIO="$2"; shift 2 ;;
        -h|--help)
            cat <<EOF
Usage: $0 [OPTIONS] [-- EXTRA_SGLANG_ARGS]

Test-aggregated launch with all observability features enabled.

Options:
  --model-path <name>             Model (default: $MODEL)
  --unified                       Use dynamo.sglang.unified_main entrypoint
  --mem-fraction-static <float>   KV pool fraction of GPU mem (default: $MEM_FRACTION)
  --max-running-requests <int>    Engine batch ceiling (default: $MAX_RUNNING)
  --page-size <int>               KV page size in tokens (default: $PAGE_SIZE)
  --chunked-prefill-size <int>    Chunked prefill chunk size (default: $CHUNKED_PREFILL)
  --hicache-ratio <float>         HiCache host:device size ratio (default: $HICACHE_RATIO)
  -h, --help                      Show this help

Anything not matched is forwarded to the SGLang worker.
Note: System metrics on \$DYN_SYSTEM_PORT (default 8081), frontend on \$DYN_HTTP_PORT (default 8000).
EOF
            exit 0 ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Tracing + JSONL logging always on for this script
export DYN_LOGGING_JSONL=true
export OTEL_EXPORT_ENABLED=1
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT:-http://localhost:4317}
TRACE_ARGS=(--enable-trace --otlp-traces-endpoint localhost:4317)

GPU_MEM_ARGS=$(build_sglang_gpu_mem_args)

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Test-Aggregated Serving (full observability)" "$MODEL" "$HTTP_PORT"

cat <<EOF

Features enabled for full Grafana coverage:
  - hierarchical KV cache (host RAM tier)        -> HiCache row
  - streaming session retention                  -> streaming session panels
  - per-scheduler metrics + MFU metrics
  - mem-fraction-static=$MEM_FRACTION, max-running-requests=$MAX_RUNNING
  - page-size=$PAGE_SIZE, chunked-prefill-size=$CHUNKED_PREFILL
  - cuda graph piecewise compile enabled
  - OTLP trace export to \$OTEL_EXPORTER_OTLP_TRACES_ENDPOINT

Dashboard: http://localhost:3000/d/sglang-engine
EOF

# Frontend
OTEL_SERVICE_NAME=dynamo-frontend \
python3 -m dynamo.frontend &

# Worker
WORKER_MODULE="dynamo.sglang"
if [ "$USE_UNIFIED" = true ]; then
    WORKER_MODULE="dynamo.sglang.unified_main"
fi
OTEL_SERVICE_NAME=dynamo-worker DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
python3 -m "$WORKER_MODULE" \
  --model-path "$MODEL" \
  --served-model-name "$MODEL" \
  --tp 1 \
  --trust-remote-code \
  --enable-metrics \
  --enable-metrics-for-all-schedulers \
  --enable-mfu-metrics \
  --enable-hierarchical-cache \
  --hicache-ratio "$HICACHE_RATIO" \
  --enable-streaming-session \
  --mem-fraction-static "$MEM_FRACTION" \
  --max-running-requests "$MAX_RUNNING" \
  --page-size "$PAGE_SIZE" \
  --chunked-prefill-size "$CHUNKED_PREFILL" \
  --log-requests \
  --log-requests-level 1 \
  $GPU_MEM_ARGS \
  "${TRACE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" &

wait_any_exit
