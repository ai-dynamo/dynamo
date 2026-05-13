#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Test-aggregated launch with TWO workers behind a KV-aware router and
# every 0.5.11 observability / KV-feature flag turned on, so a load test
# against this lights up the full Grafana dashboard (Dynamo + SGLang Engine).
#
# Same shape as agg_router.sh (2 workers, KV routing, KV events) but with:
#   - --enable-hierarchical-cache              -> HiCache row populates
#   - --enable-streaming-session               -> streaming-session held tokens
#   - --enable-metrics-for-all-schedulers      -> per-scheduler metrics
#   - --enable-mfu-metrics                     -> model FLOPs utilization
#   - --mem-fraction-static 0.92               -> larger KV pool per worker
#   - --max-running-requests 256               -> larger engine batch ceiling
#   - --page-size 32 / --chunked-prefill-size 8192
#   - cuda graph piecewise compile re-enabled  -> realistic perf
#   - OTEL trace export on by default
#
# NOTE on per-pool-type gauges (full / SWA / Mamba):
#   sglang only populates swa_token_usage and mamba_usage on hybrid attention
#   models. For Qwen/Qwen3-0.6B (default) only full_token_usage moves.
#   To exercise SWA: --model-path google/gemma-2-2b-it
#   To exercise Mamba: --model-path ai21labs/Jamba-tiny-dev (or similar)
#
# GPUs: 2

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

# Defaults
MODEL="Qwen/Qwen3-0.6B"
APPROX_MODE=false
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
        --approx)
            APPROX_MODE=true; shift ;;
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

Test-aggregated launch (2 workers + KV router) with all observability features.

Options:
  --model-path <name>             Model (default: $MODEL)
  --approx                        Use approximate KV routing (no KV events)
  --mem-fraction-static <float>   KV pool fraction of GPU mem per worker (default: $MEM_FRACTION)
  --max-running-requests <int>    Engine batch ceiling per worker (default: $MAX_RUNNING)
  --page-size <int>               KV page size in tokens (default: $PAGE_SIZE)
  --chunked-prefill-size <int>    Chunked prefill chunk size (default: $CHUNKED_PREFILL)
  --hicache-ratio <float>         HiCache host:device size ratio (default: $HICACHE_RATIO)
  -h, --help                      Show this help

Anything not matched is forwarded to BOTH SGLang workers.
Worker metrics on \$DYN_SYSTEM_PORT_WORKER1 (default 8081) and
\$DYN_SYSTEM_PORT_WORKER2 (default 8082). Frontend on \$DYN_HTTP_PORT (default 8000).
EOF
            exit 0 ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Tracing always on for this script
export DYN_LOGGING_JSONL=true
export OTEL_EXPORT_ENABLED=1
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT:-http://localhost:4317}
TRACE_ARGS=(--enable-trace --otlp-traces-endpoint localhost:4317)

GPU_MEM_ARGS=$(build_sglang_gpu_mem_args)

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Test-Aggregated + KV Router (2 GPUs, full observability)" "$MODEL" "$HTTP_PORT"

cat <<EOF

Topology:
  - 2 SGLang workers (1 GPU each, tp=1) behind dynamo.frontend KV router
  - frontend: --router-mode kv$([ "$APPROX_MODE" = true ] && echo " --no-kv-events" || echo "  (KV events ON)")
  - worker metrics: \$DYN_SYSTEM_PORT_WORKER1=\${DYN_SYSTEM_PORT_WORKER1:-8081}, \$DYN_SYSTEM_PORT_WORKER2=\${DYN_SYSTEM_PORT_WORKER2:-8082}

Features enabled for full Grafana coverage:
  - hierarchical KV cache (host RAM tier)        -> HiCache row
  - streaming session retention                  -> streaming-session held tokens
  - per-scheduler metrics + MFU metrics
  - mem-fraction-static=$MEM_FRACTION, max-running-requests=$MAX_RUNNING
  - page-size=$PAGE_SIZE, chunked-prefill-size=$CHUNKED_PREFILL
  - cuda graph piecewise compile enabled
  - OTLP trace export to \$OTEL_EXPORTER_OTLP_TRACES_ENDPOINT

Dashboard: http://localhost:3000/d/sglang-engine
EOF

# Frontend with KV-aware routing
FRONTEND_ARGS=(--router-mode kv)
if [ "$APPROX_MODE" = true ]; then
    FRONTEND_ARGS+=(--no-kv-events)
fi
OTEL_SERVICE_NAME=dynamo-frontend \
python3 -m dynamo.frontend "${FRONTEND_ARGS[@]}" &

# Per-worker KV events config (only when not in approx mode)
KV_EVENTS_ARGS_1=()
KV_EVENTS_ARGS_2=()
if [ "$APPROX_MODE" = false ]; then
    KV_EVENTS_ARGS_1=(--kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}')
    KV_EVENTS_ARGS_2=(--kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5558"}')
fi

# Common worker flags
WORKER_COMMON_ARGS=(
  --model-path "$MODEL"
  --served-model-name "$MODEL"
  --tp 1
  --trust-remote-code
  --enable-metrics
  --enable-metrics-for-all-schedulers
  --enable-mfu-metrics
  --enable-hierarchical-cache
  --hicache-ratio "$HICACHE_RATIO"
  --enable-streaming-session
  --mem-fraction-static "$MEM_FRACTION"
  --max-running-requests "$MAX_RUNNING"
  --page-size "$PAGE_SIZE"
  --chunked-prefill-size "$CHUNKED_PREFILL"
  --log-requests
  --log-requests-level 1
)

# Worker 1 on GPU 0
OTEL_SERVICE_NAME=dynamo-worker-1 DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT_WORKER1:-8081} \
CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.sglang \
  "${WORKER_COMMON_ARGS[@]}" \
  "${KV_EVENTS_ARGS_1[@]}" \
  $GPU_MEM_ARGS \
  "${TRACE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" &

# Worker 2 on GPU 1
OTEL_SERVICE_NAME=dynamo-worker-2 DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT_WORKER2:-8082} \
CUDA_VISIBLE_DEVICES=1 python3 -m dynamo.sglang \
  "${WORKER_COMMON_ARGS[@]}" \
  "${KV_EVENTS_ARGS_2[@]}" \
  $GPU_MEM_ARGS \
  "${TRACE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" &

wait_any_exit
