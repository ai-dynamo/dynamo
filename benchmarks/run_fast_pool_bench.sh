#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run the fast-pool rollout acceleration benchmark.
#
# Compares two modes:
#   1. Baseline: all workers at the same speed
#   2. Fast-pool: 75% normal workers + 25% fast workers
#
# Prerequisites:
#   - etcd running on localhost:2379
#   - NATS running on localhost:4222
#   - Model available at MODEL_PATH
#
# Usage:
#   ./benchmarks/run_fast_pool_bench.sh --model-path /data/models/Qwen3-0.6B
#   ./benchmarks/run_fast_pool_bench.sh --model-path Qwen/Qwen3-0.6B --mode fast-pool
#   ./benchmarks/run_fast_pool_bench.sh --model-path Qwen/Qwen3-0.6B --mode both

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
MODEL_PATH=""
MODE="both"           # baseline | fast-pool | both
NORMAL_WORKERS=6
FAST_WORKERS=2
NORMAL_SPEEDUP=10.0
FAST_SPEEDUP=16.0     # 1.6x of normal
NORMAL_REQUESTS=40
LONGTAIL_REQUESTS=10
NORMAL_TURNS=3
LONGTAIL_TURNS=10
NORMAL_OSL_MEAN=128
NORMAL_OSL_STDDEV=32
LONGTAIL_OSL_MEAN=512
LONGTAIL_OSL_STDDEV=128
FRONTEND_PORT=8321

usage() {
    cat <<EOF
Usage: $0 --model-path <path> [options]

Options:
  --model-path PATH       Model path (required)
  --mode MODE             baseline | fast-pool | both (default: both)
  --normal-workers N      Normal pool workers (default: $NORMAL_WORKERS)
  --fast-workers N        Fast pool workers (default: $FAST_WORKERS)
  --normal-speedup X      Base speedup ratio (default: $NORMAL_SPEEDUP)
  --fast-speedup X        Fast pool speedup ratio (default: $FAST_SPEEDUP)
  --normal-requests N     Normal request count (default: $NORMAL_REQUESTS)
  --longtail-requests N   Long-tail request count (default: $LONGTAIL_REQUESTS)
  --normal-turns N        Turns per normal request (default: $NORMAL_TURNS)
  --longtail-turns N      Turns per long-tail request (default: $LONGTAIL_TURNS)
  --normal-osl-mean N     Mean OSL per turn for normal (default: $NORMAL_OSL_MEAN)
  --normal-osl-stddev N   Stddev OSL per turn for normal (default: $NORMAL_OSL_STDDEV)
  --longtail-osl-mean N   Mean OSL per turn for long-tail (default: $LONGTAIL_OSL_MEAN)
  --longtail-osl-stddev N Stddev OSL per turn for long-tail (default: $LONGTAIL_OSL_STDDEV)
  --frontend-port N       Frontend port (default: $FRONTEND_PORT)
  -h, --help              Show this help
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path)         MODEL_PATH="$2"; shift 2 ;;
        --mode)               MODE="$2"; shift 2 ;;
        --normal-workers)     NORMAL_WORKERS="$2"; shift 2 ;;
        --fast-workers)       FAST_WORKERS="$2"; shift 2 ;;
        --normal-speedup)     NORMAL_SPEEDUP="$2"; shift 2 ;;
        --fast-speedup)       FAST_SPEEDUP="$2"; shift 2 ;;
        --normal-requests)    NORMAL_REQUESTS="$2"; shift 2 ;;
        --longtail-requests)  LONGTAIL_REQUESTS="$2"; shift 2 ;;
        --normal-turns)       NORMAL_TURNS="$2"; shift 2 ;;
        --longtail-turns)     LONGTAIL_TURNS="$2"; shift 2 ;;
        --normal-osl-mean)    NORMAL_OSL_MEAN="$2"; shift 2 ;;
        --normal-osl-stddev)  NORMAL_OSL_STDDEV="$2"; shift 2 ;;
        --longtail-osl-mean)  LONGTAIL_OSL_MEAN="$2"; shift 2 ;;
        --longtail-osl-stddev) LONGTAIL_OSL_STDDEV="$2"; shift 2 ;;
        --frontend-port)      FRONTEND_PORT="$2"; shift 2 ;;
        -h|--help)            usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [[ -z "$MODEL_PATH" ]]; then
    echo "ERROR: --model-path is required"
    usage
fi

TOTAL_WORKERS=$(( NORMAL_WORKERS + FAST_WORKERS ))

COMMON_ARGS=(
    --model-path "$MODEL_PATH"
    --normal-requests "$NORMAL_REQUESTS"
    --longtail-requests "$LONGTAIL_REQUESTS"
    --normal-turns "$NORMAL_TURNS"
    --longtail-turns "$LONGTAIL_TURNS"
    --normal-osl-mean "$NORMAL_OSL_MEAN"
    --normal-osl-stddev "$NORMAL_OSL_STDDEV"
    --longtail-osl-mean "$LONGTAIL_OSL_MEAN"
    --longtail-osl-stddev "$LONGTAIL_OSL_STDDEV"
    --frontend-port "$FRONTEND_PORT"
)

run_baseline() {
    echo ""
    echo "================================================================"
    echo "  BASELINE: ${TOTAL_WORKERS} workers at speedup=$NORMAL_SPEEDUP"
    echo "  All requests (normal + long-tail) round-robin across all workers."
    echo "================================================================"
    echo ""
    python3 "$SCRIPT_DIR/fast_pool_bench.py" \
        "${COMMON_ARGS[@]}" \
        --normal-workers "$TOTAL_WORKERS" \
        --fast-workers 0 \
        --normal-speedup "$NORMAL_SPEEDUP" \
        --fast-speedup "$NORMAL_SPEEDUP"
    # Save results
    mv -f fast_pool_bench_results.json fast_pool_bench_baseline.json 2>/dev/null || true
    echo "Baseline results saved to fast_pool_bench_baseline.json"
}

run_fast_pool() {
    echo ""
    echo "================================================================"
    echo "  FAST-POOL: ${NORMAL_WORKERS} normal at speedup=$NORMAL_SPEEDUP,"
    echo "             ${FAST_WORKERS} fast at speedup=$FAST_SPEEDUP"
    echo "================================================================"
    echo ""
    python3 "$SCRIPT_DIR/fast_pool_bench.py" \
        "${COMMON_ARGS[@]}" \
        --normal-workers "$NORMAL_WORKERS" \
        --fast-workers "$FAST_WORKERS" \
        --normal-speedup "$NORMAL_SPEEDUP" \
        --fast-speedup "$FAST_SPEEDUP"
    # Save results
    mv -f fast_pool_bench_results.json fast_pool_bench_fastpool.json 2>/dev/null || true
    echo "Fast-pool results saved to fast_pool_bench_fastpool.json"
}

case "$MODE" in
    baseline)
        run_baseline
        ;;
    fast-pool)
        run_fast_pool
        ;;
    both)
        run_baseline
        echo ""
        echo "Pausing 5s between runs..."
        sleep 5
        run_fast_pool
        echo ""
        echo "================================================================"
        echo "  Compare: fast_pool_bench_baseline.json"
        echo "       vs: fast_pool_bench_fastpool.json"
        echo "================================================================"
        # Quick summary comparison
        if command -v python3 &>/dev/null; then
            python3 -c "
import json, sys
try:
    with open('fast_pool_bench_baseline.json') as f:
        bl = json.load(f)
    with open('fast_pool_bench_fastpool.json') as f:
        fp = json.load(f)
    bl_ms = bl['batch_makespan_s']
    fp_ms = fp['batch_makespan_s']
    speedup = bl_ms / fp_ms if fp_ms > 0 else float('inf')
    print(f'')
    print(f'Baseline makespan:   {bl_ms:.3f}s')
    print(f'Fast-pool makespan:  {fp_ms:.3f}s')
    print(f'Speedup:             {speedup:.2f}x')
except Exception as e:
    print(f'Could not compare: {e}', file=sys.stderr)
"
        fi
        ;;
    *)
        echo "Unknown mode: $MODE (expected: baseline, fast-pool, both)"
        exit 1
        ;;
esac
