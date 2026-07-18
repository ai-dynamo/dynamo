#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

PHASE="${1:-all}"
RUN_ROOT="${OUTPUT_DIR:-/dynamo-tmp/logs/$(date -u +%m-%d)/qwen25-safeguard-proxy}"
WORKLOAD_DIR="${WORKLOAD_DIR:-$RUN_ROOT/workload}"
MEASURED_DIR="$RUN_ROOT/measured"
SMOKE_DIR="$RUN_ROOT/smoke"
IMAGE_SIZE="${DYN_QWEN2_VL_BENCHMARK_IMAGE_SIZE:-500}"
UNIQUE_IMAGES="${DYN_QWEN2_VL_BENCHMARK_UNIQUE_IMAGES:-9}"
GRAPH_IMAGE_SIZE="${IMAGE_SIZE}x${IMAGE_SIZE}"

: "${DYNAMO_BENCHMARK_COMMIT:?set DYNAMO_BENCHMARK_COMMIT}"
: "${DYNAMO_BENCHMARK_BRANCH:?set DYNAMO_BENCHMARK_BRANCH}"
: "${DYNAMO_BENCHMARK_IMAGE:?set DYNAMO_BENCHMARK_IMAGE}"

export DYN_QWEN2_VL_GRAPH_BATCH_BUCKETS=1,2,3,4,5,6,7,8
export DYN_QWEN2_VL_GRAPH_IMAGE_SIZES="$GRAPH_IMAGE_SIZE"
export DYN_QWEN2_VL_PREPROCESS_CONCURRENCY=4
export DYN_QWEN2_VL_MAX_BATCH_COST=8
export DYN_QWEN2_VL_PREPROCESS_CACHE_SIZE=0

mkdir -p "$WORKLOAD_DIR" "$MEASURED_DIR" "$SMOKE_DIR"

run_workload() {
    if [[ ! -f "$WORKLOAD_DIR/workload_manifest.json" ]]; then
        python -m examples.custom_encoder.benchmark.safeguard_proxy_workload \
            generate \
            --output-dir "$WORKLOAD_DIR" \
            --image-size "$IMAGE_SIZE" \
            --unique-images "$UNIQUE_IMAGES"
    fi
    python -m examples.custom_encoder.benchmark.safeguard_proxy_workload \
        validate "$WORKLOAD_DIR" \
        --image-size "$IMAGE_SIZE" \
        --unique-images "$UNIQUE_IMAGES"
}

run_graphs() {
    python -m examples.custom_encoder.benchmark.verify_safeguard_proxy_graphs \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --encoder-model Qwen/Qwen2.5-VL-3B-Instruct \
        --output-hidden-size 1536 \
        --replay-iterations 20 \
        2>&1 | tee "$MEASURED_DIR/graph_verification.log"
}

run_smoke() {
    python -m examples.custom_encoder.benchmark.run_safeguard_proxy_sweep \
        run \
        --workload-dir "$WORKLOAD_DIR" \
        --output-dir "$SMOKE_DIR" \
        --concurrencies 1 \
        --smoke \
        2>&1 | tee "$SMOKE_DIR/smoke.log"
}

run_measurement() {
    python -m examples.custom_encoder.benchmark.run_safeguard_proxy_sweep \
        run \
        --workload-dir "$WORKLOAD_DIR" \
        --output-dir "$MEASURED_DIR" \
        --concurrencies 1 2 3 4 5 6 7 8 9 10 \
        2>&1 | tee "$MEASURED_DIR/sweep.log"
}

run_report() {
    python -m examples.custom_encoder.benchmark.run_safeguard_proxy_sweep \
        validate "$MEASURED_DIR"
    python -m examples.custom_encoder.benchmark.run_safeguard_proxy_sweep \
        summarize "$MEASURED_DIR" \
        --markdown "$MEASURED_DIR/benchmark.md" \
        --csv "$MEASURED_DIR/benchmark.csv"
}

case "$PHASE" in
    workload) run_workload ;;
    graphs) run_graphs ;;
    smoke) run_smoke ;;
    measure) run_measurement ;;
    report) run_report ;;
    all)
        run_workload
        run_graphs
        run_smoke
        run_measurement
        run_report
        ;;
    *)
        echo "usage: $0 {workload|graphs|smoke|measure|report|all}" >&2
        exit 2
        ;;
esac
