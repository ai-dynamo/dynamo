#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${OUTPUT_DIR:-/dynamo-tmp/logs/$(date -u +%m-%d)/qwen2-vl-1.5b-custom-concurrency}"
WORKLOAD_DIR="${WORKLOAD_DIR:-$RUN_ROOT/workload}"
MEASURED_DIR="$RUN_ROOT/measured"
SMOKE_DIR="$RUN_ROOT/smoke"
DECODER_MODEL="${DECODER_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
ENCODER_MODEL="${ENCODER_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"

mkdir -p "$MEASURED_DIR" "$SMOKE_DIR"

if [[ ! -f "$WORKLOAD_DIR/workload_manifest.json" ]]; then
    python "$SCRIPT_DIR/generate_concurrency_workload.py" \
        --output-dir "$WORKLOAD_DIR" \
        --decoder-model "$DECODER_MODEL" \
        --encoder-model "$ENCODER_MODEL" \
        --concurrencies 8 16 32 \
        --requests-per-concurrency 1000
fi
python "$SCRIPT_DIR/validate_concurrency_workload.py" "$WORKLOAD_DIR"

export DYN_QWEN2_VL_GRAPH_BATCH_BUCKETS=1,2,4,8,16,32,64
export DYN_QWEN2_VL_GRAPH_IMAGE_SIZES=500x500
export DYN_QWEN2_VL_PREPROCESS_CONCURRENCY=64
export DYN_QWEN2_VL_MAX_BATCH_COST=64

python "$SCRIPT_DIR/verify_cuda_graph.py" \
    --model "$DECODER_MODEL" \
    --encoder-model "$ENCODER_MODEL" \
    --output-hidden-size 1536 \
    --replay-iterations 20 2>&1 | tee "$MEASURED_DIR/graph_verification.log"

python "$SCRIPT_DIR/run_concurrency_sweep.py" \
    --workload-dir "$WORKLOAD_DIR" \
    --output-dir "$SMOKE_DIR" \
    --decoder-model "$DECODER_MODEL" \
    --encoder-model "$ENCODER_MODEL" \
    --concurrencies 8 \
    --smoke 2>&1 | tee "$SMOKE_DIR/smoke.log"

python "$SCRIPT_DIR/run_concurrency_sweep.py" \
    --workload-dir "$WORKLOAD_DIR" \
    --output-dir "$MEASURED_DIR" \
    --decoder-model "$DECODER_MODEL" \
    --encoder-model "$ENCODER_MODEL" \
    --concurrencies 8 16 32 2>&1 | tee "$MEASURED_DIR/sweep.log"

python "$SCRIPT_DIR/validate_concurrency_results.py" "$MEASURED_DIR"
python "$SCRIPT_DIR/summarize_concurrency_results.py" "$MEASURED_DIR" \
    --markdown "$MEASURED_DIR/benchmark.md" \
    --csv "$MEASURED_DIR/benchmark.csv"

echo "benchmark=$MEASURED_DIR/benchmark.md"
