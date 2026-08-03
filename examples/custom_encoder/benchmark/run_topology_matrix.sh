#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run inline, frontend-routed, and PD-routed custom-encoder topologies five
# times each. Every measured cell starts fresh FE/E/PD processes.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RESULT_ROOT="${OUTPUT_DIR:-/dynamo-tmp/logs/$(date -u +%m-%d)/custom-encoder-topology-matrix}"
WORKLOAD_ROOT="$RESULT_ROOT/workloads"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
CONCURRENCY="${CONCURRENCY:-64}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
SERVER_PID=""
SAMPLER_PID=""

cleanup_cell() {
    if [[ -n "$SAMPLER_PID" ]]; then
        kill "$SAMPLER_PID" 2>/dev/null || true
        wait "$SAMPLER_PID" 2>/dev/null || true
        SAMPLER_PID=""
    fi
    if [[ -n "$SERVER_PID" ]]; then
        kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 60); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
}
trap cleanup_cell EXIT

mkdir -p "$WORKLOAD_ROOT/measured" "$WORKLOAD_ROOT/warmup" "$RESULT_ROOT"
if [[ ! -f "$WORKLOAD_ROOT/measured/workload_manifest.json" ]]; then
    python -m examples.custom_encoder.benchmark.safeguard_proxy_workload generate \
        --output-dir "$WORKLOAD_ROOT/measured" \
        --requests 1000 \
        --seed 42 \
        --image-size-count 300x300:500 \
        --image-size-count 500x500:500
fi
if [[ ! -f "$WORKLOAD_ROOT/warmup/workload_manifest.json" ]]; then
    python -m examples.custom_encoder.benchmark.safeguard_proxy_workload generate \
        --output-dir "$WORKLOAD_ROOT/warmup" \
        --requests 20 \
        --seed 1042 \
        --image-size-count 300x300:10 \
        --image-size-count 500x500:10
fi
python -m examples.custom_encoder.benchmark.safeguard_proxy_workload validate \
    "$WORKLOAD_ROOT/measured" \
    --unique-images 1000 \
    --image-size-count 300x300:500 \
    --image-size-count 500x500:500
python -m examples.custom_encoder.benchmark.safeguard_proxy_workload validate \
    "$WORKLOAD_ROOT/warmup" \
    --unique-images 20 \
    --image-size-count 300x300:10 \
    --image-size-count 500x500:10

MEASURED_INPUT="$WORKLOAD_ROOT/measured/image_custom_1000_isl644.jsonl"
WARMUP_INPUT="$WORKLOAD_ROOT/warmup/image_custom_20_isl644.jsonl"

export CUDA_VISIBLE_DEVICES="$GPU"
export DYN_HTTP_PORT="$HTTP_PORT"
export DYN_WORKER_GPU="$GPU"
export DYN_MAX_MODEL_LEN=2048
export DYN_MAX_NUM_SEQS=64
export DYN_VLLM_GPU_MEMORY_UTILIZATION=0.4
export DYN_QWEN2_VL_PREPROCESS_CONCURRENCY=64
export DYN_QWEN2_VL_MAX_BATCH_PATCHES=82944
export DYN_QWEN2_VL_GRAPH_BATCH_BUCKETS=1,2,4,8,16,32,64
export DYN_QWEN2_VL_MAX_BATCH_ITEMS=64
export DYN_QWEN2_VL_GRAPH_IMAGE_SIZES=300x300,500x500
export DYN_QWEN2_VL_PREPROCESS_CACHE_SIZE=0

launch_topology() {
    local topology="$1"
    local output_dir="$2"
    local script
    local extra=()
    case "$topology" in
        inline)
            script="$REPO_ROOT/examples/custom_encoder/launch/agg_qwen2_5_vl_benchmark.sh"
            extra=(--no-enable-prefix-caching)
            ;;
        frontend)
            script="$REPO_ROOT/examples/custom_encoder/launch/frontend_qwen2_5_vl_benchmark.sh"
            ;;
        worker)
            script="$REPO_ROOT/examples/custom_encoder/launch/worker_qwen2_5_vl_benchmark.sh"
            ;;
        *) return 2 ;;
    esac
    setsid env DYN_NAMESPACE="dynamo-${topology}-$(date +%s%N)" \
        bash "$script" "${extra[@]}" >"$output_dir/server.log" 2>&1 &
    SERVER_PID=$!
}

wait_ready() {
    local response
    for _ in $(seq 1 1200); do
        if response=$(curl -fsS "http://127.0.0.1:$HTTP_PORT/v1/models" 2>/dev/null) \
            && python -c '
import json
import sys

model = sys.argv[1]
payload = json.load(sys.stdin)
raise SystemExit(
    0 if any(item.get("id") == model for item in payload.get("data", [])) else 1
)
' "$MODEL" <<<"$response"; then
            return 0
        fi
        kill -0 "$SERVER_PID" 2>/dev/null || return 1
        sleep 1
    done
    return 1
}

run_cell() {
    local repetition="$1"
    local topology="$2"
    local output_dir="$RESULT_ROOT/rep-$repetition/$topology"
    mkdir -p "$output_dir"
    cleanup_cell
    launch_topology "$topology" "$output_dir"
    wait_ready

    python -m examples.custom_encoder.benchmark.topology_benchmark_client \
        --input "$WARMUP_INPUT" \
        --output "$output_dir/warmup.json" \
        --expected-requests 20 \
        --concurrency "$CONCURRENCY"

    nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,memory.used \
        --format=csv,noheader,nounits -l 1 >"$output_dir/gpu_memory.csv" &
    SAMPLER_PID=$!
    python -m examples.custom_encoder.benchmark.topology_benchmark_client \
        --input "$MEASURED_INPUT" \
        --output "$output_dir/measured.json" \
        --expected-requests 1000 \
        --concurrency "$CONCURRENCY"
    cleanup_cell
}

orders=(
    "inline frontend worker"
    "frontend worker inline"
    "worker inline frontend"
    "inline worker frontend"
    "frontend inline worker"
)
for repetition in 1 2 3 4 5; do
    read -r -a order <<<"${orders[$((repetition - 1))]}"
    for topology in "${order[@]}"; do
        run_cell "$repetition" "$topology"
    done
done

python -m examples.custom_encoder.benchmark.summarize_topology_matrix "$RESULT_ROOT" \
    | tee "$RESULT_ROOT/summary.log"
