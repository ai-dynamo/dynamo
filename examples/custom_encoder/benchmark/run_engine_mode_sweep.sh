#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LAUNCH_SCRIPT="$REPO_ROOT/examples/custom_encoder/launch/agg_qwen3_vl.sh"

GPU="${GPU:-0}"
HTTP_PORT="${HTTP_PORT:-18080}"
FRONTEND_SYSTEM_PORT="${FRONTEND_SYSTEM_PORT:-18081}"
WORKER_SYSTEM_PORT="${WORKER_SYSTEM_PORT:-18082}"
REQUEST_COUNT="${REQUEST_COUNT:-1000}"
WARMUP_REQUEST_COUNT="${WARMUP_REQUEST_COUNT:-32}"
KV_CACHE_BYTES="${KV_CACHE_BYTES:-8589934592}"
REPEATS="${REPEATS:-5}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-2B-Instruct}"
CONCURRENCIES="${CONCURRENCIES:-1 2 4 8}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
SWEEP_DIR="${SWEEP_DIR:-$REPO_ROOT/logs/qwen3_vl_custom_encoder/engine_mode_$RUN_ID}"
INPUT_FILE="$SCRIPT_DIR/.data/qwen3_vl_300x300_${REQUEST_COUNT}req_isl600.jsonl"

server_pid=""
monitor_pid=""

stop_server() {
    if [[ -n "$monitor_pid" ]]; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
        monitor_pid=""
    fi
    if [[ -n "$server_pid" ]]; then
        kill -TERM -- "-$server_pid" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "$server_pid" 2>/dev/null || break
            server_state="$(ps -o stat= -p "$server_pid" 2>/dev/null || true)"
            [[ "$server_state" == Z* ]] && break
            sleep 1
        done
        kill -KILL -- "-$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
        server_pid=""
    fi
}
trap stop_server EXIT

if ss -tln | grep -q ":$HTTP_PORT "; then
    echo "HTTP port $HTTP_PORT is already in use" >&2
    exit 1
fi

mkdir -p "$SWEEP_DIR"
python "$SCRIPT_DIR/generate_workload.py" \
    --request-count "$REQUEST_COUNT" \
    --target-mean-isl 600

concurrency_order() {
    if [[ "$CONCURRENCIES" != "1 2 4 8" ]]; then
        echo "$CONCURRENCIES"
        return
    fi
    case "$1" in
        1) echo "1 2 4 8" ;;
        2) echo "8 4 2 1" ;;
        3) echo "2 8 1 4" ;;
        4) echo "4 1 8 2" ;;
        *) echo "1 4 2 8" ;;
    esac
}

mode_order() {
    if (( $1 % 2 == 1 )); then
        echo "async-mp sync-inproc"
    else
        echo "sync-inproc async-mp"
    fi
}

start_resource_monitor() {
    local resource_log=$1
    (
        echo "timestamp,gpu_memory_mib,gpu_util_pct,process_count,total_rss_kib"
        while kill -0 "$server_pid" 2>/dev/null; do
            gpu_values="$(nvidia-smi \
                --id="$GPU" \
                --query-gpu=memory.used,utilization.gpu \
                --format=csv,noheader,nounits)"
            process_values="$(ps -eo pgid=,rss= | awk -v pgid="$server_pid" \
                '$1 == pgid {count += 1; rss += $2} END {print count + 0 "," rss + 0}')"
            echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),$gpu_values,$process_values"
            sleep 1
        done
    ) > "$resource_log" &
    monitor_pid=$!
}

for repeat in $(seq 1 "$REPEATS"); do
    for concurrency in $(concurrency_order "$repeat"); do
        for mode in $(mode_order "$repeat"); do
            run_dir="$SWEEP_DIR/rep$repeat/$mode/conc$concurrency"
            artifact_dir="$run_dir/aiperf"
            server_log="$run_dir/server.log"
            mkdir -p "$run_dir"
            namespace="sync-inproc-$RUN_ID-r$repeat-c$concurrency-$mode"

            echo "Starting repeat=$repeat mode=$mode concurrency=$concurrency"
            setsid env \
                CUDA_VISIBLE_DEVICES="$GPU" \
                DYN_NAMESPACE="$namespace" \
                DYN_HTTP_PORT="$HTTP_PORT" \
                DYN_FRONTEND_SYSTEM_PORT="$FRONTEND_SYSTEM_PORT" \
                DYN_WORKER_SYSTEM_PORT="$WORKER_SYSTEM_PORT" \
                DYN_WORKER_GPU="$GPU" \
                DYN_VLLM_ENGINE_CLIENT_MODE="$mode" \
                DYN_VLLM_GPU_MEMORY_UTILIZATION=0.7 \
                DYN_QWEN3_VL_GRAPH_IMAGE_SIZES=300x300 \
                DYN_QWEN3_VL_GRAPH_BATCH_BUCKETS=1,2,4,8 \
                DYN_QWEN3_VL_PREPROCESS_CACHE_SIZE=0 \
                DYN_QWEN3_VL_EMBEDDING_CACHE_BYTES=0 \
                DYN_CUSTOM_ENCODER_QUEUE_WAIT_MS=0 \
                DYN_CUSTOM_ENCODER_TIMING=0 \
                "$LAUNCH_SCRIPT" \
                    --kv-cache-memory-bytes "$KV_CACHE_BYTES" \
                    > "$server_log" 2>&1 &
            server_pid=$!
            start_resource_monitor "$run_dir/resources.csv"

            ready=0
            for _ in $(seq 1 600); do
                if curl -sf "http://127.0.0.1:$HTTP_PORT/v1/models" \
                    | grep -Fq "$MODEL_NAME"; then
                    ready=1
                    break
                fi
                kill -0 "$server_pid" 2>/dev/null || break
                sleep 1
            done
            if [[ "$ready" != 1 ]]; then
                echo "Server failed to become ready; see $server_log" >&2
                exit 1
            fi

            SERVER_URL="http://127.0.0.1:$HTTP_PORT" \
            CONCURRENCY="$concurrency" \
            REQUEST_COUNT="$REQUEST_COUNT" \
            WARMUP_REQUEST_COUNT="$WARMUP_REQUEST_COUNT" \
            OSL=70 \
            INPUT_FILE="$INPUT_FILE" \
            ARTIFACT_DIR="$artifact_dir" \
                "$SCRIPT_DIR/run_aiperf.sh"

            cp "$server_log" "$artifact_dir/custom_encoder.log"
            stop_server
        done
    done
done

python "$SCRIPT_DIR/summarize_results.py" \
    "$SWEEP_DIR" \
    --output-dir "$SWEEP_DIR"

echo "sweep_dir=$SWEEP_DIR"
