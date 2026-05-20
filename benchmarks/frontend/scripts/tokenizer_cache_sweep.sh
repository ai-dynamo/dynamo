#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Tokenizer L1 cache impact sweep: cache_off vs cache_on across a concurrency grid.
#
# Isolates frontend overhead by pinning to 4 CPU cores and running the backend
# under mocker with --speedup-ratio 1000000 (effectively-instant token generation).
# Uses aiperf's --shared-system-prompt-length / --user-context-prompt-length so the
# 48k system prompt is shared across all requests — maximizing the cache hit rate.

set -euo pipefail

WORKTREE="/data/dynamo/.worktrees/tokenizer-cache"
VENV="/data/dynamo/.venv-test"
PYTHON="$VENV/bin/python"
AIPERF="$VENV/bin/aiperf"

MODEL="Qwen/Qwen3-0.6B"
NUM_WORKERS=2
FRONTEND_PORT=8000
BENCHMARK_DURATION=60
WARMUP_DURATION=10
SHARED_PROMPT_LEN=48000
USER_CONTEXT_LEN=12000
NUM_DATASET_ENTRIES=10000
OUTPUT_TOKENS_MEAN=500
CONCURRENCIES=(32 64 128 256)
SPEEDUP_RATIO=1000000

OUT_ROOT="${WORKTREE}/artifacts/tok_cache_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_ROOT"
echo "Results -> $OUT_ROOT" | tee "$OUT_ROOT/run.log"

# Common env for the python invocations
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
export HF_HUB_OFFLINE=0

# Bring up etcd + NATS once for the whole sweep (faster than per-run teardown).
if ! curl -sf http://localhost:2379/health >/dev/null 2>&1; then
    ETCD_DIR=$(mktemp -d)
    /usr/bin/etcd --data-dir="$ETCD_DIR" \
        --listen-client-urls=http://localhost:2379 \
        --advertise-client-urls=http://localhost:2379 \
        --listen-peer-urls=http://localhost:2380 \
        --initial-advertise-peer-urls=http://localhost:2380 \
        --initial-cluster=default=http://localhost:2380 \
        > "$OUT_ROOT/etcd.log" 2>&1 &
    ETCD_PID=$!
    for i in $(seq 1 30); do
        curl -sf http://localhost:2379/health >/dev/null 2>&1 && break
        sleep 1
    done
    echo "etcd PID=$ETCD_PID dir=$ETCD_DIR" | tee -a "$OUT_ROOT/run.log"
fi

if ! nc -z localhost 4222 2>/dev/null; then
    nats-server > "$OUT_ROOT/nats.log" 2>&1 &
    NATS_PID=$!
    for i in $(seq 1 30); do
        nc -z localhost 4222 2>/dev/null && break
        sleep 1
    done
    echo "nats PID=$NATS_PID" | tee -a "$OUT_ROOT/run.log"
fi

cleanup_all() {
    pkill -f 'dynamo.mocker' 2>/dev/null || true
    pkill -f 'dynamo.frontend' 2>/dev/null || true
    [[ -n "${ETCD_PID:-}" ]] && kill "$ETCD_PID" 2>/dev/null || true
    [[ -n "${NATS_PID:-}" ]] && kill "$NATS_PID" 2>/dev/null || true
    sleep 1
    pkill -9 -f 'dynamo.mocker' 2>/dev/null || true
    pkill -9 -f 'dynamo.frontend' 2>/dev/null || true
}
trap cleanup_all EXIT INT TERM

# ──────────────────────────────────────────────────────────────────────────────
# Per-cell function: bring up workers + frontend, run aiperf, capture, tear down
# ──────────────────────────────────────────────────────────────────────────────
run_cell() {
    local cache_setting="$1"   # "off" or "on"
    local concurrency="$2"
    local out_dir="$OUT_ROOT/cache=${cache_setting}_conc=${concurrency}"
    mkdir -p "$out_dir/aiperf"

    echo "" | tee -a "$OUT_ROOT/run.log"
    echo "=== cache=$cache_setting concurrency=$concurrency ===" | tee -a "$OUT_ROOT/run.log"

    # Mockers: identical for both cache settings.
    local worker_pids=()
    for i in $(seq 1 "$NUM_WORKERS"); do
        DYN_SYSTEM_PORT=$((8080 + i)) DYN_EVENT_PLANE=nats HF_HUB_OFFLINE=0 \
            "$PYTHON" -m dynamo.mocker \
                --model-path "$MODEL" \
                --model-name "$MODEL" \
                --speedup-ratio "$SPEEDUP_RATIO" \
                --request-plane tcp \
                > "$out_dir/mocker_${i}.log" 2>&1 &
        worker_pids+=($!)
    done
    echo "mocker PIDs: ${worker_pids[*]}" | tee -a "$OUT_ROOT/run.log"

    # Frontend with optional cache enable, pinned to cores 0-3.
    local cache_env=()
    if [[ "$cache_setting" == "on" ]]; then
        cache_env=(DYN_TOKENIZER_CACHE=1)
    fi

    env DYN_HTTP_PORT="$FRONTEND_PORT" DYN_REQUEST_PLANE=tcp DYN_EVENT_PLANE=nats HF_HUB_OFFLINE=0 \
        "${cache_env[@]}" \
        taskset -c 0-3 "$PYTHON" -m dynamo.frontend \
        > "$out_dir/frontend.log" 2>&1 &
    local fe_pid=$!
    echo "frontend PID: $fe_pid (taskset 0-3, cache=$cache_setting)" | tee -a "$OUT_ROOT/run.log"

    # Wait for model registration.
    local waited=0
    local ready=false
    while [[ $waited -lt 120 ]]; do
        if curl -sf --max-time 3 "http://127.0.0.1:$FRONTEND_PORT/v1/models" 2>/dev/null | \
            jq -e --arg m "$MODEL" '.data[]? | select(.id == $m)' >/dev/null 2>&1; then
            ready=true
            break
        fi
        if ! kill -0 "$fe_pid" 2>/dev/null; then
            echo "FRONTEND DIED" | tee -a "$OUT_ROOT/run.log"
            tail -30 "$out_dir/frontend.log" | tee -a "$OUT_ROOT/run.log"
            break
        fi
        sleep 2
        waited=$((waited + 2))
    done
    if [[ "$ready" != true ]]; then
        echo "skip cell, model not ready" | tee -a "$OUT_ROOT/run.log"
        kill "${worker_pids[@]}" "$fe_pid" 2>/dev/null || true
        sleep 2
        return 1
    fi
    echo "ready after ${waited}s" | tee -a "$OUT_ROOT/run.log"

    # Initial metrics snapshot.
    curl -s --max-time 5 "http://127.0.0.1:$FRONTEND_PORT/metrics" > "$out_dir/metrics_initial.txt" 2>/dev/null || true

    # Run aiperf.
    HF_HUB_OFFLINE=0 "$AIPERF" profile \
        --artifact-dir "$out_dir/aiperf" \
        --model "$MODEL" \
        --endpoint-type chat \
        --endpoint /v1/chat/completions \
        --streaming \
        --url "http://127.0.0.1:$FRONTEND_PORT" \
        --shared-system-prompt-length "$SHARED_PROMPT_LEN" \
        --user-context-prompt-length "$USER_CONTEXT_LEN" \
        --num-dataset-entries "$NUM_DATASET_ENTRIES" \
        --output-tokens-mean "$OUTPUT_TOKENS_MEAN" \
        --output-tokens-stddev 0 \
        --extra-inputs max_tokens:"$OUTPUT_TOKENS_MEAN" \
        --extra-inputs min_tokens:"$OUTPUT_TOKENS_MEAN" \
        --extra-inputs ignore_eos:true \
        --extra-inputs repetition_penalty:1.0 \
        --extra-inputs temperature:0.0 \
        --concurrency "$concurrency" \
        --benchmark-duration "$BENCHMARK_DURATION" \
        --warmup-duration "$WARMUP_DURATION" \
        --workers-max "$concurrency" \
        --record-processors 8 \
        --random-seed 42 \
        --ui simple \
        > "$out_dir/aiperf.stdout.log" 2>&1 \
        || echo "WARNING: aiperf failed for cache=$cache_setting conc=$concurrency" | tee -a "$OUT_ROOT/run.log"

    # Final metrics snapshot.
    curl -s --max-time 5 "http://127.0.0.1:$FRONTEND_PORT/metrics" > "$out_dir/metrics_final.txt" 2>/dev/null || true

    # Tear down frontend + mockers between cells (forces clean state per cache setting).
    kill "$fe_pid" "${worker_pids[@]}" 2>/dev/null || true
    sleep 2
    kill -9 "$fe_pid" "${worker_pids[@]}" 2>/dev/null || true
    # Wait for port release.
    for i in $(seq 1 15); do
        ss -tlnp 2>/dev/null | grep -q ":${FRONTEND_PORT} " || break
        sleep 1
    done
}

# ──────────────────────────────────────────────────────────────────────────────
# Main sweep loop
# ──────────────────────────────────────────────────────────────────────────────
for cache_setting in off on; do
    for conc in "${CONCURRENCIES[@]}"; do
        run_cell "$cache_setting" "$conc" || true
    done
done

echo "" | tee -a "$OUT_ROOT/run.log"
echo "All cells done. Artifacts in $OUT_ROOT" | tee -a "$OUT_ROOT/run.log"
