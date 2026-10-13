#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end check of Dynamo's SGLang /generate through the Rust sidecar, on
# one GPU, in two phases:
#
#   agg: SGLang :31000 is both the reference and the backend of
#        frontend :18000 -> sidecar. compare.py saves the reference answers.
#   pd:  SGLang prefill :31010 and decode :31020 behind
#        frontend :18100 -> two sidecars, compared with the saved answers.
#
# Usage: run.sh [agg|pd|all] [compare.py options...]

set -euo pipefail

PHASE="${1:-all}"
shift || true
E2E_DIR="$(dirname "$(readlink -f "$0")")"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
LOG_DIR="${LOG_DIR:-/tmp/dynamo-e2e}"
read -r -a SGLANG_ARGS <<<"${SGLANG_ARGS:---moe-runner-backend triton --enable-return-routed-experts}"
mkdir -p "$LOG_DIR"
PIDS=()

stop_all() {
    for pid in "${PIDS[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
    for _ in $(seq 30); do
        local alive=0
        for pid in "${PIDS[@]}"; do kill -0 -- "-$pid" 2>/dev/null && alive=1; done
        ((alive)) || break
        sleep 1
    done
    for pid in "${PIDS[@]}"; do kill -KILL -- "-$pid" 2>/dev/null || true; done
    PIDS=()
}
trap stop_all EXIT INT TERM

start() {
    local name="$1"
    shift
    setsid "$@" >"$LOG_DIR/$name.log" 2>&1 &
    PIDS+=("$!")
}

wait_http() {
    local url="$1" deadline=$((SECONDS + ${2:-1200}))
    until curl -sf -o /dev/null "$url"; do
        for pid in "${PIDS[@]}"; do
            kill -0 "$pid" 2>/dev/null || { echo "a process exited while waiting for $url" >&2; return 1; }
        done
        ((SECONDS < deadline)) || { echo "timed out waiting for $url" >&2; return 1; }
        sleep 2
    done
}

sglang() {
    local name="$1" http="$2" grpc="$3" mem="$4"
    shift 4
    start "$name" python3 -m sglang.launch_server \
        --model-path "$MODEL" --host 127.0.0.1 --port "$http" --grpc-port "$grpc" \
        --incremental-streaming-output --mem-fraction-static "$mem" \
        --context-length 4096 --max-running-requests 8 "$@" "${SGLANG_ARGS[@]}"
}

compare() {
    python3 "$E2E_DIR/compare.py" --model "$MODEL" --reference-file "$LOG_DIR/reference.json" "$@"
}

run_agg() {
    local stack=(env DYN_FILE_KV="$LOG_DIR/kv-agg" DYN_SGLANG_ENABLE_GENERATE=true)
    sglang sglang-agg 31000 31001 0.8
    wait_http http://127.0.0.1:31000/health
    start sidecar-agg "${stack[@]}" DYN_SYSTEM_PORT=18081 dynamo-sglang-sidecar --grpc-endpoint 127.0.0.1:31001
    start frontend-agg "${stack[@]}" python3 -m dynamo.frontend --http-port 18000
    wait_http http://127.0.0.1:18081/health 300
    compare --name agg --dynamo http://127.0.0.1:18000 --reference http://127.0.0.1:31000 "$@"
}

run_pd() {
    local stack=(env DYN_FILE_KV="$LOG_DIR/kv-pd" DYN_SGLANG_ENABLE_GENERATE=true)
    local disagg=(--disaggregation-bootstrap-port 31998 --disaggregation-transfer-backend nixl)
    # Start one at a time: each server sizes its KV cache from the memory free
    # when it starts, so concurrent loads miscount each other's weights.
    sglang sglang-prefill 31010 31011 0.45 --disaggregation-mode prefill "${disagg[@]}"
    wait_http http://127.0.0.1:31010/health
    sglang sglang-decode 31020 31021 0.85 --disaggregation-mode decode "${disagg[@]}"
    wait_http http://127.0.0.1:31020/health
    start sidecar-prefill "${stack[@]}" DYN_SYSTEM_PORT=18082 dynamo-sglang-sidecar \
        --grpc-endpoint 127.0.0.1:31011 --bootstrap-host 127.0.0.1
    start sidecar-decode "${stack[@]}" DYN_SYSTEM_PORT=18083 dynamo-sglang-sidecar --grpc-endpoint 127.0.0.1:31021
    start frontend-pd "${stack[@]}" python3 -m dynamo.frontend --http-port 18100
    wait_http http://127.0.0.1:18082/health 300
    wait_http http://127.0.0.1:18083/health 300
    compare --name pd --dynamo http://127.0.0.1:18100 "$@"
}

status=0
if [[ "$PHASE" == agg || "$PHASE" == all ]]; then
    run_agg "$@" || status=1
    stop_all
fi
if [[ "$PHASE" == pd || "$PHASE" == all ]]; then
    run_pd "$@" || status=1
fi
exit "$status"
