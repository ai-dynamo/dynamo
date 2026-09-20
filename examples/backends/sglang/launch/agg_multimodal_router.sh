#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# MM-aware KV routing for SGLang.
#
# Requires dynamo built with `--features mm-routing` (default in the
# dynamo sglang container image) and sglang carrying the
# sgl-project/sglang#25300 mm_hashes patch. Without the mm_hashes kwarg
# the dynamo glue silently falls back to text-prefix routing.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${DYNAMO_ROOT}"
# shellcheck source=../../../common/gpu_utils.sh
source "${SCRIPT_DIR}/../../../common/gpu_utils.sh"
# shellcheck source=../../../common/launch_utils.sh
source "${SCRIPT_DIR}/../../../common/launch_utils.sh"

MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
NAMESPACE="${NAMESPACE:-sglang-mm-router}"
# Honor DYN_HTTP_PORT (set by tests/serve harness for dynamic port allocation)
HTTP_PORT="${HTTP_PORT:-${DYN_HTTP_PORT:-8000}}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SINGLE_GPU="${SINGLE_GPU:-false}"  # pack all workers onto GPU 0 (functional testing)
NATS_SERVER="${NATS_SERVER:-nats://127.0.0.1:4222}"
ETCD_ENDPOINTS="${ETCD_ENDPOINTS:-http://127.0.0.1:2379}"
SGLANG_SYSTEM_PORT_BASE="${SGLANG_SYSTEM_PORT_BASE:-18091}"
# Differs from agg_router.sh's 5557 so the two variants can co-run.
KV_EVENTS_PORT_BASE="${KV_EVENTS_PORT_BASE:-29090}"

# One shared budget for workers plus frontend, so NUM_WORKERS cannot multiply
# it; stays under the serve tests' 400s so a stall is reported here.
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-330}"

DYN_LOG_VAL="${DYN_LOG:-info,mm_routing=debug,dynamo_kv_router::scheduling=debug,dynamo_llm::kv_router=debug}"

# Pass-through extra args for `python -m dynamo.sglang`.
SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}"

PASSTHRU_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --single-gpu) SINGLE_GPU="true"; shift ;;
        -h|--help)
            cat <<EOF
Usage: $0 [--model NAME] [--num-workers N] [--single-gpu] [EXTRA_SGLANG_ARGS...]

Env vars:
  MODEL                       (default Qwen/Qwen3-VL-2B-Instruct)
  NUM_WORKERS                 (default 2)
  HTTP_PORT                   (default 8000)
  BLOCK_SIZE                  (default 16 — SGLang \`--page-size\`)
  MAX_MODEL_LEN               (default 4096)
  KV_EVENTS_PORT_BASE         (default 29090 — worker i uses port BASE + (i-1))
  STARTUP_TIMEOUT             (default 330 — whole-startup budget in seconds)
  DYN_LOG                     (default info + mm_routing + scheduling debug)
EOF
            exit 0
            ;;
        *) PASSTHRU_ARGS+=("$1"); shift ;;
    esac
done

# Hybrid GDN families (Qwen3.5/Qwen3.6) need extra_buffer scheduling for
# MambaRadixCache to engage; default no_buffer silently disables prefix
# cache. Evaluated after CLI parsing so --model overrides take effect.
MAMBA_ARGS=()
case "${MODEL}" in
    *Qwen3.5*|*Qwen3.6*) MAMBA_ARGS=(--mamba-scheduler-strategy extra_buffer) ;;
esac

print_launch_banner --multimodal --no-curl \
    "MM Exact Routing (SGLang)" "${MODEL}" "${HTTP_PORT}" \
    "NUM_WORKERS:  ${NUM_WORKERS}" \
    "BLOCK_SIZE:   ${BLOCK_SIZE}" \
    "NAMESPACE:    ${NAMESPACE}"

trap 'trap - EXIT INT TERM; echo; kill 0' EXIT INT TERM

# Non-zero as soon as any launched worker has died, naming it and its exit
# status. Every startup poll checks all of WORKER_PIDS, not just the process it
# happens to be waiting on: the waits are sequential, so a worker that dies
# while a different one is still loading would otherwise go unnoticed until the
# shared deadline — the slow, diagnostics-free failure this startup path exists
# to avoid.
check_workers_alive() {
    local i pid status
    for i in "${!WORKER_PIDS[@]}"; do
        pid="${WORKER_PIDS[i]}"
        if ! kill -0 "${pid}" 2>/dev/null; then
            status=0
            wait "${pid}" 2>/dev/null || status=$?
            echo "SGLang backend $((i + 1)) (pid ${pid}) exited with status ${status} before startup completed" >&2
            return 1
        fi
    done
    return 0
}

# Transfer timeout for one readiness probe, given an absolute ${1} deadline.
# Without it a peer that accepts the connection and then never answers blocks
# curl forever, and the poll loop only tests the deadline between probes, so the
# whole startup budget could be overrun. Capped well below the budget so one
# hung probe cannot suppress the worker liveness scan either.
probe_timeout() {
    local remaining=$(( $1 - SECONDS ))
    if (( remaining > 10 )); then remaining=10; fi
    if (( remaining < 1 )); then remaining=1; fi
    echo "${remaining}"
}

# Poll ${url} until ready, giving up at the absolute ${deadline} (a SECONDS
# value) or as soon as any worker dies, so the worker's own error is reported.
wait_ready() {
    local url="$1" name="$2" deadline="$3"
    echo "Waiting for ${name} ..."
    while (( SECONDS < deadline )); do
        if curl -fsS --max-time "$(probe_timeout "${deadline}")" "${url}" 2>/dev/null | grep -q '"status"[[:space:]]*:[[:space:]]*"ready"'; then
            echo "${name} is ready"
            return 0
        fi
        check_workers_alive || return 1
        sleep 1
    done
    echo "${name} did not become ready within the ${STARTUP_TIMEOUT}s startup budget" >&2
    return 1
}

COMMON_ENV=(
    "DYN_NAMESPACE=${NAMESPACE}"
    "DYN_REQUEST_PLANE=tcp"
    "NATS_SERVER=${NATS_SERVER}"
    "ETCD_ENDPOINTS=${ETCD_ENDPOINTS}"
    "DYN_MM_ALLOW_INTERNAL=1"
)

GPU_MEM_ARGS=$(build_sglang_gpu_mem_args)

# Per-worker DYN_SYSTEM_PORT{i} is set by xdist for parallel test runs; fall
# back to script defaults otherwise. KV-event ports always come from the
# script's own KV_EVENTS_PORT_BASE block (29090+) — xdist only reserves one
# DYN_VLLM_KV_EVENT_PORT so deriving `base + (i-1)` from it would collide
# with adjacent test slots.
WORKER_PORTS=()
KV_EVENTS_PORTS=()
WORKER_PIDS=()
STARTUP_DEADLINE=$((SECONDS + STARTUP_TIMEOUT))
for i in $(seq 1 "${NUM_WORKERS}"); do
    DEFAULT_WORKER_PORT=$((SGLANG_SYSTEM_PORT_BASE + (i - 1) * 2))
    HARNESS_VAR="DYN_SYSTEM_PORT${i}"
    WORKER_PORT="${!HARNESS_VAR:-${DEFAULT_WORKER_PORT}}"
    WORKER_PORTS+=("${WORKER_PORT}")
    KV_EVENTS_PORT=$((KV_EVENTS_PORT_BASE + (i - 1)))
    KV_EVENTS_PORTS+=("${KV_EVENTS_PORT}")
    if [[ "${SINGLE_GPU}" == "true" ]]; then GPU_ID=0; else GPU_ID=$((i - 1)); fi

    KV_EVENTS_CONFIG="{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${KV_EVENTS_PORT}\"}"

    echo "--- launching SGLang worker $i (GPU=${GPU_ID}, system_port=${WORKER_PORT}, kv_events=tcp://*:${KV_EVENTS_PORT}) ---"
    env "${COMMON_ENV[@]}" \
        "DYN_SYSTEM_PORT=${WORKER_PORT}" \
        "CUDA_VISIBLE_DEVICES=${GPU_ID}" \
    python -m dynamo.sglang \
        --model-path "${MODEL}" \
        --served-model-name "${MODEL}" \
        --page-size "${BLOCK_SIZE}" \
        --context-length "${MAX_MODEL_LEN}" \
        --tp 1 \
        --trust-remote-code \
        --kv-events-config "${KV_EVENTS_CONFIG}" \
        --enable-metrics \
        --disable-piecewise-cuda-graph \
        "${MAMBA_ARGS[@]}" \
        ${GPU_MEM_ARGS} ${SGLANG_EXTRA_ARGS} "${PASSTHRU_ARGS[@]}" &
    WORKER_PIDS+=("$!")
done

for i in $(seq 1 "${NUM_WORKERS}"); do
    wait_ready "http://127.0.0.1:${WORKER_PORTS[i-1]}/health" "SGLang backend $i" \
        "${STARTUP_DEADLINE}"
done

echo "=== Starting frontend (KV router, MM-aware routing) ==="
env "${COMMON_ENV[@]}" \
    "DYN_LOG=${DYN_LOG_VAL}" \
python -m dynamo.frontend \
    --http-port "${HTTP_PORT}" \
    --router-mode kv \
    --kv-cache-block-size "${BLOCK_SIZE}" &

echo "Waiting for frontend to accept requests ..."
FRONTEND_READY=false
while (( SECONDS < STARTUP_DEADLINE )); do
    HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" \
        --max-time "$(probe_timeout "${STARTUP_DEADLINE}")" \
        -X POST "http://127.0.0.1:${HTTP_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" \
        2>/dev/null || echo "000")
    [[ "$HTTP_CODE" == "200" ]] && { FRONTEND_READY=true; echo "Frontend ready"; break; }
    # A worker that dies after reporting ready would otherwise only surface as
    # the frontend never answering, at the far end of the shared deadline.
    check_workers_alive || exit 1
    sleep 2
done
if [[ "$FRONTEND_READY" != true ]]; then
    echo "Frontend did not become ready within the ${STARTUP_TIMEOUT}s startup budget" >&2
    exit 1
fi

echo
echo "=== All services are ready ==="
echo "Frontend:        http://127.0.0.1:${HTTP_PORT}"
for i in $(seq 1 "${NUM_WORKERS}"); do
    # Use the actual port values from the launch loop above so the
    # summary reflects DYN_SYSTEM_PORT{i} / KV_EVENTS_PORT_BASE overrides
    # the harness may have applied (instead of the default formula).
    echo "Worker $i health: http://127.0.0.1:${WORKER_PORTS[i-1]}/health"
    echo "Worker $i kv-events: tcp://*:${KV_EVENTS_PORTS[i-1]}"
done
echo
echo "Architecture: Rust frontend (MM-aware KV router) -> ${NUM_WORKERS}x SGLang workers"
echo "  - mm_hashes forwarded to SGLang GenerateReqInput.mm_hashes -> matching pad_value"
echo "  - Image dims via header-only HTTP fetch (Range: bytes=0-65535)"
echo "  - No PyO3, no GIL, no Python deps in the routing path"
echo
echo "Press Ctrl+C to stop all services"

wait_any_exit
