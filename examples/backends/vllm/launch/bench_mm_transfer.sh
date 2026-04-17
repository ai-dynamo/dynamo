#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Benchmark mm_kwargs transfer performance for SHM and/or NIXL modes.
# Wraps agg_multimodal_router.sh, sends N image requests, and reports
# per-stage timing from mm_kwargs_transfer.py's [TIMING] print lines.
#
# Usage (run from the dynamo repo root, inside the dynamo container):
#   bash examples/backends/vllm/launch/bench_mm_transfer.sh [shm] [nixl]
#
# Examples:
#   # Both modes (default)
#   SINGLE_GPU=true bash bench_mm_transfer.sh
#
#   # SHM only
#   SINGLE_GPU=true bash bench_mm_transfer.sh shm
#
#   # NIXL with Mikhail's UCX fix for clusters without CMA/XPMEM
#   SINGLE_GPU=true UCX_MM_ERROR_HANDLING=y bash bench_mm_transfer.sh nixl
#
#   # NIXL forcing CMA transport (same-node, bypasses InfiniBand HCA)
#   SINGLE_GPU=true UCX_TLS=cma,self bash bench_mm_transfer.sh nixl
#
#   # Both modes with more requests
#   SINGLE_GPU=true NUM_REQUESTS=10 bash bench_mm_transfer.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SINGLE_GPU="${SINGLE_GPU:-false}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.40}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
HTTP_PORT="${HTTP_PORT:-8000}"
NAMESPACE="${NAMESPACE:-dynamo}"
NATS_SERVER="${NATS_SERVER:-nats://127.0.0.1:4222}"
ETCD_ENDPOINTS="${ETCD_ENDPOINTS:-http://127.0.0.1:2379}"
NUM_REQUESTS="${NUM_REQUESTS:-5}"
IMAGE_URL="${IMAGE_URL:-http://images.cocodataset.org/test2017/000000000001.jpg}"

MODES=("${@:-shm nixl}")
if [[ $# -gt 0 ]]; then
    MODES=("$@")
else
    MODES=(shm nixl)
fi

LOGDIR="/tmp/bench_mm_transfer"
mkdir -p "${LOGDIR}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

kill_stack() {
    pkill -9 -f 'dynamo.frontend|dynamo.vllm' 2>/dev/null || true
    # Wait for GPU to free
    local deadline=$((SECONDS + 30))
    while (( SECONDS < deadline )); do
        [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]] && break
        sleep 2
    done
    sleep 1
}

wait_url() {
    local url="$1" label="$2" timeout_s="${3:-600}"
    local deadline=$((SECONDS + timeout_s))
    printf "  Waiting for %s" "${label}"
    while (( SECONDS < deadline )); do
        if curl -fsS "${url}" 2>/dev/null | grep -q '"object"'; then
            echo " ready"
            return 0
        fi
        printf "."
        sleep 5
    done
    echo " TIMEOUT"
    return 1
}

wait_frontend_processor() {
    local timeout_s="${1:-300}"
    local deadline=$((SECONDS + timeout_s))
    printf "  Waiting for frontend processor"
    while (( SECONDS < deadline )); do
        local code
        code=$(curl -sf -o /dev/null -w "%{http_code}" \
            -X POST "http://127.0.0.1:${HTTP_PORT}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" \
            2>/dev/null || echo "000")
        [[ "$code" == "200" ]] && echo " ready" && return 0
        printf "."
        sleep 5
    done
    echo " TIMEOUT"
    return 1
}

send_requests() {
    local n="$1"
    for i in $(seq 1 "${n}"); do
        local resp
        resp=$(curl -sf --max-time 60 \
            -X POST "http://127.0.0.1:${HTTP_PORT}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Describe this image briefly\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"${IMAGE_URL}\"}}]}],\"max_tokens\":16}" \
            2>/dev/null \
            | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'][:60])" 2>/dev/null \
            || echo "(failed)")
        echo "    req ${i}: ${resp}"
    done
}

print_timing_summary() {
    local logfile="$1"
    local mode="$2"

    echo
    echo "  Raw [TIMING] lines:"
    grep '\[TIMING\]' "${logfile}" 2>/dev/null | sed 's/^/    /' || echo "    (none found)"

    echo
    echo "  Averages:"
    python3 - "${logfile}" "${mode}" <<'PYEOF'
import sys, re, statistics

logfile, mode = sys.argv[1], sys.argv[2]
fields = {}
sender_totals = []
receiver_totals = []

with open(logfile) as f:
    for line in f:
        if '[TIMING]' not in line:
            continue
        # Track sender prepare total and receiver total separately for e2e.
        if 'Sender] prepare total=' in line:
            m = re.search(r'prepare total=([\d.]+)ms', line)
            if m:
                sender_totals.append(float(m.group(1)))
        elif re.search(r'Receiver\].*\btotal=([\d.]+)ms', line):
            m = re.search(r'\btotal=([\d.]+)ms', line)
            if m:
                receiver_totals.append(float(m.group(1)))
        # Collect per-stage fields (skip 'total' to avoid conflation).
        for m in re.finditer(r'(\w+)=([\d.]+)ms', line):
            key, val = m.group(1), float(m.group(2))
            if key != 'total':
                fields.setdefault(key, []).append(val)

if not fields and not sender_totals:
    print("    (no [TIMING] data — check that mm_kwargs_transfer.py has timing instrumentation)")
    sys.exit(0)

# Per-stage breakdown.
order = ['pickle', 'encode', 'create_write', 'register', 'begin_read', 'wait', 'open_read', 'decode', 'unpickle', 'gather']
shown = set()
for k in order:
    if k in fields:
        vals = fields[k]
        print(f"    {k:20s} n={len(vals):2d}  avg={statistics.mean(vals):7.2f}ms  "
              f"min={min(vals):7.2f}ms  max={max(vals):7.2f}ms")
        shown.add(k)
for k in fields:
    if k not in shown:
        vals = fields[k]
        print(f"    {k:20s} n={len(vals):2d}  avg={statistics.mean(vals):7.2f}ms  "
              f"min={min(vals):7.2f}ms  max={max(vals):7.2f}ms")

# Sender / receiver / e2e totals.
print()
if sender_totals:
    print(f"    {'sender_prepare':20s} n={len(sender_totals):2d}  avg={statistics.mean(sender_totals):7.2f}ms  "
          f"min={min(sender_totals):7.2f}ms  max={max(sender_totals):7.2f}ms")
if receiver_totals:
    print(f"    {'receiver_total':20s} n={len(receiver_totals):2d}  avg={statistics.mean(receiver_totals):7.2f}ms  "
          f"min={min(receiver_totals):7.2f}ms  max={max(receiver_totals):7.2f}ms")
if sender_totals and receiver_totals:
    n = min(len(sender_totals), len(receiver_totals))
    e2e = [sender_totals[i] + receiver_totals[i] for i in range(n)]
    print(f"    {'e2e (sender+recv)':20s} n={n:2d}  avg={statistics.mean(e2e):7.2f}ms  "
          f"min={min(e2e):7.2f}ms  max={max(e2e):7.2f}ms")
PYEOF
}

# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------
declare -A SUMMARY

for mode in "${MODES[@]}"; do
    mode="${mode,,}"
    logfile="${LOGDIR}/stack_${mode}.log"

    echo
    echo "======================================================="
    echo " Mode: ${mode^^}  (log: ${logfile})"
    if [[ "${mode}" == "nixl" ]]; then
        [[ -n "${UCX_TLS:-}" ]]              && echo " UCX_TLS=${UCX_TLS}"
        [[ -n "${UCX_MM_ERROR_HANDLING:-}" ]] && echo " UCX_MM_ERROR_HANDLING=${UCX_MM_ERROR_HANDLING}"
    fi
    echo "======================================================="

    kill_stack

    # Build explicit UCX overrides to pass into the stack (nixl mode only).
    # Using an array avoids word-splitting issues with multiple vars.
    UCX_VARS=()
    if [[ "${mode}" == "nixl" ]]; then
        [[ -n "${UCX_TLS:-}" ]]              && UCX_VARS+=("UCX_TLS=${UCX_TLS}")
        [[ -n "${UCX_MM_ERROR_HANDLING:-}" ]] && UCX_VARS+=("UCX_MM_ERROR_HANDLING=${UCX_MM_ERROR_HANDLING}")
    fi

    env \
        "${UCX_VARS[@]+"${UCX_VARS[@]}"}" \
        MODEL="${MODEL}" \
        NAMESPACE="${NAMESPACE}" \
        HTTP_PORT="${HTTP_PORT}" \
        BLOCK_SIZE="${BLOCK_SIZE}" \
        GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
        MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
        NUM_WORKERS="${NUM_WORKERS}" \
        SINGLE_GPU="${SINGLE_GPU}" \
        NATS_SERVER="${NATS_SERVER}" \
        ETCD_ENDPOINTS="${ETCD_ENDPOINTS}" \
        DYNAMO_MM_TRANSFER="${mode}" \
        bash "${SCRIPT_DIR}/agg_multimodal_router.sh" \
        > "${logfile}" 2>&1 &
    STACK_PID=$!

    # Wait for the stack to declare itself ready
    if ! wait_url "http://127.0.0.1:${HTTP_PORT}/v1/models" "frontend models API" 600; then
        echo "ERROR: stack failed to start. Check ${logfile}" >&2
        kill "${STACK_PID}" 2>/dev/null || true
        kill_stack
        continue
    fi
    wait_frontend_processor 300

    echo "  Sending ${NUM_REQUESTS} requests with the same image..."
    send_requests "${NUM_REQUESTS}"

    print_timing_summary "${logfile}" "${mode}"
    SUMMARY[$mode]="${logfile}"

    kill "${STACK_PID}" 2>/dev/null || true
    kill_stack
done

# ---------------------------------------------------------------------------
# Side-by-side comparison
# ---------------------------------------------------------------------------
if [[ ${#SUMMARY[@]} -gt 1 ]]; then
    echo
    echo "======================================================="
    echo " Comparison summary"
    echo "======================================================="
    for mode in shm nixl tcp; do
        [[ -z "${SUMMARY[$mode]+x}" ]] && continue
        echo
        echo "  [${mode^^}]"
        python3 - "${SUMMARY[$mode]}" <<'PYEOF'
import sys, re, statistics

fields = {}
sender_totals = []
receiver_totals = []

with open(sys.argv[1]) as f:
    for line in f:
        if '[TIMING]' not in line:
            continue
        if 'Sender] prepare total=' in line:
            m = re.search(r'prepare total=([\d.]+)ms', line)
            if m:
                sender_totals.append(float(m.group(1)))
        elif re.search(r'Receiver\].*\btotal=([\d.]+)ms', line):
            m = re.search(r'\btotal=([\d.]+)ms', line)
            if m:
                receiver_totals.append(float(m.group(1)))
        for m in re.finditer(r'(\w+)=([\d.]+)ms', line):
            key, val = m.group(1), float(m.group(2))
            if key != 'total':
                fields.setdefault(key, []).append(val)

for k in ['pickle', 'encode', 'create_write', 'register', 'begin_read', 'wait', 'open_read', 'decode', 'unpickle', 'gather']:
    if k in fields:
        print(f"    {k:20s} avg={statistics.mean(fields[k]):7.2f}ms")
if sender_totals:
    print(f"    {'sender_prepare':20s} avg={statistics.mean(sender_totals):7.2f}ms")
if receiver_totals:
    print(f"    {'receiver_total':20s} avg={statistics.mean(receiver_totals):7.2f}ms")
if sender_totals and receiver_totals:
    n = min(len(sender_totals), len(receiver_totals))
    e2e = [sender_totals[i] + receiver_totals[i] for i in range(n)]
    print(f"    {'e2e (sender+recv)':20s} avg={statistics.mean(e2e):7.2f}ms")
PYEOF
    done
fi

echo
echo "Logs: ${LOGDIR}/stack_shm.log  ${LOGDIR}/stack_nixl.log  ${LOGDIR}/stack_tcp.log"
echo "Grep:  grep '\[TIMING\]' <logfile>"
