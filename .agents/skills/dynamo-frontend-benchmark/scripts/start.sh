#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Start the benchmark topology:
#   - 4 mock workers (one process, --num-workers 4) on cores 4-23
#   - 1 frontend (HTTP + KV router + tokenizer) on cores 0-3
# Assumes etcd (:2379) and nats (:4222, JetStream) are already running.
#
# Hardened against the two failure modes we hit:
#   1. frontend silently failing to bind :8000 while a dying old frontend or a
#      stale socket answers the readiness poll -> we wait for the port to be
#      free first, and fail fast if the launched frontend PID dies.
#   2. stale worker registrations in etcd from a previous (killed) run -> we
#      wait for the instance prefix to be empty before launching, then require
#      EXACTLY NUM_WORKERS fresh instances to appear.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

# --- Sanity: infra ---------------------------------------------------------
ss -ltn 2>/dev/null | grep -q ':2379' || { echo "ERROR: etcd not listening on :2379"; exit 1; }
ss -ltn 2>/dev/null | grep -q ':4222' || { echo "ERROR: nats not listening on :4222 (start with: nats-server -js)"; exit 1; }
echo "[infra] etcd :2379 and nats :4222 are up."

# --- Pre-flight: port :8000 must be free -----------------------------------
echo "[preflight] waiting for port :${HTTP_PORT} to be free ..."
for i in $(seq 1 30); do
    ss -ltn 2>/dev/null | grep -q ":${HTTP_PORT}\b" || { echo "[preflight] :${HTTP_PORT} free."; break; }
    [[ $i -eq 30 ]] && { echo "ERROR: port :${HTTP_PORT} still in use after 30s. Run ./stop.sh."; exit 1; }
    sleep 1
done

# --- Pre-flight: no stale worker instances in etcd -------------------------
echo "[preflight] waiting for stale worker instances to clear from etcd ..."
for i in $(seq 1 30); do
    n="$(count_workers)"
    [[ "$n" -eq 0 ]] && { echo "[preflight] etcd worker instances clear."; break; }
    [[ $i -eq 30 ]] && { echo "ERROR: $n stale worker instance(s) still in etcd after 30s (prefix $INSTANCE_PREFIX)."; exit 1; }
    sleep 1
done

# --- 4 mock workers (cores ${OTHER_CORES}) ---------------------------------
echo "[workers] launching ${NUM_WORKERS} mock workers on cores ${OTHER_CORES} ..."
taskset -c "$OTHER_CORES" "$DYN_PY" -m dynamo.mocker \
    --model-path "$MODEL" \
    --endpoint "$ENDPOINT" \
    --num-workers "$NUM_WORKERS" \
    --speedup-ratio 1000000 \
    --max-num-seqs 100000 \
    --max-num-batched-tokens 10000000 \
    --block-size "$BLOCK_SIZE" \
    > "$LOG_DIR/workers.log" 2>&1 &
WORKERS_PID=$!
echo "$WORKERS_PID" > "$LOG_DIR/workers.pid"
echo "[workers] pid $WORKERS_PID, logging to $LOG_DIR/workers.log"

STARTUP_OK=0
terminate_process_tree() {
    local pid="$1"
    local child
    [[ -n "$pid" ]] || return
    for child in $(pgrep -P "$pid" 2>/dev/null || true); do
        terminate_process_tree "$child"
    done
    kill "$pid" 2>/dev/null || true
}

cleanup_startup_failure() {
    [[ "$STARTUP_OK" == "1" ]] && return
    echo "[cleanup] startup failed; stopping launched worker/frontend processes ..."
    terminate_process_tree "${FRONTEND_PID:-}"
    terminate_process_tree "${WORKERS_PID:-}"
}
trap cleanup_startup_failure EXIT

# Wait for EXACTLY NUM_WORKERS fresh instances to register.
echo "[workers] waiting for ${NUM_WORKERS} instances to register in etcd ..."
for i in $(seq 1 60); do
    kill -0 "$WORKERS_PID" 2>/dev/null || { echo "ERROR: workers process died. Tail:"; tail -20 "$LOG_DIR/workers.log"; exit 1; }
    n="$(count_workers)"
    [[ "$n" -eq "$NUM_WORKERS" ]] && { echo "[workers] ${n} instances registered."; break; }
    [[ "$n" -gt "$NUM_WORKERS" ]] && { echo "ERROR: $n worker instances registered (> ${NUM_WORKERS}); stale entries present."; exit 1; }
    [[ $i -eq 60 ]] && { echo "ERROR: only $n/${NUM_WORKERS} workers registered after 60s."; exit 1; }
    sleep 1
done

# --- Frontend (cores ${FRONTEND_CORES}) ------------------------------------
# Optional: FRONTEND_LD_PRELOAD applies an allocator (e.g. jemalloc) to the
# frontend ONLY, leaving the mock workers on glibc — keeps the A/B isolated.
if [[ -n "${FRONTEND_LD_PRELOAD:-}" ]]; then
    echo "[frontend] LD_PRELOAD=$FRONTEND_LD_PRELOAD"
fi
if [[ -n "${FRONTEND_MALLOC_CONF:-}" ]]; then
    echo "[frontend] MALLOC_CONF=$FRONTEND_MALLOC_CONF"
fi
if [[ "${ISOLATE:-0}" == "1" ]]; then
    # Reserved-core mode (see bench/isolate.sh): cores ${FRONTEND_CORES} are kept
    # clear of all other userspace via slice confinement. Launch the frontend in a
    # top-level `bench.slice` pinned to those cores (a child of the confined
    # user.slice could NOT claim them — cgroup cpuset is hierarchical). Needs root
    # for systemd-run; `env ...` carries the vars through sudo's env scrub.
    echo "[frontend] ISOLATE=1: launching in bench.slice on cores ${FRONTEND_CORES} (systemd-run, needs root) ..."
    sudo systemd-run --scope --slice=bench -p AllowedCPUs="${FRONTEND_CORES}" \
        env LD_PRELOAD="${FRONTEND_LD_PRELOAD:-}" \
            MALLOC_CONF="${FRONTEND_MALLOC_CONF:-}" \
            FASTOKENS_SEQUENTIAL="${FASTOKENS_SEQUENTIAL:-}" \
            DYN_TOKENIZER="$DYN_TOKENIZER" \
            DYN_TOKENIZER_CACHE="$DYN_TOKENIZER_CACHE" \
            DYN_TOKENIZER_CACHE_BYTES="$DYN_TOKENIZER_CACHE_BYTES" \
            "$DYN_PY" -m dynamo.frontend \
                --router-mode kv \
                --kv-cache-block-size "$BLOCK_SIZE" \
                --http-port "$HTTP_PORT" \
        > "$LOG_DIR/frontend.log" 2>&1 &
    # PID is a descendant of systemd-run/sudo; resolve the python process.
    FRONTEND_PID=""
    for _ in $(seq 1 20); do
        FRONTEND_PID="$(pgrep -f 'dynamo.frontend --router-mode kv' | head -1)"
        [[ -n "$FRONTEND_PID" ]] && break
        sleep 0.5
    done
    [[ -n "$FRONTEND_PID" ]] || { echo "ERROR: frontend (isolated) did not start; check $LOG_DIR/frontend.log"; tail -20 "$LOG_DIR/frontend.log"; exit 1; }
else
    echo "[frontend] launching on cores ${FRONTEND_CORES} (fastokens + 1GiB cache, kv routing) ..."
    LD_PRELOAD="${FRONTEND_LD_PRELOAD:-}" \
    MALLOC_CONF="${FRONTEND_MALLOC_CONF:-}" \
    FASTOKENS_SEQUENTIAL="${FASTOKENS_SEQUENTIAL:-}" \
    DYN_TOKENIZER="$DYN_TOKENIZER" \
    DYN_TOKENIZER_CACHE="$DYN_TOKENIZER_CACHE" \
    DYN_TOKENIZER_CACHE_BYTES="$DYN_TOKENIZER_CACHE_BYTES" \
    taskset -c "$FRONTEND_CORES" "$DYN_PY" -m dynamo.frontend \
        --router-mode kv \
        --kv-cache-block-size "$BLOCK_SIZE" \
        --http-port "$HTTP_PORT" \
        > "$LOG_DIR/frontend.log" 2>&1 &
    FRONTEND_PID=$!
fi
echo "$FRONTEND_PID" > "$LOG_DIR/frontend.pid"
echo "[frontend] pid $FRONTEND_PID, logging to $LOG_DIR/frontend.log"

# --- Wait for readiness: PID must stay alive AND model must be served ------
echo "[wait] polling http://localhost:${HTTP_PORT}/v1/models for ${MODEL} ..."
for i in $(seq 1 60); do
    if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
        echo "ERROR: frontend process died during startup (likely bind failure). Tail:"
        tail -25 "$LOG_DIR/frontend.log"
        exit 1
    fi
    if curl -sf "http://localhost:${HTTP_PORT}/v1/models" 2>/dev/null | grep -Fq "$MODEL"; then
        echo "[ready] frontend (pid $FRONTEND_PID) serving model after ${i}s; ${NUM_WORKERS} workers live."
        STARTUP_OK=1
        trap - EXIT
        exit 0
    fi
    sleep 1
done
echo "ERROR: model did not appear within 60s. Check $LOG_DIR/frontend.log and $LOG_DIR/workers.log"
exit 1
