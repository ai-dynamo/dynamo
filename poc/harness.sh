#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Engine-level GMS shadow-failover smoke test — K8s pod port of
# validate-failover/test_lock_driven_failover.sh.
#
# Runs INSIDE the dev container of the failover-smoke pod. Needs only the
# installed modules (gpu_memory_service, dynamo.vllm, dynamo.frontend) plus
# etcd (localhost:2379) and NATS (localhost:4222) sidecars. No repo source,
# no venv activation, no .env — env comes from the pod (HF_TOKEN,
# ETCD_ENDPOINTS, NATS_SERVER).
#
# Validates:
#   D5 — Deterministic weight loading (ENGINE_ID=1 RO blocks until ENGINE_ID=0 commits)
#   D4 — Lock-driven auto-wake (flock release on process death triggers failover)
#   D2 — Health probe behavior (200 in STANDBY and ACTIVE)
#   D7 — Process death as fencing (exactly one engine registered at a time)
#
# Usage: ./run_failover_test.sh [MODEL_NAME] [TP_SIZE]
set -u

MODEL_NAME="${1:-Qwen/Qwen3-0.6B}"
TP_SIZE="${2:-2}"

LOG_DIR="/tmp/failover_test_$$"
mkdir -p "$LOG_DIR"
LOCK_PATH="$LOG_DIR/failover.lock"

GMS_W0_LOG="$LOG_DIR/gms_w0.log"   # device-0 weights GMS server (commit marker lives here)
ENGINE_A_LOG="$LOG_DIR/engine_a.log"; ENGINE_B_LOG="$LOG_DIR/engine_b.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"

GMS_PIDS=(); ENGINE_A_PID=""; ENGINE_B_PID=""; FRONTEND_PID=""
WINNER_PID=""; LOSER_PID=""; WINNER_LOG=""; LOSER_LOG=""; WINNER_PORT=""; LOSER_PORT=""
ENGINE_A_SYSTEM_PORT=8100; ENGINE_B_SYSTEM_PORT=8101

pass_count=0; fail_count=0
pass() { pass_count=$((pass_count+1)); echo "  PASS: $1"; }
fail() { fail_count=$((fail_count+1)); echo "  FAIL: $1"; }
strip_ansi() { sed 's/\x1b\[[0-9;]*m//g'; }
have() { cat "$1" 2>/dev/null | strip_ansi | grep -q "$2"; }

assert_log_contains() { have "$1" "$2" && pass "$3" || fail "$3 (missing: $2)"; }
assert_log_not_contains() { have "$1" "$2" && fail "$3 (unexpected: $2)" || pass "$3"; }

log_ts_to_epoch_ms() {
    local ts; ts=$(echo "$1" | grep -oP '\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z' | head -1)
    [ -z "$ts" ] && return 1
    local base="${ts%Z}" secs frac
    secs=$(date -u -d "${base}" +%s 2>/dev/null) || return 1
    frac=$(echo "$base" | grep -oP '\.\K\d+' | head -1); frac="${frac}000"; frac="${frac:0:3}"
    echo $(( secs*1000 + 10#$frac ))
}

cleanup() {
    echo ""; echo "=== Cleaning up ==="
    for pid in "$FRONTEND_PID" "$ENGINE_B_PID" "$ENGINE_A_PID" "${GMS_PIDS[@]}"; do
        [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && { kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; }
    done
    # vLLM EngineCore/worker subprocesses do NOT die with the dynamo.vllm parent —
    # sweep them by pattern (bracket trick avoids matching this script), then reap any
    # process still holding GPU memory so the next run starts on a clean slate.
    pkill -9 -f "dynamo[.]vllm"        2>/dev/null
    pkill -9 -f "dynamo[.]frontend"    2>/dev/null
    pkill -9 -f "[g]pu_memory_service" 2>/dev/null
    pkill -9 -f "[E]ngineCore"         2>/dev/null
    sleep 2
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
    rm -f /tmp/gms_*.sock 2>/dev/null
    echo "Logs saved in: $LOG_DIR"
    echo "=============================================="
    echo "  Results: $pass_count passed, $fail_count failed"
    echo "=============================================="
    [ "$fail_count" -gt 0 ] && exit 1 || exit 0
}
trap cleanup EXIT

wait_for() {  # wait_for <log> <pattern> <timeout_s> <pid> <desc>
    local log="$1" pat="$2" to="$3" pid="$4" desc="$5"
    for i in $(seq 1 "$to"); do
        have "$log" "$pat" && return 0
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: process died waiting for: $desc"; tail -40 "$log"; return 1
        fi
        sleep 1
    done
    echo "ERROR: timeout (${to}s) waiting for: $desc"; tail -40 "$log"; return 1
}

echo "=============================================="
echo "  Engine-Level GMS Shadow Failover Test (TP=$TP_SIZE)"
echo "=============================================="
echo "Model: $MODEL_NAME | Logs: $LOG_DIR | Lock: $LOCK_PATH"
echo "ETCD_ENDPOINTS=${ETCD_ENDPOINTS:-unset} NATS_SERVER=${NATS_SERVER:-unset}"
echo ""

# ---- Phase 0: GMS on each device ----
# Shadow mode (DYN_GMS_SCRATCH_KV_ENABLED=1) needs TWO GMS servers per device:
#   - tag "weights"  : VA-stable model weights, survives failover
#   - tag "kv_cache" : scratch KV pool, reclaimed/realloc'd on failover
# (worker.py wake_up does kv_cache_manager.connect(RW) against the kv_cache sock.)
echo "=== Phase 0: Start GPU Memory Service x2/device (devices 0..$((TP_SIZE-1))) ==="
for dev in $(seq 0 $((TP_SIZE-1))); do
    python3 -m gpu_memory_service --device "$dev" --tag weights  > "$LOG_DIR/gms_w${dev}.log"  2>&1 &
    GMS_PIDS+=($!)
    python3 -m gpu_memory_service --device "$dev" --tag kv_cache > "$LOG_DIR/gms_kv${dev}.log" 2>&1 &
    GMS_PIDS+=($!)
done
echo "GMS PIDs: ${GMS_PIDS[*]}"
for dev in $(seq 0 $((TP_SIZE-1))); do
    wait_for "$LOG_DIR/gms_w${dev}.log"  "Server started" 30 "" "GMS weights  dev $dev ready" || exit 1
    wait_for "$LOG_DIR/gms_kv${dev}.log" "Server started" 30 "" "GMS kv_cache dev $dev ready" || exit 1
    echo "GMS device $dev ready (weights + kv_cache)"
done

# ---- Phase 1: Deterministic weight loading (D5) ----
echo ""; echo "=== Phase 1: Deterministic Weight Loading ==="
echo "Starting Engine B (ENGINE_ID=1, RO) FIRST — should block until A commits"
ENGINE_ID=1 FAILOVER_LOCK_PATH="$LOCK_PATH" DYN_SYSTEM_PORT=$ENGINE_B_SYSTEM_PORT \
VLLM_NIXL_SIDE_CHANNEL_PORT=5601 DYN_VLLM_KV_EVENT_PORT=20081 \
python3 -m dynamo.vllm --model "$MODEL_NAME" -tp "$TP_SIZE" \
    --gpu-memory-utilization "${GPU_UTIL:-0.4}" \
    --load-format gms --gms-shadow-mode > "$ENGINE_B_LOG" 2>&1 &
ENGINE_B_PID=$!
echo "Engine B PID: $ENGINE_B_PID — waiting 20s"
sleep 20
kill -0 "$ENGINE_B_PID" 2>/dev/null || { echo "ERROR: Engine B died early"; cat "$ENGINE_B_LOG"; exit 1; }
# NOTE: the client-side "Connected with <ro> lock" line (session.py) is logged by the
# GMS client logger inside the vLLM worker subprocess and is NOT captured in the engine
# stdout here. We key D5 off observable markers instead: the *weights GMS server* commit
# line, and the engine [Shadow] lifecycle lines. Pre-commit, Engine B (RO) must still be
# blocked importing weights — i.e. it has NOT yet reached shadow STANDBY.
assert_log_not_contains "$ENGINE_B_LOG" "waiting for lock" \
    "D5: Engine B (RO) still blocked pre-commit (not yet in STANDBY)"

echo "Starting Engine A (ENGINE_ID=0, RW_OR_RO) — should load + commit weights"
ENGINE_ID=0 FAILOVER_LOCK_PATH="$LOCK_PATH" DYN_SYSTEM_PORT=$ENGINE_A_SYSTEM_PORT \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 DYN_VLLM_KV_EVENT_PORT=20080 \
python3 -m dynamo.vllm --model "$MODEL_NAME" -tp "$TP_SIZE" \
    --gpu-memory-utilization "${GPU_UTIL:-0.4}" \
    --load-format gms --gms-shadow-mode > "$ENGINE_A_LOG" 2>&1 &
ENGINE_A_PID=$!
echo "Engine A PID: $ENGINE_A_PID"
# Commit is observable on the weights GMS server (device 0): "Committed layout with state hash".
wait_for "$GMS_W0_LOG" "Committed layout with state hash" 300 "$ENGINE_A_PID" \
    "Engine A (RW writer) committed weights to GMS" || exit 1
assert_log_contains "$GMS_W0_LOG" "Committed layout with state hash" \
    "D5: weights committed to GMS by RW writer (Engine A)"
# After commit, Engine B (RO) imports the committed weights and proceeds to STANDBY.
wait_for "$ENGINE_B_LOG" "waiting for lock" 180 "$ENGINE_B_PID" "Engine B import + reach STANDBY" || exit 1
assert_log_contains "$ENGINE_B_LOG" "waiting for lock" \
    "D5: Engine B reached STANDBY after import (RO, post-commit)"

# ---- Phase 2: Lock-driven wake (D4) ----
echo ""; echo "=== Phase 2: Lock-Driven Wake ==="
for n in A B; do
    eval "log=\$ENGINE_${n}_LOG; pid=\$ENGINE_${n}_PID"
    wait_for "$log" "waiting for lock" 300 "$pid" "Engine $n reach STANDBY" || exit 1
    echo "Engine $n reached STANDBY"
done
# Scratch-KV engagement (D8): each shadow must have aliased its full KV via GMS
# scratch (VA-reserved, physically cheap) rather than fully backing it — that is
# what lets two engines colocate without ~2x KV OOM. worker.initialize_from_config
# emits a visible summary; the per-mapping "[GMS] Reserved ... scratch" INFO is
# suppressed in the vLLM worker subprocess, so this summary is the observable proof.
for n in A B; do
    eval "log=\$ENGINE_${n}_LOG"
    assert_log_contains "$log" "Scratch-KV engaged" "D8: Engine $n aliased its KV via GMS scratch"
done
echo "Waiting for a lock winner..."
for i in $(seq 1 120); do
    A_WOKE=$(cat "$ENGINE_A_LOG" 2>/dev/null | strip_ansi | grep -c "Lock acquired, waking engine")
    B_WOKE=$(cat "$ENGINE_B_LOG" 2>/dev/null | strip_ansi | grep -c "Lock acquired, waking engine")
    [ "$A_WOKE" -gt 0 ] || [ "$B_WOKE" -gt 0 ] && break
    [ "$i" -eq 120 ] && { echo "ERROR: no lock winner in 120s"; tail -20 "$ENGINE_A_LOG" "$ENGINE_B_LOG"; exit 1; }
    sleep 1
done
if [ "${A_WOKE:-0}" -gt 0 ]; then
    echo "Engine A won"; WINNER_PID=$ENGINE_A_PID WINNER_LOG=$ENGINE_A_LOG WINNER_PORT=$ENGINE_A_SYSTEM_PORT
    LOSER_PID=$ENGINE_B_PID LOSER_LOG=$ENGINE_B_LOG LOSER_PORT=$ENGINE_B_SYSTEM_PORT
else
    echo "Engine B won"; WINNER_PID=$ENGINE_B_PID WINNER_LOG=$ENGINE_B_LOG WINNER_PORT=$ENGINE_B_SYSTEM_PORT
    LOSER_PID=$ENGINE_A_PID LOSER_LOG=$ENGINE_A_LOG LOSER_PORT=$ENGINE_A_SYSTEM_PORT
fi
wait_for "$WINNER_LOG" "Engine awake, registering with discovery" 120 "$WINNER_PID" "winner register" || exit 1
sleep 5
assert_log_contains "$WINNER_LOG" "Lock acquired, waking engine" "D4: Winner acquired flock and auto-woke"
assert_log_not_contains "$LOSER_LOG" "Lock acquired" "D4: Loser still blocked on flock"

# ---- Phase 3: Health probes (D2) ----
echo ""; echo "=== Phase 3: Health Probes ==="
LH=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$LOSER_PORT/health" 2>/dev/null || echo 000)
[ "$LH" = "200" ] && pass "D2: Loser health 200 in STANDBY" || fail "D2: Loser health $LH (want 200)"
WH=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$WINNER_PORT/health" 2>/dev/null || echo 000)
[ "$WH" = "200" ] && pass "D2: Winner health 200 in ACTIVE" || fail "D2: Winner health $WH (want 200)"

# ---- Phase 4: Discovery + inference ----
echo ""; echo "=== Phase 4: Discovery & Inference ==="
python3 -m dynamo.frontend > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!
wait_for "$FRONTEND_LOG" "Completions is ready" 60 "$FRONTEND_PID" "frontend ready" || exit 1
assert_log_not_contains "$LOSER_LOG" "registering with discovery" "D7: Loser never registered with discovery"

infer() {
    curl -s -X POST http://localhost:8000/v1/completions -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL_NAME\",\"prompt\":\"The capital of France is\",\"max_tokens\":20,\"temperature\":0}"
}
R=$(infer)
if echo "$R" | grep -q '"choices"'; then
    pass "Inference on winner: $(echo "$R" | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null)"
else
    fail "Inference on winner failed: $R"
fi

# ---- Phase 5: Failover ----
echo ""; echo "=== Phase 5: Failover ==="
KILL_MS=$(date +%s%3N)
echo "Killing winner (PID $WINNER_PID) — SIGKILL whole process tree (crash model)..."
# Crash semantics (POC C1+C3): SIGKILL the winner's ENTIRE process tree bottom-up
# (parent + EngineCore + any TP workers), not just the parent. This (a) releases the
# flock instantly — the kernel drops it on any death, incl. SIGKILL — and (b) frees the
# winner's GPU/KV mappings and drops its GMS connection so the loser's KV realloc has room
# and no tag conflict. SIGTERM is deliberately avoided: graceful shutdown on this build
# (main @ vLLM 0.26) hangs at "Phase 3: backend services disconnect" and never releases
# the flock, so the standby can never wake. Killing only the parent (leaving the
# EngineCore orphaned) strands ~70 GiB of KV + a live GMS holder, which stalls the loser's
# realloc past its timeout — hence the full-tree kill.
kill_tree() {
    local _p=$1 _c
    for _c in $(pgrep -P "$_p" 2>/dev/null); do kill_tree "$_c"; done
    kill -9 "$_p" 2>/dev/null
}
kill_tree "$WINNER_PID"; wait "$WINNER_PID" 2>/dev/null
[ "$WINNER_PID" = "$ENGINE_A_PID" ] && ENGINE_A_PID="" || ENGINE_B_PID=""
WINNER_PID=""
wait_for "$LOSER_LOG" "Lock acquired, waking engine" 120 "$LOSER_PID" "loser auto-wake on lock release" || exit 1
echo "Loser acquired lock!"
wait_for "$LOSER_LOG" "Registered endpoint 'dynamo.backend.generate'" 120 "$LOSER_PID" "loser register generate" || exit 1
sleep 5
assert_log_contains "$LOSER_LOG" "Lock acquired, waking engine" "D4: Loser auto-woke via lock release"

LOCK_LINE=$(cat "$LOSER_LOG" | strip_ansi | grep -F "Lock acquired, waking engine" | tail -1)
REG_LINE=$(cat "$LOSER_LOG" | strip_ansi | grep -F "Registered endpoint 'dynamo.backend.generate'" | tail -1)
LOCK_MS=$(log_ts_to_epoch_ms "$LOCK_LINE" 2>/dev/null || echo "")
REG_MS=$(log_ts_to_epoch_ms "$REG_LINE" 2>/dev/null || echo "")
echo "=========================================="
echo "  FAILOVER TIMING  (winner SIGKILLed — crash model, no graceful drain)"
[ -n "$LOCK_MS" ] && echo "  Kill -> Lock acquired:       $((LOCK_MS-KILL_MS)) ms"
[ -n "$REG_MS" ]  && echo "  Kill -> Generate registered: $((REG_MS-KILL_MS)) ms"
echo "=========================================="

sleep 3
# Post-failover the woken engine needs a moment to be routable end-to-end
# (discovery propagation + scratch->real KV remap + first-token warmup). Retry
# with backoff instead of a single shot so a transient not-ready doesn't read as
# a failover regression (kills the single-shot false negative).
R=""; infer_ok=0; attempt=0
for attempt in $(seq 1 10); do
    R=$(infer)
    if echo "$R" | grep -q '"choices"'; then infer_ok=1; break; fi
    echo "  post-failover infer attempt $attempt not ready; retry in 2s..."
    sleep 2
done
if [ "$infer_ok" = "1" ]; then
    pass "Inference after failover (attempt $attempt): $(echo "$R" | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null)"
else
    fail "Inference after failover failed after $attempt attempts: $R"
fi
kill -0 "$LOSER_PID" 2>/dev/null && pass "D7: Exactly one engine alive after failover" || fail "D7: Loser not alive"

echo ""; echo "  TEST COMPLETE"
