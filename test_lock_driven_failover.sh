#!/bin/bash
# Lock-Driven Failover Test (TP=2)
#
# Validates:
#   D5  — Deterministic weight loading (ENGINE_ID=1 blocks until ENGINE_ID=0 commits)
#   D4  — Lock-driven auto-wake (flock release on process death triggers failover)
#   D2  — Health probe behavior (branch 3: 200 in STANDBY, 200 in ACTIVE)
#   D7  — Process death as fencing (only one engine registered at a time)
#
# Usage: ./test_lock_driven_failover.sh [MODEL_NAME]
# Default model: Qwen/Qwen3-0.6B

set -e

MODEL_NAME="${1:-Qwen/Qwen3-0.6B}"
TP_SIZE=2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_NAME="${VENV_NAME:-dynamo}"
source "${VENV_NAME}/bin/activate"
source .env

LOG_DIR="/tmp/failover_tp2_test_$$"
mkdir -p "$LOG_DIR"

LOCK_PATH="$LOG_DIR/failover.lock"

GMS0_LOG="$LOG_DIR/gms0.log"
GMS1_LOG="$LOG_DIR/gms1.log"
ENGINE_A_LOG="$LOG_DIR/engine_a.log"
ENGINE_B_LOG="$LOG_DIR/engine_b.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"

GMS0_PID=""
GMS1_PID=""
ENGINE_A_PID=""
ENGINE_B_PID=""
FRONTEND_PID=""
WINNER_PID=""
LOSER_PID=""
WINNER_LOG=""
LOSER_LOG=""
WINNER_PORT=""
LOSER_PORT=""

ENGINE_A_SYSTEM_PORT=8100
ENGINE_B_SYSTEM_PORT=8101

pass_count=0
fail_count=0

pass() {
    pass_count=$((pass_count + 1))
    echo "  PASS: $1"
}

fail() {
    fail_count=$((fail_count + 1))
    echo "  FAIL: $1"
}

assert_log_contains() {
    local log_file="$1" pattern="$2" description="$3"
    if grep -q "$pattern" "$log_file" 2>/dev/null; then
        pass "$description"
    else
        fail "$description (pattern not found: $pattern)"
    fi
}

assert_log_not_contains() {
    local log_file="$1" pattern="$2" description="$3"
    if grep -q "$pattern" "$log_file" 2>/dev/null; then
        fail "$description (pattern unexpectedly found: $pattern)"
    else
        pass "$description"
    fi
}

strip_ansi() {
    sed 's/\x1b\[[0-9;]*m//g'
}

cleanup() {
    echo ""
    echo "=== Cleaning up ==="
    for pid_var in FRONTEND_PID ENGINE_B_PID ENGINE_A_PID GMS1_PID GMS0_PID; do
        pid="${!pid_var}"
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "Killing $pid_var (PID: $pid)"
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    echo "Logs saved in: $LOG_DIR"
    echo ""
    echo "=============================================="
    echo "  Results: $pass_count passed, $fail_count failed"
    echo "=============================================="
    if [ "$fail_count" -gt 0 ]; then
        exit 1
    fi
}

trap cleanup EXIT

echo "=============================================="
echo "  Lock-Driven Failover Test (TP=$TP_SIZE)"
echo "=============================================="
echo "Model: $MODEL_NAME"
echo "Log directory: $LOG_DIR"
echo "Lock path: $LOCK_PATH"
echo ""

# ============================================================
# Phase 0: Start GMS on both devices
# ============================================================
echo "=== Phase 0: Starting GPU Memory Service (device 0 & 1) ==="

python3 -m gpu_memory_service --device 0 > "$GMS0_LOG" 2>&1 &
GMS0_PID=$!
python3 -m gpu_memory_service --device 1 > "$GMS1_LOG" 2>&1 &
GMS1_PID=$!
echo "GMS PIDs: device0=$GMS0_PID, device1=$GMS1_PID"

for dev in 0 1; do
    log_var="GMS${dev}_LOG"
    log_file="${!log_var}"
    for i in $(seq 1 30); do
        if grep -q "waiting for connections" "$log_file" 2>/dev/null; then
            echo "GMS device $dev ready"
            break
        fi
        if [ "$i" -eq 30 ]; then
            echo "ERROR: GMS device $dev failed to start"
            cat "$log_file"
            exit 1
        fi
        sleep 1
    done
done

# ============================================================
# Phase 1: Deterministic weight loading (D5)
# Start Engine B (RO) first, then Engine A (RW_OR_RO).
# Engine B must block until Engine A commits.
# ============================================================
echo ""
echo "=== Phase 1: Deterministic Weight Loading ==="
echo "Starting Engine B (ENGINE_ID=1, RO) FIRST — it should block"

ENGINE_ID=1 \
FAILOVER_LOCK_PATH="$LOCK_PATH" \
DYN_SYSTEM_PORT=$ENGINE_B_SYSTEM_PORT \
VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
DYN_VLLM_KV_EVENT_PORT=20081 \
python3 -m dynamo.vllm \
    --model "$MODEL_NAME" \
    -tp "$TP_SIZE" \
    --load-format gms \
    --gms-mode shadow \
    > "$ENGINE_B_LOG" 2>&1 &
ENGINE_B_PID=$!
echo "Engine B PID: $ENGINE_B_PID"

echo "Waiting 20s for Engine B to block on RO lock..."
sleep 20

if ! kill -0 "$ENGINE_B_PID" 2>/dev/null; then
    echo "ERROR: Engine B died before Engine A started"
    cat "$ENGINE_B_LOG"
    exit 1
fi

assert_log_not_contains "$ENGINE_B_LOG" "Connected with ro lock" \
    "D5: Engine B (RO) blocked — no lock granted yet (no committed weights)"

echo ""
echo "Starting Engine A (ENGINE_ID=0, RW_OR_RO) — it should load weights"

ENGINE_ID=0 \
FAILOVER_LOCK_PATH="$LOCK_PATH" \
DYN_SYSTEM_PORT=$ENGINE_A_SYSTEM_PORT \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
DYN_VLLM_KV_EVENT_PORT=20080 \
python3 -m dynamo.vllm \
    --model "$MODEL_NAME" \
    -tp "$TP_SIZE" \
    --load-format gms \
    --gms-mode shadow \
    > "$ENGINE_A_LOG" 2>&1 &
ENGINE_A_PID=$!
echo "Engine A PID: $ENGINE_A_PID"

echo "Waiting for Engine A to commit weights..."
for i in $(seq 1 300); do
    if cat "$ENGINE_A_LOG" 2>/dev/null | strip_ansi | grep -q "Committed weights"; then
        echo "Engine A committed weights"
        break
    fi
    if ! kill -0 "$ENGINE_A_PID" 2>/dev/null; then
        echo "ERROR: Engine A died during weight loading"
        tail -50 "$ENGINE_A_LOG"
        exit 1
    fi
    if [ "$i" -eq 300 ]; then
        echo "ERROR: Engine A did not commit weights within 300s"
        tail -50 "$ENGINE_A_LOG"
        exit 1
    fi
    sleep 1
done

assert_log_contains "$ENGINE_A_LOG" "Connected with rw_or_ro lock (granted=rw)" \
    "D5: Engine A got RW lock (first writer)"

echo "Waiting for Engine B to unblock (get RO after commit)..."
for i in $(seq 1 120); do
    if cat "$ENGINE_B_LOG" 2>/dev/null | strip_ansi | grep -q "Connected with ro lock"; then
        echo "Engine B unblocked"
        break
    fi
    if ! kill -0 "$ENGINE_B_PID" 2>/dev/null; then
        echo "ERROR: Engine B died while waiting for RO"
        tail -50 "$ENGINE_B_LOG"
        exit 1
    fi
    if [ "$i" -eq 120 ]; then
        echo "ERROR: Engine B did not get RO lock within 120s after commit"
        tail -50 "$ENGINE_B_LOG"
        exit 1
    fi
    sleep 1
done

assert_log_contains "$ENGINE_B_LOG" "Connected with ro lock (granted=ro), committed=True" \
    "D5: Engine B got RO lock with committed=True (import, not load)"

# ============================================================
# Phase 2: Lock-driven wake (D4)
# Both engines sleep and race for the flock.
# Exactly one should win.
# ============================================================
echo ""
echo "=== Phase 2: Lock-Driven Wake ==="
echo "Waiting for both engines to reach STANDBY (sleeping, waiting for lock)..."

for engine_name in "Engine A" "Engine B"; do
    if [ "$engine_name" = "Engine A" ]; then
        log="$ENGINE_A_LOG"; pid=$ENGINE_A_PID
    else
        log="$ENGINE_B_LOG"; pid=$ENGINE_B_PID
    fi
    for i in $(seq 1 300); do
        if cat "$log" 2>/dev/null | strip_ansi | grep -q "waiting for lock"; then
            echo "$engine_name reached STANDBY"
            break
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: $engine_name died before reaching STANDBY"
            tail -50 "$log"
            exit 1
        fi
        if [ "$i" -eq 300 ]; then
            echo "ERROR: $engine_name did not reach STANDBY within 300s"
            tail -50 "$log"
            exit 1
        fi
        sleep 1
    done
done

echo "Waiting for lock winner to wake and register..."
for i in $(seq 1 120); do
    A_WOKE=$(cat "$ENGINE_A_LOG" 2>/dev/null | strip_ansi | grep -c "Lock acquired, waking engine" || echo 0)
    B_WOKE=$(cat "$ENGINE_B_LOG" 2>/dev/null | strip_ansi | grep -c "Lock acquired, waking engine" || echo 0)
    if [ "$A_WOKE" -gt 0 ] || [ "$B_WOKE" -gt 0 ]; then
        break
    fi
    if [ "$i" -eq 120 ]; then
        echo "ERROR: Neither engine acquired the lock within 120s"
        tail -20 "$ENGINE_A_LOG"
        tail -20 "$ENGINE_B_LOG"
        exit 1
    fi
    sleep 1
done

if [ "$A_WOKE" -gt 0 ]; then
    echo "Engine A won the lock"
    WINNER_PID=$ENGINE_A_PID; WINNER_LOG=$ENGINE_A_LOG; WINNER_PORT=$ENGINE_A_SYSTEM_PORT
    LOSER_PID=$ENGINE_B_PID;  LOSER_LOG=$ENGINE_B_LOG;  LOSER_PORT=$ENGINE_B_SYSTEM_PORT
else
    echo "Engine B won the lock"
    WINNER_PID=$ENGINE_B_PID; WINNER_LOG=$ENGINE_B_LOG; WINNER_PORT=$ENGINE_B_SYSTEM_PORT
    LOSER_PID=$ENGINE_A_PID;  LOSER_LOG=$ENGINE_A_LOG;  LOSER_PORT=$ENGINE_A_SYSTEM_PORT
fi

echo "Waiting for winner to finish registration..."
for i in $(seq 1 120); do
    if cat "$WINNER_LOG" 2>/dev/null | strip_ansi | grep -q "Engine awake, registering with discovery"; then
        break
    fi
    if [ "$i" -eq 120 ]; then
        echo "ERROR: Winner did not register within 120s"
        tail -30 "$WINNER_LOG"
        exit 1
    fi
    sleep 1
done

# Give serve_endpoint time to start
sleep 5

assert_log_contains "$WINNER_LOG" "Lock acquired, waking engine" \
    "D4: Winner acquired flock and auto-woke"

assert_log_not_contains "$LOSER_LOG" "Lock acquired" \
    "D4: Loser still blocked on flock"

# ============================================================
# Phase 3: Health probe validation (D2)
# ============================================================
echo ""
echo "=== Phase 3: Health Probe Validation ==="

LOSER_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$LOSER_PORT/health" 2>/dev/null || echo "000")
if [ "$LOSER_HEALTH" = "200" ]; then
    pass "D2: Loser health probe returns 200 in STANDBY (branch 3)"
else
    fail "D2: Loser health probe returned $LOSER_HEALTH, expected 200 in STANDBY"
fi

WINNER_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$WINNER_PORT/health" 2>/dev/null || echo "000")
if [ "$WINNER_HEALTH" = "200" ]; then
    pass "D2: Winner health probe returns 200 in ACTIVE"
else
    fail "D2: Winner health probe returned $WINNER_HEALTH, expected 200 in ACTIVE"
fi

# ============================================================
# Phase 4: Discovery single-engine invariant + inference
# ============================================================
echo ""
echo "=== Phase 4: Discovery & Inference ==="

echo "Starting Frontend..."
python3 -m dynamo.frontend > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

for i in $(seq 1 60); do
    if grep -q "Completions is ready" "$FRONTEND_LOG" 2>/dev/null; then
        echo "Frontend ready"
        break
    fi
    if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
        echo "ERROR: Frontend died"
        cat "$FRONTEND_LOG"
        exit 1
    fi
    if [ "$i" -eq 60 ]; then
        echo "ERROR: Frontend not ready within 60s"
        tail -30 "$FRONTEND_LOG"
        exit 1
    fi
    sleep 1
done

assert_log_not_contains "$LOSER_LOG" "registering with discovery" \
    "D7: Loser never registered with discovery"

INFERENCE_RESPONSE=$(curl -s -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL_NAME\",
        \"prompt\": \"The capital of France is\",
        \"max_tokens\": 20,
        \"temperature\": 0
    }")

if echo "$INFERENCE_RESPONSE" | grep -q '"choices"'; then
    GENERATED=$(echo "$INFERENCE_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null)
    pass "Inference on winner succeeded: $GENERATED"
else
    fail "Inference on winner failed: $INFERENCE_RESPONSE"
fi

# ============================================================
# Phase 5: Failover (D4 fencing, D7 process-death-as-release)
# ============================================================
echo ""
echo "=== Phase 5: Failover ==="

FAILOVER_START_NS=$(date +%s%N)

echo "Killing winner (PID: $WINNER_PID)..."
kill "$WINNER_PID" 2>/dev/null || true
wait "$WINNER_PID" 2>/dev/null || true

# Clear the PID variable so cleanup doesn't try to kill it again
if [ "$WINNER_PID" = "$ENGINE_A_PID" ]; then ENGINE_A_PID=""; else ENGINE_B_PID=""; fi
WINNER_PID=""

echo "Winner killed. Waiting for loser to auto-wake via lock release..."

for i in $(seq 1 120); do
    if cat "$LOSER_LOG" 2>/dev/null | strip_ansi | grep -q "Lock acquired, waking engine"; then
        echo "Loser acquired lock!"
        break
    fi
    if ! kill -0 "$LOSER_PID" 2>/dev/null; then
        echo "ERROR: Loser died during failover"
        tail -50 "$LOSER_LOG"
        exit 1
    fi
    if [ "$i" -eq 120 ]; then
        echo "ERROR: Loser did not acquire lock within 120s"
        tail -50 "$LOSER_LOG"
        exit 1
    fi
    sleep 1
done

echo "Waiting for loser to register with discovery..."
for i in $(seq 1 120); do
    if cat "$LOSER_LOG" 2>/dev/null | strip_ansi | grep -q "Engine awake, registering with discovery"; then
        break
    fi
    if [ "$i" -eq 120 ]; then
        echo "ERROR: Loser did not register within 120s"
        tail -30 "$LOSER_LOG"
        exit 1
    fi
    sleep 1
done

# Wait for serve_endpoint + discovery propagation
sleep 5

FAILOVER_END_NS=$(date +%s%N)
FAILOVER_MS=$(( (FAILOVER_END_NS - FAILOVER_START_NS) / 1000000 ))

assert_log_contains "$LOSER_LOG" "Lock acquired, waking engine" \
    "D4: Loser auto-woke via lock release (no HTTP wake)"

echo ""
echo "=========================================="
echo "  FAILOVER TIME: ${FAILOVER_MS} ms"
echo "=========================================="

# Inference on new active engine (former loser)
echo ""
echo "Testing inference after failover..."
sleep 3

INFERENCE_RESPONSE=$(curl -s -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL_NAME\",
        \"prompt\": \"The capital of France is\",
        \"max_tokens\": 20,
        \"temperature\": 0
    }")

if echo "$INFERENCE_RESPONSE" | grep -q '"choices"'; then
    GENERATED=$(echo "$INFERENCE_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null)
    pass "Inference after failover succeeded: $GENERATED"
else
    fail "Inference after failover failed: $INFERENCE_RESPONSE"
fi

# Verify only one engine alive
if kill -0 "$LOSER_PID" 2>/dev/null; then
    pass "D7: Exactly one engine alive after failover"
else
    fail "D7: Loser engine is not alive after failover"
fi

echo ""
echo "=============================================="
echo "  TEST COMPLETE"
echo "=============================================="
echo "Summary:"
echo "  - Engine B (RO) blocked until Engine A (RW) committed weights"
echo "  - Both engines slept, raced for flock"
echo "  - Winner auto-woke, served inference"
echo "  - Loser health probe returned 200 while sleeping (STANDBY)"
echo "  - Kill winner → loser auto-woke via lock release"
echo "  - Failover time: ${FAILOVER_MS} ms"
echo "  - Inference after failover: OK"
echo ""
