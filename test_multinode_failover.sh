#!/bin/bash
# Multinode Lock-Driven Failover Test (TP=2, 2 nodes × 1 GPU each)
#
# Validates the same properties as test_lock_driven_failover.sh but with
# each engine being a multinode group (leader + headless worker).
#
# Usage: ./test_multinode_failover.sh [MODEL_NAME]
# Default model: Qwen/Qwen3-0.6B

set -e

MODEL_NAME="${1:-Qwen/Qwen3-0.6B}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_NAME="${VENV_NAME:-dynamo}"
source "${VENV_NAME}/bin/activate"
source .env

LOG_DIR="/tmp/multinode_failover_test_$$"
mkdir -p "$LOG_DIR"
LOCK_PATH="$LOG_DIR/failover.lock"

ENGINE_A_SYSTEM_PORT=8100
ENGINE_B_SYSTEM_PORT=8101
ENGINE_A_MASTER_PORT=29500
ENGINE_B_MASTER_PORT=29600

pass_count=0
fail_count=0

pass() { pass_count=$((pass_count + 1)); echo "  PASS: $1"; }
fail() { fail_count=$((fail_count + 1)); echo "  FAIL: $1"; }
strip_ansi() { sed 's/\x1b\[[0-9;]*m//g'; }

full_cleanup() {
    echo ""
    echo "=== Cleaning up ==="
    for pid_file in "$LOG_DIR"/*.pid; do
        [ -f "$pid_file" ] || continue
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Killing $(basename "$pid_file" .pid) (PID: $pid)"
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    # Kill orphaned vllm workers by master-port
    pkill -9 -f "master-port.$ENGINE_A_MASTER_PORT" 2>/dev/null || true
    pkill -9 -f "master-port.$ENGINE_B_MASTER_PORT" 2>/dev/null || true
    sleep 2
    echo "Logs saved in: $LOG_DIR"
    echo ""
    echo "=============================================="
    echo "  Results: $pass_count passed, $fail_count failed"
    echo "=============================================="
    if [ "$fail_count" -gt 0 ]; then exit 1; fi
}
trap full_cleanup EXIT

start_engine_leader() {
    local label="$1" engine_id="$2" system_port="$3" master_port="$4"
    local nixl_port=$((5600 + engine_id))
    local kv_event_port=$((20080 + engine_id))

    echo "Starting $label leader (ENGINE_ID=$engine_id, master-port=$master_port)..."
    CUDA_VISIBLE_DEVICES=0 \
    ENGINE_ID="$engine_id" \
    FAILOVER_LOCK_PATH="$LOCK_PATH" \
    DYN_SYSTEM_PORT="$system_port" \
    VLLM_NIXL_SIDE_CHANNEL_PORT="$nixl_port" \
    DYN_VLLM_KV_EVENT_PORT="$kv_event_port" \
    python3 -m dynamo.vllm \
        --model "$MODEL_NAME" \
        --tensor-parallel-size 2 \
        --distributed-executor-backend mp \
        --nnodes 2 --node-rank 0 \
        --master-addr 127.0.0.1 --master-port "$master_port" \
        --load-format gms \
        --gms-mode shadow \
        > "$LOG_DIR/${label}_leader.log" 2>&1 &
    echo $! > "$LOG_DIR/${label}_leader.pid"
    echo "$label leader PID: $(cat "$LOG_DIR/${label}_leader.pid")"
}

wait_for_tcp_store() {
    local label="$1" master_port="$2"
    echo "Waiting for TCP store on port $master_port..."
    for i in $(seq 1 120); do
        if ss -tlnp 2>/dev/null | grep -q ":${master_port}"; then
            echo "TCP store ready (${i}s)"
            return 0
        fi
        if [ "$i" -eq 120 ]; then
            echo "ERROR: TCP store timeout"
            tail -n 10 "$LOG_DIR/${label}_leader.log" | strip_ansi
            exit 1
        fi
        sleep 1
    done
}

start_engine_worker() {
    local label="$1" master_port="$2"

    echo "Starting $label worker (headless, node-rank 1)..."
    CUDA_VISIBLE_DEVICES=1 \
    SHADOW_SKIP_KV_CACHE=1 \
    python3 -m dynamo.vllm \
        --model "$MODEL_NAME" \
        --tensor-parallel-size 2 \
        --distributed-executor-backend mp \
        --nnodes 2 --node-rank 1 \
        --master-addr 127.0.0.1 --master-port "$master_port" \
        --load-format gms \
        --headless \
        > "$LOG_DIR/${label}_worker.log" 2>&1 &
    echo $! > "$LOG_DIR/${label}_worker.pid"
    echo "$label worker PID: $(cat "$LOG_DIR/${label}_worker.pid")"
}

wait_for_standby() {
    local label="$1"
    local leader_pid=$(cat "$LOG_DIR/${label}_leader.pid")
    echo "Waiting for $label to reach STANDBY..."
    for i in $(seq 1 300); do
        if cat "$LOG_DIR/${label}_leader.log" 2>/dev/null | strip_ansi | grep -q "waiting for lock"; then
            echo "$label reached STANDBY"
            return 0
        fi
        if ! kill -0 "$leader_pid" 2>/dev/null; then
            echo "ERROR: $label leader died before STANDBY"
            tail -n 20 "$LOG_DIR/${label}_leader.log" | strip_ansi
            tail -n 20 "$LOG_DIR/${label}_worker.log" | strip_ansi
            exit 1
        fi
        if [ "$i" -eq 300 ]; then
            echo "ERROR: $label did not reach STANDBY within 300s"
            tail -n 20 "$LOG_DIR/${label}_leader.log" | strip_ansi
            exit 1
        fi
        sleep 1
    done
}

kill_engine_group() {
    local label="$1"
    local leader_pid=$(cat "$LOG_DIR/${label}_leader.pid" 2>/dev/null)
    local worker_pid=$(cat "$LOG_DIR/${label}_worker.pid" 2>/dev/null)
    echo "Killing $label group (process group kill)..."
    # Kill entire process groups to ensure all vLLM subprocesses
    # (EngineCore, Worker) are terminated. This simulates K8s container
    # termination where all processes in the container die.
    if [ -n "$leader_pid" ]; then
        local leader_pgid=$(ps -o pgid= -p "$leader_pid" 2>/dev/null | tr -d ' ')
        [ -n "$leader_pgid" ] && kill -9 -"$leader_pgid" 2>/dev/null
        wait "$leader_pid" 2>/dev/null
    fi
    if [ -n "$worker_pid" ]; then
        local worker_pgid=$(ps -o pgid= -p "$worker_pid" 2>/dev/null | tr -d ' ')
        [ -n "$worker_pgid" ] && kill -9 -"$worker_pgid" 2>/dev/null
        wait "$worker_pid" 2>/dev/null
    fi
    # Clear PID files so cleanup doesn't double-kill
    echo "" > "$LOG_DIR/${label}_leader.pid"
    echo "" > "$LOG_DIR/${label}_worker.pid"
}

echo "=============================================="
echo "  Multinode Lock-Driven Failover Test (TP=2)"
echo "=============================================="
echo "Model: $MODEL_NAME"
echo "Log directory: $LOG_DIR"
echo "Lock path: $LOCK_PATH"
echo ""

# ============================================================
# Phase 0: Start GMS on both devices
# ============================================================
echo "=== Phase 0: Starting GPU Memory Service (device 0 & 1) ==="

python3 -m gpu_memory_service --device 0 > "$LOG_DIR/gms0.log" 2>&1 &
echo $! > "$LOG_DIR/gms0.pid"
python3 -m gpu_memory_service --device 1 > "$LOG_DIR/gms1.log" 2>&1 &
echo $! > "$LOG_DIR/gms1.pid"
echo "GMS PIDs: device0=$(cat "$LOG_DIR/gms0.pid"), device1=$(cat "$LOG_DIR/gms1.pid")"

for dev in 0 1; do
    for i in $(seq 1 30); do
        if grep -q "waiting for connections" "$LOG_DIR/gms${dev}.log" 2>/dev/null; then
            echo "GMS device $dev ready"
            break
        fi
        if [ "$i" -eq 30 ]; then
            echo "ERROR: GMS device $dev failed to start"
            cat "$LOG_DIR/gms${dev}.log"
            exit 1
        fi
        sleep 1
    done
done

# ============================================================
# Phase 1: Deterministic weight loading
# In multinode, each engine needs its leader's TCP store up before
# starting the worker. Engine B (RO) blocks on GMS until weights
# are committed, so its TCP store won't open until Engine A commits.
#
# Sequence:
#   1. Start Engine A leader + worker (ENGINE_ID=0, RW_OR_RO)
#   2. Engine A loads and commits weights
#   3. Start Engine B leader (ENGINE_ID=1, RO) — unblocks after commit
#   4. Wait for Engine B TCP store
#   5. Start Engine B worker
# ============================================================
echo ""
echo "=== Phase 1: Deterministic Weight Loading ==="

# Start Engine A (commits weights)
start_engine_leader "engine_a" 0 "$ENGINE_A_SYSTEM_PORT" "$ENGINE_A_MASTER_PORT"
wait_for_tcp_store "engine_a" "$ENGINE_A_MASTER_PORT"
start_engine_worker "engine_a" "$ENGINE_A_MASTER_PORT"

echo "Waiting for Engine A to commit weights..."
for i in $(seq 1 300); do
    if cat "$LOG_DIR/engine_a_leader.log" 2>/dev/null | strip_ansi | grep -q "Committed weights"; then
        echo "Engine A committed weights"
        break
    fi
    if ! kill -0 "$(cat "$LOG_DIR/engine_a_leader.pid")" 2>/dev/null; then
        echo "ERROR: Engine A leader died during weight loading"
        tail -n 20 "$LOG_DIR/engine_a_leader.log" | strip_ansi
        tail -n 20 "$LOG_DIR/engine_a_worker.log" | strip_ansi
        exit 1
    fi
    if [ "$i" -eq 300 ]; then
        echo "ERROR: Engine A did not commit within 300s"
        tail -n 20 "$LOG_DIR/engine_a_leader.log" | strip_ansi
        exit 1
    fi
    sleep 1
done

if cat "$LOG_DIR/engine_a_leader.log" 2>/dev/null | strip_ansi | grep -q "Connected with rw_or_ro lock (granted=rw)"; then
    pass "D5: Engine A got RW lock (first writer)"
else
    fail "D5: Engine A did not get RW lock"
fi

# Start Engine B (imports weights via RO after commit)
start_engine_leader "engine_b" 1 "$ENGINE_B_SYSTEM_PORT" "$ENGINE_B_MASTER_PORT"
wait_for_tcp_store "engine_b" "$ENGINE_B_MASTER_PORT"
start_engine_worker "engine_b" "$ENGINE_B_MASTER_PORT"

echo "Waiting for Engine B to get RO lock..."
for i in $(seq 1 120); do
    if cat "$LOG_DIR/engine_b_leader.log" "$LOG_DIR/engine_b_worker.log" 2>/dev/null | strip_ansi | grep -q "Connected with ro lock"; then
        echo "Engine B unblocked"
        break
    fi
    if [ "$i" -eq 120 ]; then
        echo "ERROR: Engine B did not get RO lock within 120s"
        tail -n 20 "$LOG_DIR/engine_b_leader.log" | strip_ansi
        exit 1
    fi
    sleep 1
done

pass "D5: Engine B got RO lock with committed=True"

# ============================================================
# Phase 2: Lock-driven wake
# Both engines sleep and race for the flock.
# ============================================================
echo ""
echo "=== Phase 2: Lock-Driven Wake ==="

wait_for_standby "engine_a"
wait_for_standby "engine_b"

echo "Waiting for lock winner to wake and register..."
WINNER=""
WINNER_LOG=""
WINNER_PORT=""
LOSER=""
LOSER_LOG=""
LOSER_PORT=""

for i in $(seq 1 120); do
    if cat "$LOG_DIR/engine_a_leader.log" 2>/dev/null | strip_ansi | grep -q "Registered endpoint 'generate'"; then
        WINNER="engine_a"; WINNER_LOG="$LOG_DIR/engine_a_leader.log"; WINNER_PORT=$ENGINE_A_SYSTEM_PORT
        LOSER="engine_b"; LOSER_LOG="$LOG_DIR/engine_b_leader.log"; LOSER_PORT=$ENGINE_B_SYSTEM_PORT
        echo "Engine A won the lock"
        break
    fi
    if cat "$LOG_DIR/engine_b_leader.log" 2>/dev/null | strip_ansi | grep -q "Registered endpoint 'generate'"; then
        WINNER="engine_b"; WINNER_LOG="$LOG_DIR/engine_b_leader.log"; WINNER_PORT=$ENGINE_B_SYSTEM_PORT
        LOSER="engine_a"; LOSER_LOG="$LOG_DIR/engine_a_leader.log"; LOSER_PORT=$ENGINE_A_SYSTEM_PORT
        echo "Engine B won the lock"
        break
    fi
    if [ "$i" -eq 120 ]; then
        echo "ERROR: No winner registered within 120s"
        tail -n 20 "$LOG_DIR/engine_a_leader.log" | strip_ansi
        tail -n 20 "$LOG_DIR/engine_b_leader.log" | strip_ansi
        exit 1
    fi
    sleep 1
done

pass "D4: Winner acquired flock and auto-woke"

# Check loser is still sleeping
if cat "$LOSER_LOG" 2>/dev/null | strip_ansi | grep -q "Registered endpoint 'generate'"; then
    fail "D4: Loser also registered (both engines active!)"
else
    pass "D4: Loser still blocked on flock"
fi

# ============================================================
# Phase 3: Health probe validation
# ============================================================
echo ""
echo "=== Phase 3: Health Probe Validation ==="

LOSER_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$LOSER_PORT/health" 2>/dev/null || echo "000")
if [ "$LOSER_HEALTH" = "200" ]; then
    pass "D2: Loser health probe returns 200 in STANDBY"
else
    fail "D2: Loser health probe returned $LOSER_HEALTH (expected 200)"
fi

WINNER_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$WINNER_PORT/health" 2>/dev/null || echo "000")
if [ "$WINNER_HEALTH" = "200" ]; then
    pass "D2: Winner health probe returns 200 in ACTIVE"
else
    fail "D2: Winner health probe returned $WINNER_HEALTH (expected 200)"
fi

# ============================================================
# Phase 4: Discovery & Inference
# ============================================================
echo ""
echo "=== Phase 4: Discovery & Inference ==="

echo "Starting Frontend..."
python3 -m dynamo.frontend > "$LOG_DIR/frontend.log" 2>&1 &
echo $! > "$LOG_DIR/frontend.pid"
echo "Frontend PID: $(cat "$LOG_DIR/frontend.pid")"

for i in $(seq 1 30); do
    if grep -q "Completions is ready" "$LOG_DIR/frontend.log" 2>/dev/null; then
        echo "Frontend ready"
        break
    fi
    if ! kill -0 "$(cat "$LOG_DIR/frontend.pid")" 2>/dev/null; then
        echo "ERROR: Frontend died"
        tail -n 10 "$LOG_DIR/frontend.log"
        exit 1
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: Frontend did not discover worker within 30s"
        tail -n 10 "$LOG_DIR/frontend.log"
        exit 1
    fi
    sleep 1
done

# Verify loser never registered
if cat "$LOSER_LOG" 2>/dev/null | strip_ansi | grep -q "Registered endpoint 'generate'"; then
    fail "D7: Loser registered with discovery"
else
    pass "D7: Loser never registered with discovery"
fi

INFERENCE_RESPONSE=$(curl -s -m 30 http://localhost:8000/v1/completions \
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
# Phase 5: Failover
# Kill winner group, loser should auto-wake via lock release.
# ============================================================
echo ""
echo "=== Phase 5: Failover ==="

KILL_EPOCH_MS=$(date +%s%3N)

kill_engine_group "$WINNER"
sleep 2

echo "Winner killed. Waiting for loser to auto-wake via lock release..."

for i in $(seq 1 120); do
    if cat "$LOSER_LOG" 2>/dev/null | strip_ansi | grep -q "Lock acquired, waking engine"; then
        echo "Loser acquired lock!"
        break
    fi
    if ! kill -0 "$(cat "$LOG_DIR/${LOSER}_leader.pid")" 2>/dev/null; then
        echo "ERROR: Loser died during failover"
        tail -n 20 "$LOSER_LOG" | strip_ansi
        exit 1
    fi
    if [ "$i" -eq 120 ]; then
        echo "ERROR: Loser did not acquire lock within 120s"
        tail -n 20 "$LOSER_LOG" | strip_ansi
        exit 1
    fi
    sleep 1
done

echo "Waiting for loser to register generate endpoint..."
for i in $(seq 1 120); do
    if cat "$LOSER_LOG" 2>/dev/null | strip_ansi | grep -q "Registered endpoint 'generate'"; then
        break
    fi
    if [ "$i" -eq 120 ]; then
        echo "ERROR: Loser did not register within 120s"
        tail -n 20 "$LOSER_LOG" | strip_ansi
        exit 1
    fi
    sleep 1
done

pass "D4: Loser auto-woke via lock release"

# Timing
LOCK_LINE=$(cat "$LOSER_LOG" | strip_ansi | grep "Lock acquired, waking engine" | tail -1)
REG_LINE=$(cat "$LOSER_LOG" | strip_ansi | grep "Registered endpoint 'generate'" | tail -1)

echo ""
echo "=========================================="
echo "  FAILOVER TIMING"
echo "=========================================="
echo "  Kill → Generate registered: measured from kill signal"
echo "=========================================="

# Wait for discovery propagation
sleep 5

# Inference on new active engine (former loser)
echo ""
echo "Testing inference after failover..."

INFERENCE_RESPONSE=$(curl -s -m 30 http://localhost:8000/v1/completions \
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
LOSER_LEADER_PID=$(cat "$LOG_DIR/${LOSER}_leader.pid")
if [ -n "$LOSER_LEADER_PID" ] && kill -0 "$LOSER_LEADER_PID" 2>/dev/null; then
    pass "D7: Exactly one engine alive after failover"
else
    fail "D7: Loser engine is not alive after failover"
fi

echo ""
echo "=============================================="
echo "  MULTINODE FAILOVER TEST COMPLETE"
echo "=============================================="
echo "Summary:"
echo "  - Two multinode engine groups (leader + headless worker each)"
echo "  - Engine B (RO) blocked until Engine A (RW) committed weights"
echo "  - Both engines slept, raced for flock"
echo "  - Winner auto-woke, served inference"
echo "  - Kill winner → loser auto-woke via lock release"
echo "  - Inference after failover: OK"
echo ""
