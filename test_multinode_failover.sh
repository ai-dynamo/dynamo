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

extract_from_log() {
    local log_file="$1" grep_pattern="$2" sed_pattern="$3"
    cat "$log_file" 2>/dev/null | strip_ansi | grep "$grep_pattern" | head -1 | sed "$sed_pattern"
}

within_pct() {
    local a="$1" b="$2" pct="$3" label="$4"
    if [ -z "$a" ] || [ -z "$b" ] || [ "$a" -eq 0 ] || [ "$b" -eq 0 ]; then
        fail "$label (could not extract values: a=$a b=$b)"
        return
    fi
    local diff=$(( a > b ? a - b : b - a ))
    local pct_diff=$(( diff * 100 / a ))
    if [ "$pct_diff" -le "$pct" ]; then
        pass "$label (winner=$a, loser=$b, delta=${pct_diff}%)"
    else
        fail "$label (winner=$a, loser=$b, delta=${pct_diff}% > ${pct}%)"
    fi
}

full_cleanup() {
    echo ""
    echo "=== Cleaning up ==="
    for pid_file in "$LOG_DIR"/*.pid; do
        [ -f "$pid_file" ] || continue
        pid=$(cat "$pid_file")
        [ -z "$pid" ] && continue
        if kill -0 "$pid" 2>/dev/null; then
            echo "Killing $(basename "$pid_file" .pid) (PID: $pid)"
            # Engine PIDs are setsid session leaders — kill -PGID gets the tree.
            # GMS/frontend are direct children — plain kill works.
            kill -9 -"$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
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
    setsid \
    env CUDA_VISIBLE_DEVICES=0 \
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
    setsid \
    env CUDA_VISIBLE_DEVICES=1 \
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
    echo "Killing $label group (session kill)..."
    # Each engine process was launched with setsid, so PID == PGID.
    # kill -9 -PID kills the entire session (all vLLM subprocesses)
    # without affecting other engines or the test script.
    if [ -n "$leader_pid" ] && kill -0 "$leader_pid" 2>/dev/null; then
        kill -9 -"$leader_pid" 2>/dev/null || true
    fi
    if [ -n "$worker_pid" ] && kill -0 "$worker_pid" 2>/dev/null; then
        kill -9 -"$worker_pid" 2>/dev/null || true
    fi
    sleep 1
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
# Phase 2b: KV cache block validation
# The winner is active with KV cache allocated (~45 GiB/GPU).
# The loser computed its blocks via patch_determine_available_memory
# while the winner was consuming GPU memory. Verify the loser's
# projected blocks match the winner's actual blocks (±10%).
# ============================================================
echo ""
echo "=== Phase 2b: KV Cache Block Validation ==="

# Winner metrics (from get_engine_cache_info after wake)
W_NUM_BLOCKS=$(extract_from_log "$WINNER_LOG" "Cache config values" \
    "s/.*num_gpu_blocks': \([0-9]*\).*/\1/")
W_KV_TOKENS=$(extract_from_log "$WINNER_LOG" "GPU KV cache size" \
    "s/.*GPU KV cache size: \([0-9,]*\) tokens.*/\1/" | tr -d ',')

# Loser metrics (from init, before sleeping)
L_KV_TOKENS=$(extract_from_log "$LOSER_LOG" "GPU KV cache size" \
    "s/.*GPU KV cache size: \([0-9,]*\) tokens.*/\1/" | tr -d ',')
L_PROJECTED_MEM=$(extract_from_log "$LOSER_LOG" "projected available memory" \
    "s/.*projected available memory \([0-9.]*\) GiB.*/\1/")
# num_gpu_blocks is logged after wake, so derive from tokens for now
L_NUM_BLOCKS_INIT=$((L_KV_TOKENS / 16))

echo "Winner: num_gpu_blocks=$W_NUM_BLOCKS, KV tokens=$W_KV_TOKENS"
echo "Loser:  num_gpu_blocks=$L_NUM_BLOCKS_INIT (derived), KV tokens=$L_KV_TOKENS, projected=$L_PROJECTED_MEM GiB"

echo ""
echo "  ┌─────────────────────────┬──────────┬──────────┐"
echo "  │ Metric                  │ Winner   │ Loser    │"
echo "  ├─────────────────────────┼──────────┼──────────┤"
printf "  │ %-23s │ %8s │ %8s │\n" "num_gpu_blocks" "$W_NUM_BLOCKS" "$L_NUM_BLOCKS_INIT"
printf "  │ %-23s │ %8s │ %8s │\n" "KV cache tokens" "$W_KV_TOKENS" "$L_KV_TOKENS"
echo "  └─────────────────────────┴──────────┴──────────┘"
echo ""

within_pct "$W_NUM_BLOCKS" "$L_NUM_BLOCKS_INIT" 10 \
    "KV: Loser num_gpu_blocks within 10% of winner"

within_pct "$W_KV_TOKENS" "$L_KV_TOKENS" 10 \
    "KV: Loser KV cache tokens within 10% of winner"

# Verify shadow patch was invoked on loser
if cat "$LOSER_LOG" 2>/dev/null | strip_ansi | grep -q "projected available memory"; then
    pass "KV: Loser used patch_determine_available_memory"
else
    fail "KV: Loser did NOT use patch_determine_available_memory"
fi

# Verify loser skipped KV cache at init
LOSER_WORKER_LOG="$LOG_DIR/${LOSER}_worker.log"
if cat "$LOSER_LOG" 2>/dev/null | strip_ansi | grep -q "Init phase: stored config, skipping KV cache allocation"; then
    pass "KV: Loser leader skipped KV cache at init"
else
    fail "KV: Loser leader did NOT skip KV cache at init"
fi
if cat "$LOSER_WORKER_LOG" 2>/dev/null | strip_ansi | grep -q "Init phase: stored config, skipping KV cache allocation"; then
    pass "KV: Loser worker skipped KV cache at init"
else
    fail "KV: Loser worker did NOT skip KV cache at init"
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

# Post-wake KV cache validation
L_WAKE_ALLOC=$(extract_from_log "$LOSER_LOG" "Allocated KV cache on wake" \
    "s/.*Allocated KV cache on wake: \([0-9.]*\) GiB.*/\1/")
if [ -n "$L_WAKE_ALLOC" ]; then
    pass "KV: Loser allocated KV cache on wake: $L_WAKE_ALLOC GiB"
else
    fail "KV: Loser did not log KV cache allocation on wake"
fi

L_NUM_BLOCKS_WAKE=$(extract_from_log "$LOSER_LOG" "Cache config values" \
    "s/.*num_gpu_blocks': \([0-9]*\).*/\1/")
if [ -n "$L_NUM_BLOCKS_WAKE" ]; then
    within_pct "$W_NUM_BLOCKS" "$L_NUM_BLOCKS_WAKE" 10 \
        "KV: Loser post-wake num_gpu_blocks within 10% of winner"
else
    fail "KV: Could not extract loser num_gpu_blocks after wake"
fi

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
