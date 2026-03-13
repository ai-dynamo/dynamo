#!/bin/bash
# Shadow KV Cache Block Validation Test (TP=2)
#
# Validates that patch_determine_available_memory correctly projects available
# memory when a primary (non-shadow) engine is actively consuming GPU memory.
#
# Flow:
#   1. Start GMS on both devices
#   2. Start Engine A as NORMAL (non-shadow) — loads weights, allocates KV cache
#   3. Record Engine A's block count and GPU memory
#   4. Start Engine B as SHADOW while Engine A is running
#   5. Compare Engine B's projected blocks to Engine A's actual blocks (±10%)
#   6. Verify Engine B did NOT allocate KV cache
#   7. Kill Engine A → Engine B wakes, allocates KV cache
#   8. Verify Engine B's post-wake GPU memory ≈ Engine A's
#   9. Verify inference works on the woken Engine B
#
# Usage: ./test_shadow_kv_cache_blocks.sh [MODEL_NAME]
#   env: WORKTREE_ROOT  — path to the dynamo worktree (default: current directory)
#   env: VENV_NAME      — virtualenv directory name (default: dynamo)
# Default model: Qwen/Qwen3-0.6B

set -e

MODEL_NAME="${1:-Qwen/Qwen3-0.6B}"
TP_SIZE=2

WORKTREE_ROOT="${WORKTREE_ROOT:-$(pwd)}"
cd "$WORKTREE_ROOT"

VENV_NAME="${VENV_NAME:-dynamo}"
source "${VENV_NAME}/bin/activate"
source .env

LOG_DIR="/tmp/shadow_kv_test_$$"
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

strip_ansi() {
    sed 's/\x1b\[[0-9;]*m//g'
}

# Extract a numeric value from a log line matching a pattern.
# Usage: extract_number "log_file" "grep_pattern" "sed_extract_pattern"
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
    # Integer percentage: diff * 100 / a
    local pct_diff=$(( diff * 100 / a ))
    if [ "$pct_diff" -le "$pct" ]; then
        pass "$label (primary=$a, shadow=$b, delta=${pct_diff}%)"
    else
        fail "$label (primary=$a, shadow=$b, delta=${pct_diff}% > ${pct}%)"
    fi
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
echo "  Shadow KV Cache Block Validation (TP=$TP_SIZE)"
echo "=============================================="
echo "Model: $MODEL_NAME"
echo "Log directory: $LOG_DIR"
echo ""

# ============================================================
# Phase 0: Start GMS on both devices
# ============================================================
echo "=== Phase 0: Starting GPU Memory Service ==="

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
# Phase 1: Start Engine A as NORMAL (non-shadow)
# This engine loads weights, allocates KV cache, and stays active.
# ============================================================
echo ""
echo "=== Phase 1: Start Primary Engine (non-shadow) ==="

ENGINE_ID=0 \
DYN_SYSTEM_PORT=$ENGINE_A_SYSTEM_PORT \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
DYN_VLLM_KV_EVENT_PORT=20080 \
python3 -m dynamo.vllm \
    --model "$MODEL_NAME" \
    -tp "$TP_SIZE" \
    --load-format gms \
    > "$ENGINE_A_LOG" 2>&1 &
ENGINE_A_PID=$!
echo "Engine A PID: $ENGINE_A_PID (non-shadow, normal mode)"

echo "Waiting for Engine A to fully initialize..."
for i in $(seq 1 300); do
    if cat "$ENGINE_A_LOG" 2>/dev/null | strip_ansi | grep -q "Cache config values"; then
        echo "Engine A initialized"
        break
    fi
    if ! kill -0 "$ENGINE_A_PID" 2>/dev/null; then
        echo "ERROR: Engine A died during initialization"
        tail -50 "$ENGINE_A_LOG"
        exit 1
    fi
    if [ "$i" -eq 300 ]; then
        echo "ERROR: Engine A did not initialize within 300s"
        tail -50 "$ENGINE_A_LOG"
        exit 1
    fi
    sleep 1
done

# Wait a few more seconds for KV cache allocation to complete
sleep 5

# ============================================================
# Phase 2: Record Engine A metrics and GPU memory
# ============================================================
echo ""
echo "=== Phase 2: Record Primary Engine Metrics ==="

# Extract Engine A's block count from "Cache config values: {'num_gpu_blocks': N, ...}"
A_NUM_BLOCKS=$(extract_from_log "$ENGINE_A_LOG" "Cache config values" \
    "s/.*num_gpu_blocks': \([0-9]*\).*/\1/")

# Extract Engine A's token count from "GPU KV cache size: N tokens"
A_KV_TOKENS=$(extract_from_log "$ENGINE_A_LOG" "GPU KV cache size" \
    "s/.*GPU KV cache size: \([0-9,]*\) tokens.*/\1/" | tr -d ',')

echo "Engine A (primary):"
echo "  num_gpu_blocks: $A_NUM_BLOCKS"
echo "  GPU KV cache size: $A_KV_TOKENS tokens"

GPU_MEM_AFTER_PRIMARY=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader)
echo "  GPU memory after primary:"
echo "$GPU_MEM_AFTER_PRIMARY" | sed 's/^/    /'

# ============================================================
# Phase 3: Start Engine B as SHADOW while Engine A is running
# Engine A is consuming ~44 GiB per GPU. The shadow patch must
# project available memory from total capacity, not free memory.
# ============================================================
echo ""
echo "=== Phase 3: Start Shadow Engine (concurrent with primary) ==="

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
echo "Engine B PID: $ENGINE_B_PID (shadow mode)"

echo "Waiting for Engine B to compute blocks and enter STANDBY..."
for i in $(seq 1 300); do
    if cat "$ENGINE_B_LOG" 2>/dev/null | strip_ansi | grep -q "waiting for lock"; then
        echo "Engine B reached STANDBY"
        break
    fi
    if ! kill -0 "$ENGINE_B_PID" 2>/dev/null; then
        echo "ERROR: Engine B died during initialization"
        echo "--- Last 50 lines of Engine B log ---"
        tail -50 "$ENGINE_B_LOG"
        exit 1
    fi
    if [ "$i" -eq 300 ]; then
        echo "ERROR: Engine B did not reach STANDBY within 300s"
        tail -50 "$ENGINE_B_LOG"
        exit 1
    fi
    sleep 1
done

# ============================================================
# Phase 4: Compare block counts (the core validation)
# ============================================================
echo ""
echo "=== Phase 4: Compare Block Counts ==="

# Extract Engine B's projected available memory
B_PROJECTED_MEM=$(extract_from_log "$ENGINE_B_LOG" "projected available memory" \
    "s/.*projected available memory \([0-9.]*\) GiB.*/\1/")

# Extract Engine B's token count (available during init, before "waiting for lock")
B_KV_TOKENS=$(extract_from_log "$ENGINE_B_LOG" "GPU KV cache size" \
    "s/.*GPU KV cache size: \([0-9,]*\) tokens.*/\1/" | tr -d ',')

# Note: num_gpu_blocks is logged by get_engine_cache_info which runs after wake,
# so we extract it in Phase 6. For now, derive it from tokens (blocks = tokens / block_size).
# Block size is 16 for this model.
B_NUM_BLOCKS_INIT=$((B_KV_TOKENS / 16))

echo "Engine B (shadow, computed at init while primary holds ~44 GiB/GPU):"
echo "  projected available memory: $B_PROJECTED_MEM GiB"
echo "  num_gpu_blocks (derived): $B_NUM_BLOCKS_INIT"
echo "  GPU KV cache size: $B_KV_TOKENS tokens"

echo ""
echo "  ┌─────────────────────────┬──────────┬──────────┐"
echo "  │ Metric                  │ Primary  │ Shadow   │"
echo "  ├─────────────────────────┼──────────┼──────────┤"
printf "  │ %-23s │ %8s │ %8s │\n" "num_gpu_blocks" "$A_NUM_BLOCKS" "$B_NUM_BLOCKS_INIT"
printf "  │ %-23s │ %8s │ %8s │\n" "KV cache tokens" "$A_KV_TOKENS" "$B_KV_TOKENS"
echo "  └─────────────────────────┴──────────┴──────────┘"
echo ""

within_pct "$A_NUM_BLOCKS" "$B_NUM_BLOCKS_INIT" 10 \
    "Shadow num_gpu_blocks within 10% of primary"

within_pct "$A_KV_TOKENS" "$B_KV_TOKENS" 10 \
    "Shadow KV cache tokens within 10% of primary"

# Verify shadow patch was actually invoked (not the normal path)
if cat "$ENGINE_B_LOG" 2>/dev/null | strip_ansi | grep -q "projected available memory"; then
    pass "Shadow used patch_determine_available_memory (projected, not actual free)"
else
    fail "Shadow did NOT use patch_determine_available_memory"
fi

# ============================================================
# Phase 5: Verify shadow did NOT allocate KV cache
# ============================================================
echo ""
echo "=== Phase 5: Verify Shadow Skipped KV Cache Allocation ==="

if cat "$ENGINE_B_LOG" 2>/dev/null | strip_ansi | grep -q "Init phase: stored config, skipping KV cache allocation"; then
    pass "Shadow skipped KV cache allocation during init"
else
    fail "Shadow did NOT skip KV cache allocation"
fi

GPU_MEM_WITH_SHADOW=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader)
echo "  GPU memory (primary + shadow overhead):"
echo "$GPU_MEM_WITH_SHADOW" | sed 's/^/    /'

# ============================================================
# Phase 6: Kill primary → shadow wakes, allocates KV cache
# ============================================================
echo ""
echo "=== Phase 6: Failover — Kill Primary, Wake Shadow ==="

KILL_EPOCH_MS=$(date +%s%3N)
echo "Killing Engine A (primary, PID: $ENGINE_A_PID)..."
kill "$ENGINE_A_PID" 2>/dev/null || true
wait "$ENGINE_A_PID" 2>/dev/null || true
ENGINE_A_PID=""

echo "Primary killed. Waiting for shadow to acquire lock and wake..."
for i in $(seq 1 120); do
    if cat "$ENGINE_B_LOG" 2>/dev/null | strip_ansi | grep -q "Lock acquired, waking engine"; then
        echo "Shadow acquired lock!"
        break
    fi
    if ! kill -0 "$ENGINE_B_PID" 2>/dev/null; then
        echo "ERROR: Shadow died during failover"
        tail -50 "$ENGINE_B_LOG"
        exit 1
    fi
    if [ "$i" -eq 120 ]; then
        echo "ERROR: Shadow did not acquire lock within 120s"
        tail -50 "$ENGINE_B_LOG"
        exit 1
    fi
    sleep 1
done

echo "Waiting for shadow to register generate endpoint..."
for i in $(seq 1 120); do
    if cat "$ENGINE_B_LOG" 2>/dev/null | strip_ansi | grep -q "Registered endpoint 'generate'"; then
        break
    fi
    if ! kill -0 "$ENGINE_B_PID" 2>/dev/null; then
        echo "ERROR: Shadow died during wake"
        tail -50 "$ENGINE_B_LOG"
        exit 1
    fi
    if [ "$i" -eq 120 ]; then
        echo "ERROR: Shadow did not register within 120s"
        tail -50 "$ENGINE_B_LOG"
        exit 1
    fi
    sleep 1
done
sleep 3

# Extract shadow's allocated KV cache size on wake
B_WAKE_ALLOC=$(extract_from_log "$ENGINE_B_LOG" "Allocated KV cache on wake" \
    "s/.*Allocated KV cache on wake: \([0-9.]*\) GiB.*/\1/")

if [ -n "$B_WAKE_ALLOC" ]; then
    pass "Shadow allocated KV cache on wake: $B_WAKE_ALLOC GiB"
else
    fail "Shadow did not log KV cache allocation on wake"
fi

# Now extract the actual num_gpu_blocks (logged by get_engine_cache_info after wake)
B_NUM_BLOCKS=$(extract_from_log "$ENGINE_B_LOG" "Cache config values" \
    "s/.*num_gpu_blocks': \([0-9]*\).*/\1/")

if [ -n "$B_NUM_BLOCKS" ]; then
    echo "  Shadow num_gpu_blocks (post-wake): $B_NUM_BLOCKS"
    within_pct "$A_NUM_BLOCKS" "$B_NUM_BLOCKS" 10 \
        "Shadow post-wake num_gpu_blocks within 10% of primary"
else
    fail "Could not extract shadow num_gpu_blocks after wake"
fi

# ============================================================
# Phase 7: Verify post-wake GPU memory matches primary's
# ============================================================
echo ""
echo "=== Phase 7: Verify Post-Wake GPU Memory ==="

GPU_MEM_AFTER_WAKE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader)
echo "  GPU memory after shadow wake (should be ~same as primary was):"
echo "$GPU_MEM_AFTER_WAKE" | sed 's/^/    /'
echo "  GPU memory when primary was active:"
echo "$GPU_MEM_AFTER_PRIMARY" | sed 's/^/    /'

# Compare GPU 0 memory: primary vs post-wake shadow
PRIMARY_GPU0_MEM=$(echo "$GPU_MEM_AFTER_PRIMARY" | head -1 | sed 's/.*,\s*\([0-9]*\) MiB/\1/')
WAKE_GPU0_MEM=$(echo "$GPU_MEM_AFTER_WAKE" | head -1 | sed 's/.*,\s*\([0-9]*\) MiB/\1/')

within_pct "$PRIMARY_GPU0_MEM" "$WAKE_GPU0_MEM" 10 \
    "Post-wake GPU 0 memory within 10% of primary"

# ============================================================
# Phase 8: Test inference on the woken shadow
# ============================================================
echo ""
echo "=== Phase 8: Inference on Woken Shadow ==="

echo "Starting Frontend..."
python3 -m dynamo.frontend > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!

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
    pass "Inference on woken shadow succeeded: $GENERATED"
else
    fail "Inference on woken shadow failed: $INFERENCE_RESPONSE"
fi

echo ""
echo "=============================================="
echo "  TEST COMPLETE"
echo "=============================================="
echo "Summary:"
echo "  - Engine A (primary) started in normal mode, allocated full KV cache"
echo "  - Engine A: num_gpu_blocks=$A_NUM_BLOCKS, KV tokens=$A_KV_TOKENS"
echo "  - Engine B (shadow) started while Engine A was consuming ~44 GiB/GPU"
echo "  - Engine B: num_gpu_blocks=$B_NUM_BLOCKS (init=$B_NUM_BLOCKS_INIT), KV tokens=$B_KV_TOKENS (projected=$B_PROJECTED_MEM GiB)"
echo "  - Shadow skipped KV cache allocation at init"
echo "  - Kill primary → shadow woke, allocated $B_WAKE_ALLOC GiB KV cache"
echo "  - Post-wake inference: OK"
echo ""
