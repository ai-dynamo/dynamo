#!/bin/bash
# Shadow Engine Failover Test Script
# Tests full failover scenario: primary engine dies, shadow wakes up and takes over
#
# Usage: ./test_shadow_failover.sh [MODEL_NAME]
# Default model: Qwen/Qwen3-0.6B

set -e

# Configuration
MODEL_NAME="${1:-Qwen/Qwen3-0.6B}"

cd /home/mabdulwahhab/repos/dynamo-7

# Two venvs available:
# 1. .venv - has editable vLLM install (with direct modifications)
# 2. dynamo-standalone - has PyPI vLLM (with monkey patches only)
#
# Use dynamo-standalone by default for testing patch-based approach
VENV_NAME="${VENV_NAME:-dynamo-standalone}"
source "${VENV_NAME}/bin/activate"
source .env

LOG_DIR="/tmp/failover_test_$$"
mkdir -p "$LOG_DIR"

GMS_LOG="$LOG_DIR/gms.log"
PRIMARY_LOG="$LOG_DIR/primary.log"
SHADOW_LOG="$LOG_DIR/shadow.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"
GMS_PID=""
PRIMARY_PID=""
SHADOW_PID=""
FRONTEND_PID=""

cleanup() {
    echo ""
    echo "=== Cleaning up ==="
    if [ -n "$FRONTEND_PID" ]; then
        echo "Killing frontend (PID: $FRONTEND_PID)"
        kill $FRONTEND_PID 2>/dev/null || true
        wait $FRONTEND_PID 2>/dev/null || true
    fi
    if [ -n "$SHADOW_PID" ]; then
        echo "Killing shadow engine (PID: $SHADOW_PID)"
        kill $SHADOW_PID 2>/dev/null || true
        wait $SHADOW_PID 2>/dev/null || true
    fi
    if [ -n "$PRIMARY_PID" ]; then
        echo "Killing primary engine (PID: $PRIMARY_PID)"
        kill $PRIMARY_PID 2>/dev/null || true
        wait $PRIMARY_PID 2>/dev/null || true
    fi
    if [ -n "$GMS_PID" ]; then
        echo "Killing GMS (PID: $GMS_PID)"
        kill $GMS_PID 2>/dev/null || true
        wait $GMS_PID 2>/dev/null || true
    fi
    echo "Logs saved in: $LOG_DIR"
}

trap cleanup EXIT

echo "=============================================="
echo "=== Shadow Engine Failover Test ==="
echo "=============================================="
echo "Model: $MODEL_NAME"
echo "Log directory: $LOG_DIR"
echo ""

# Step 1: Start GMS
echo "=== Step 1: Starting GPU Memory Service ==="
python3 -m gpu_memory_service --device 0 > "$GMS_LOG" 2>&1 &
GMS_PID=$!
echo "GMS PID: $GMS_PID"

for i in {1..30}; do
    if grep -q "waiting for connections" "$GMS_LOG" 2>/dev/null; then
        echo "✓ GMS is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "ERROR: GMS failed to start"
        cat "$GMS_LOG"
        exit 1
    fi
    sleep 1
done

# Step 2: Start Primary Engine (normal mode with full KV cache)
echo ""
echo "=== Step 2: Starting Primary Engine (normal mode) ==="
DYN_SYSTEM_PORT=8100 \
python3 -m dynamo.vllm \
    --model "$MODEL_NAME" \
    -tp 1 \
    --load-format gms \
    > "$PRIMARY_LOG" 2>&1 &
PRIMARY_PID=$!
echo "Primary Engine PID: $PRIMARY_PID"

echo "Waiting for primary engine to be ready..."
for i in {1..180}; do
    if grep -q "Registered endpoint 'generate'" "$PRIMARY_LOG" 2>/dev/null; then
        echo "✓ Primary engine is ready!"
        break
    fi
    if ! kill -0 $PRIMARY_PID 2>/dev/null; then
        echo "ERROR: Primary engine process died"
        cat "$PRIMARY_LOG"
        exit 1
    fi
    if [ $i -eq 180 ]; then
        echo "ERROR: Primary engine failed to start within 180 seconds"
        tail -50 "$PRIMARY_LOG"
        exit 1
    fi
    sleep 1
done

# Step 3: Start Frontend
echo ""
echo "=== Step 3: Starting Frontend ==="
python3 -m dynamo.frontend > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

for i in {1..30}; do
    if grep -q "Completions is ready" "$FRONTEND_LOG" 2>/dev/null; then
        echo "✓ Frontend is ready!"
        break
    fi
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        echo "ERROR: Frontend process died"
        cat "$FRONTEND_LOG"
        exit 1
    fi
    if [ $i -eq 30 ]; then
        echo "ERROR: Frontend failed to start within 30 seconds"
        tail -30 "$FRONTEND_LOG"
        exit 1
    fi
    sleep 1
done

# Step 4: Test inference on primary
echo ""
echo "=== Step 4: Testing Inference on Primary Engine ==="
INFERENCE_RESPONSE=$(curl -s -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL_NAME\",
        \"prompt\": \"The capital of France is\",
        \"max_tokens\": 20,
        \"temperature\": 0
    }")

if echo "$INFERENCE_RESPONSE" | grep -q '"choices"'; then
    echo "✓ Inference on primary succeeded!"
    GENERATED_TEXT=$(echo "$INFERENCE_RESPONSE" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['choices'][0]['text'])" 2>/dev/null)
    echo "Generated text: $GENERATED_TEXT"
else
    echo "✗ Inference on primary failed!"
    echo "$INFERENCE_RESPONSE"
    exit 1
fi

# Step 5: Start Shadow Engine (--gms-mode shadow)
echo ""
echo "=== Step 5: Starting Shadow Engine (--gms-mode shadow) ==="
DYN_SYSTEM_PORT=8101 \
VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
DYN_VLLM_KV_EVENT_PORT=20081 \
python3 -m dynamo.vllm \
    --model "$MODEL_NAME" \
    -tp 1 \
    --load-format gms \
    --gms-mode shadow \
    > "$SHADOW_LOG" 2>&1 &
SHADOW_PID=$!
echo "Shadow Engine PID: $SHADOW_PID"

echo "Waiting for shadow engine to initialize and auto-sleep..."
for i in {1..180}; do
    if grep -q "\[Shadow\] Engine is now sleeping" "$SHADOW_LOG" 2>/dev/null; then
        echo "✓ Shadow engine is sleeping!"
        break
    fi
    if ! kill -0 $SHADOW_PID 2>/dev/null; then
        echo "ERROR: Shadow engine process died"
        cat "$SHADOW_LOG"
        exit 1
    fi
    if [ $i -eq 180 ]; then
        echo "ERROR: Shadow engine failed to start within 180 seconds"
        tail -50 "$SHADOW_LOG"
        exit 1
    fi
    sleep 1
done

# Verify KV cache was skipped
if grep -q "\[Shadow\] Init phase: stored config, skipping KV cache allocation" "$SHADOW_LOG"; then
    echo "✓ Shadow engine skipped KV cache allocation"
else
    echo "✗ ERROR: Shadow engine did not skip KV cache!"
    echo "Looking for: [Shadow] Init phase: stored config, skipping KV cache allocation"
    tail -30 "$SHADOW_LOG"
    exit 1
fi

# Check GPU memory with both engines
echo ""
echo "=== GPU Memory (primary active, shadow sleeping) ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# Step 6: Kill the primary engine (simulate failure)
echo ""
echo "=== Step 6: Simulating Primary Engine Failure ==="
echo "Killing primary engine (PID: $PRIMARY_PID)..."
kill $PRIMARY_PID 2>/dev/null || true
wait $PRIMARY_PID 2>/dev/null || true
PRIMARY_PID=""
echo "✓ Primary engine killed"

# Give discovery time to notice
sleep 3

# Step 7: Wake the shadow engine (with timing)
echo ""
echo "=== Step 7: Waking Shadow Engine (Failover) ==="

# Count current re-registration lines before wake
REREGISTER_COUNT_BEFORE=$(cat "$SHADOW_LOG" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | grep -c "Re-registered endpoint to discovery" || echo 0)
REREGISTER_COUNT_BEFORE=$(echo "$REREGISTER_COUNT_BEFORE" | tr -d '[:space:]')

# Record start time
FAILOVER_START_NS=$(date +%s%N)

WAKE_RESPONSE=$(curl -s -X POST http://localhost:8101/engine/wake_up \
    -H "Content-Type: application/json" \
    -d '{}')
echo "Wake response: $WAKE_RESPONSE"

if echo "$WAKE_RESPONSE" | grep -q '"status":"ok"'; then
    echo "✓ Wake API call succeeded"
else
    echo "✗ Wake failed!"
    echo "$WAKE_RESPONSE"
    tail -30 "$SHADOW_LOG"
    exit 1
fi

# Wait for shadow engine to be ready
echo "Waiting for shadow engine to be ready after wake..."
for i in {1..60}; do
    REREGISTER_COUNT_NOW=$(cat "$SHADOW_LOG" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | grep -c "Re-registered endpoint to discovery" || echo 0)
    REREGISTER_COUNT_NOW=$(echo "$REREGISTER_COUNT_NOW" | tr -d '[:space:]')
    if [ "$REREGISTER_COUNT_NOW" -gt "$REREGISTER_COUNT_BEFORE" ]; then
        FAILOVER_END_NS=$(date +%s%N)
        FAILOVER_DURATION_MS=$(( (FAILOVER_END_NS - FAILOVER_START_NS) / 1000000 ))
        echo "✓ Shadow engine is ready!"
        echo ""
        echo "=========================================="
        echo "  FAILOVER TIME: ${FAILOVER_DURATION_MS} ms"
        echo "=========================================="
        break
    fi
    if ! kill -0 $SHADOW_PID 2>/dev/null; then
        echo "ERROR: Shadow engine process died during wake"
        tail -50 "$SHADOW_LOG"
        exit 1
    fi
    if [ $i -eq 60 ]; then
        echo "ERROR: Shadow engine failed to become ready after wake"
        tail -50 "$SHADOW_LOG"
        exit 1
    fi
    sleep 0.1
done

# Verify KV cache was allocated on wake
if grep -q "Allocated KV cache on wake" "$SHADOW_LOG"; then
    echo "✓ KV cache allocated on wake"
    grep "Allocated KV cache on wake" "$SHADOW_LOG" | tail -1
fi

# Check GPU memory after failover
echo ""
echo "=== GPU Memory (after failover) ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# Step 8: Test inference on shadow (now active)
echo ""
echo "=== Step 8: Testing Inference on Shadow Engine (Post-Failover) ==="

# Wait a bit for discovery to propagate
sleep 2

INFERENCE_RESPONSE=$(curl -s -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL_NAME\",
        \"prompt\": \"The capital of France is\",
        \"max_tokens\": 20,
        \"temperature\": 0
    }")

if echo "$INFERENCE_RESPONSE" | grep -q '"choices"'; then
    echo "✓ Inference on shadow engine succeeded!"
    GENERATED_TEXT=$(echo "$INFERENCE_RESPONSE" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['choices'][0]['text'])" 2>/dev/null)
    echo "Generated text: $GENERATED_TEXT"
else
    echo "✗ Inference on shadow engine failed!"
    echo "$INFERENCE_RESPONSE"
    echo ""
    echo "=== Shadow engine logs (last 30 lines) ==="
    tail -30 "$SHADOW_LOG"
    exit 1
fi

echo ""
echo "=================================================="
echo "=== TEST PASSED: Failover completed successfully ==="
echo "=================================================="
echo ""
echo "Summary:"
echo "  - Primary engine started and served inference"
echo "  - Shadow engine started with --gms-mode shadow (no KV cache, auto-sleep)"
echo "  - Primary engine killed (simulated failure)"
echo "  - Shadow engine woke up in ${FAILOVER_DURATION_MS} ms"
echo "  - Shadow engine successfully serves inference"
echo ""
