#!/bin/bash
# Shadow Engine Sleep/Wake Test Script
# Tests the shadow KV cache skip and wake allocation functionality
#
# Usage: ./test_shadow_sleep_wake.sh [MODEL_NAME]
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

LOG_DIR="/tmp/shadow_test_$$"
mkdir -p "$LOG_DIR"

GMS_LOG="$LOG_DIR/gms.log"
ENGINE_LOG="$LOG_DIR/engine.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"
GMS_PID=""
ENGINE_PID=""
FRONTEND_PID=""

cleanup() {
    echo ""
    echo "=== Cleaning up ==="
    if [ -n "$FRONTEND_PID" ]; then
        echo "Killing frontend (PID: $FRONTEND_PID)"
        kill $FRONTEND_PID 2>/dev/null || true
        wait $FRONTEND_PID 2>/dev/null || true
    fi
    if [ -n "$ENGINE_PID" ]; then
        echo "Killing engine (PID: $ENGINE_PID)"
        kill $ENGINE_PID 2>/dev/null || true
        wait $ENGINE_PID 2>/dev/null || true
    fi
    if [ -n "$GMS_PID" ]; then
        echo "Killing GMS (PID: $GMS_PID)"
        kill $GMS_PID 2>/dev/null || true
        wait $GMS_PID 2>/dev/null || true
    fi
    echo "Logs saved in: $LOG_DIR"
}

trap cleanup EXIT

echo "=== Shadow Engine Sleep/Wake Test ==="
echo "Model: $MODEL_NAME"
echo "Log directory: $LOG_DIR"
echo ""

# Step 1: Start GMS
echo "=== Step 1: Starting GPU Memory Service ==="
python3 -m gpu_memory_service --device 0 > "$GMS_LOG" 2>&1 &
GMS_PID=$!
echo "GMS PID: $GMS_PID"

# Wait for GMS to be ready
for i in {1..30}; do
    if grep -q "waiting for connections" "$GMS_LOG" 2>/dev/null; then
        echo "GMS is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "ERROR: GMS failed to start"
        cat "$GMS_LOG"
        exit 1
    fi
    sleep 1
done

# Step 2: Start Shadow Engine with --gms-mode shadow
echo ""
echo "=== Step 2: Starting Shadow Engine (--gms-mode shadow) ==="
DYN_SYSTEM_PORT=8100 \
python3 -m dynamo.vllm \
    --model "$MODEL_NAME" \
    -tp 1 \
    --load-format gms \
    --gms-mode shadow \
    > "$ENGINE_LOG" 2>&1 &
ENGINE_PID=$!
echo "Engine PID: $ENGINE_PID"

# Wait for engine to auto-sleep (shadow mode auto-sleeps after init)
echo "Waiting for shadow engine to initialize and auto-sleep..."
for i in {1..180}; do
    if grep -q "\[Shadow\] Engine is now sleeping" "$ENGINE_LOG" 2>/dev/null; then
        echo "Shadow engine is sleeping!"
        break
    fi
    if ! kill -0 $ENGINE_PID 2>/dev/null; then
        echo "ERROR: Engine process died"
        cat "$ENGINE_LOG"
        exit 1
    fi
    if [ $i -eq 180 ]; then
        echo "ERROR: Engine failed to start within 180 seconds"
        tail -50 "$ENGINE_LOG"
        exit 1
    fi
    sleep 1
done

# Verify KV cache was skipped
echo ""
echo "=== Verifying KV cache was skipped ==="
if grep -q "\[Shadow\] Init phase: stored config, skipping KV cache allocation" "$ENGINE_LOG"; then
    echo "✓ KV cache allocation was skipped"
else
    echo "✗ ERROR: KV cache was not skipped!"
    echo "Looking for: [Shadow] Init phase: stored config, skipping KV cache allocation"
    echo "Engine log tail:"
    tail -30 "$ENGINE_LOG"
    exit 1
fi

# Step 2b: Start Frontend
echo ""
echo "=== Step 2b: Starting Frontend ==="
python3 -m dynamo.frontend > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

# Wait for frontend to be ready
echo "Waiting for frontend..."
for i in {1..30}; do
    if grep -q "Completions is ready" "$FRONTEND_LOG" 2>/dev/null; then
        echo "Frontend is ready!"
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

# Check GPU memory (should be low - no KV cache)
echo ""
echo "=== GPU Memory (engine sleeping, no KV cache) ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# Step 3: Wake the shadow engine (with timing)
echo ""
echo "=== Step 3: Testing Wake ==="

# Count current re-registration lines before wake
REREGISTER_COUNT_BEFORE=$(cat "$ENGINE_LOG" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | grep -c "Re-registered endpoint to discovery" || echo 0)
REREGISTER_COUNT_BEFORE=$(echo "$REREGISTER_COUNT_BEFORE" | tr -d '[:space:]')

# Record start time
WAKE_START_NS=$(date +%s%N)

WAKE_RESPONSE=$(curl -s -X POST http://localhost:8100/engine/wake_up \
    -H "Content-Type: application/json" \
    -d '{}')
echo "Wake response: $WAKE_RESPONSE"

if echo "$WAKE_RESPONSE" | grep -q '"status":"ok"'; then
    echo "✓ Wake API call succeeded"
else
    echo "✗ Wake failed!"
    echo "Full response: $WAKE_RESPONSE"
    echo ""
    echo "=== Engine logs (last 30 lines) ==="
    tail -30 "$ENGINE_LOG"
    exit 1
fi

# Wait for engine to be ready (re-registered to discovery)
echo "Waiting for engine to be ready after wake..."
for i in {1..60}; do
    REREGISTER_COUNT_NOW=$(cat "$ENGINE_LOG" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | grep -c "Re-registered endpoint to discovery" || echo 0)
    REREGISTER_COUNT_NOW=$(echo "$REREGISTER_COUNT_NOW" | tr -d '[:space:]')
    if [ "$REREGISTER_COUNT_NOW" -gt "$REREGISTER_COUNT_BEFORE" ]; then
        WAKE_END_NS=$(date +%s%N)
        WAKE_DURATION_MS=$(( (WAKE_END_NS - WAKE_START_NS) / 1000000 ))
        echo "✓ Engine is ready after wake!"
        echo ""
        echo "=========================================="
        echo "  WAKE TIME: ${WAKE_DURATION_MS} ms"
        echo "=========================================="
        break
    fi
    if ! kill -0 $ENGINE_PID 2>/dev/null; then
        echo "ERROR: Engine process died during wake"
        tail -50 "$ENGINE_LOG"
        exit 1
    fi
    if [ $i -eq 60 ]; then
        echo "ERROR: Engine failed to become ready after wake within 60 seconds"
        tail -50 "$ENGINE_LOG"
        exit 1
    fi
    sleep 0.1
done

# Verify wake logs
echo ""
echo "=== Verifying Wake Logs ==="
if grep -q "Allocating KV cache on wake" "$ENGINE_LOG"; then
    echo "✓ KV cache allocation on wake detected"
else
    echo "? KV cache allocation message not found (may have already existed)"
fi

if grep -q "Allocated KV cache on wake" "$ENGINE_LOG"; then
    echo "✓ KV cache successfully allocated on wake"
    grep "Allocated KV cache on wake" "$ENGINE_LOG" | tail -1
fi

# Check GPU memory after wake (should be higher - KV cache allocated)
echo ""
echo "=== GPU Memory (after wake, KV cache allocated) ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# Step 4: Test Inference
echo ""
echo "=== Step 4: Testing Inference ==="
echo "Sending completion request to frontend (port 8000)..."

INFERENCE_RESPONSE=$(curl -s -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL_NAME\",
        \"prompt\": \"The capital of France is\",
        \"max_tokens\": 20,
        \"temperature\": 0
    }")

echo "Inference response:"
echo "$INFERENCE_RESPONSE" | python3 -c "import sys, json; d=json.load(sys.stdin); print(json.dumps(d, indent=2))" 2>/dev/null || echo "$INFERENCE_RESPONSE"

# Check if inference succeeded
if echo "$INFERENCE_RESPONSE" | grep -q '"choices"'; then
    echo ""
    echo "✓ Inference succeeded!"
    # Extract and show the generated text
    GENERATED_TEXT=$(echo "$INFERENCE_RESPONSE" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['choices'][0]['text'])" 2>/dev/null)
    echo "Generated text: $GENERATED_TEXT"
else
    echo ""
    echo "✗ Inference failed!"
    echo "Full response: $INFERENCE_RESPONSE"
    echo ""
    echo "=== Engine logs (last 30 lines) ==="
    tail -30 "$ENGINE_LOG"
    exit 1
fi

echo ""
echo "=============================================="
echo "=== TEST PASSED: Shadow Wake/Inference works ==="
echo "=============================================="
