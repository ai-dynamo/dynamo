#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# All-in-one test script that:
# 1. Rebuilds Python bindings
# 2. Launches vLLM with connector
# 3. Waits for server to be ready
# 4. Sends test completion request
# 5. Captures all output

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="$SCRIPT_DIR/test_e2e_$(date +%Y%m%d_%H%M%S).log"

echo "===================================================================="
echo "End-to-End KVBM Connector Test"
echo "===================================================================="
echo ""
echo "Log file: $LOG_FILE"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Cleaning up..."
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    pkill -9 -f "EngineCore" 2>/dev/null || true
}

trap cleanup EXIT

# Step 1: Rebuild bindings
echo "Step 1: Rebuilding Python bindings..."
echo "---------------------------------------------------------------------"
if ! "$SCRIPT_DIR/rebuild.sh" 2>&1 | tee -a "$LOG_FILE"; then
    echo "❌ Rebuild failed!"
    exit 1
fi

echo "✅ Rebuild complete"
echo ""

# Step 2: Launch vLLM in background
echo "Step 2: Launching vLLM with DynamoConnector..."
echo "---------------------------------------------------------------------"

# Launch vLLM in background using the existing script
"$SCRIPT_DIR/launch_vllm_with_connector.sh" 2>&1 >> "$LOG_FILE" &
VLLM_PID=$!

echo "vLLM launched with PID: $VLLM_PID"
echo ""

# Step 3: Wait for server to be ready
echo "Step 3: Waiting for vLLM server to be ready on port 8000..."
echo "---------------------------------------------------------------------"

MAX_WAIT=60
WAIT_COUNT=0

while ! curl -s http://127.0.0.1:8000/health &>/dev/null; do
    if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
        echo "❌ Timeout waiting for vLLM server (waited ${MAX_WAIT}s)"
        exit 1
    fi

    # Check if process is still running
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "❌ vLLM process died during startup"
        exit 1
    fi

    echo -n "."
    sleep 1
    WAIT_COUNT=$((WAIT_COUNT + 1))
done

echo ""
echo "✅ vLLM server is ready!"
echo ""

# Step 4: Send test request
echo "Step 4: Sending test completion request..."
echo "---------------------------------------------------------------------"

RESPONSE=$("$SCRIPT_DIR/test_cmpl_1.sh" 2>&1 | tee -a "$LOG_FILE")

echo "$RESPONSE"
echo ""

# Check if response contains error
if echo "$RESPONSE" | grep -q '"error"'; then
    echo "❌ Request failed with error"

    echo ""
    echo "Recent logs from vLLM:"
    echo "---------------------------------------------------------------------"
    tail -50 "$LOG_FILE" | grep -A 10 -i "error\|exception\|traceback" || tail -50 "$LOG_FILE"

    exit 1
else
    echo "✅ Request completed successfully"
fi

echo ""
echo "===================================================================="
echo "Test Summary"
echo "===================================================================="
echo "✅ Bindings rebuilt"
echo "✅ vLLM launched and initialized"
echo "✅ Completion request processed"
echo ""
echo "Full logs saved to: $LOG_FILE"
echo ""
