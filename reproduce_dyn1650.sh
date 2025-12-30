#!/bin/bash
# Script to reproduce DYN-1650: thinking_mode cannot be disabled in Dynamo+TRT-LLM

set -e

echo "================================================"
echo "Reproducing DYN-1650: thinking_mode issue"
echo "Using Qwen/Qwen3-0.6B model"
echo "Using file-based discovery for simplicity"
echo "================================================"

# Environment variables
export DYNAMO_HOME="/workspace"
export MODEL_PATH="Qwen/Qwen3-0.6B"
export SERVED_MODEL_NAME="qwen"
export AGG_ENGINE_ARGS="$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3/agg.yaml"

# File-based discovery configuration
export DYN_STORE_KV="file"
export DYN_FILE_KV="/tmp/dynamo_store_kv_test_dyn1650"
export DYN_REQUEST_PLANE="tcp"

# Function to test thinking mode
test_thinking_mode() {
    local enable_thinking=$1
    local test_name=$2

    echo ""
    echo "Test: $test_name"
    echo "----------------------------------------"

    local request_body='{
        "model": "qwen",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful AI assistant."
            },
            {
                "role": "user",
                "content": "Please introduce Hangzhou in 10 words"
            }
        ],'

    # Add chat_template_args if enable_thinking is specified
    if [ "$enable_thinking" != "default" ]; then
        request_body+='
        "chat_template_args": {
            "enable_thinking": '$enable_thinking'
        },'
    fi

    request_body+='
        "max_tokens": 100,
        "temperature": 0
    }'

    echo "Request body:"
    echo "$request_body" | python -m json.tool

    echo ""
    echo "Response:"
    curl -X POST -s http://0.0.0.0:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "$request_body" | python -m json.tool

    echo ""
}

# Start the server in the container
./container/run.sh --framework trtllm --mount-workspace -- bash -c '
cd /workspace

# Environment variables
export DYNAMO_HOME="/workspace"
export MODEL_PATH="Qwen/Qwen3-0.6B"
export SERVED_MODEL_NAME="qwen"
export AGG_ENGINE_ARGS="$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3/agg.yaml"

# File-based discovery configuration
export DYN_STORE_KV="file"
export DYN_FILE_KV="/tmp/dynamo_store_kv_test_dyn1650"
export DYN_REQUEST_PLANE="tcp"

# Clean up any existing file store
rm -rf "$DYN_FILE_KV"
mkdir -p "$DYN_FILE_KV"

echo "Using file-based discovery at: $DYN_FILE_KV"
echo "Request plane: $DYN_REQUEST_PLANE"

# Setup cleanup trap
cleanup() {
    echo "Cleaning up background processes..."
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
        wait $FRONTEND_PID 2>/dev/null || true
    fi
    if [ ! -z "$WORKER_PID" ]; then
        kill $WORKER_PID 2>/dev/null || true
        wait $WORKER_PID 2>/dev/null || true
    fi
    echo "Cleaning up file store..."
    rm -rf "$DYN_FILE_KV"
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

echo "Starting Dynamo frontend with file-based discovery..."
python3 -m dynamo.frontend \
  --store-kv file \
  --request-plane tcp &
FRONTEND_PID=$!

echo "Starting Dynamo TRT-LLM worker with file-based discovery..."
python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$AGG_ENGINE_ARGS" \
  --store-kv file \
  --request-plane tcp &
WORKER_PID=$!

echo "Frontend PID: $FRONTEND_PID"
echo "Worker PID: $WORKER_PID"

# Wait for server to be ready
echo "Waiting for server to initialize..."

# Wait for models to be available
echo "Waiting for frontend to list models..."
max_wait=60
wait_interval=2
elapsed=0

while [ $elapsed -lt $max_wait ]; do
    # Check if models are available
    echo "Checking for available models... ($elapsed/$max_wait seconds)"
    models_response=$(curl -s http://0.0.0.0:8000/v1/models 2>&1)

    # Always show what we got
    echo "Response from /v1/models:"
    echo "$models_response"

    # Check if we got a valid response with data
    if echo "$models_response" | grep -q "\"data\""; then
        # Check if there are any models in the data array
        model_count=$(echo "$models_response" | python3 -c "import sys, json; data = json.load(sys.stdin); print(len(data.get('data', [])))" 2>/dev/null || echo "0")

        if [ "$model_count" -gt "0" ]; then
            echo "Models are now available (count: $model_count):"
            echo "$models_response" | python3 -m json.tool
            break
        else
            echo "Response has 'data' field but no models yet"
        fi
    else
        echo "No valid 'data' field in response yet"
    fi

    sleep $wait_interval
    elapsed=$((elapsed + wait_interval))
done

if [ $elapsed -ge $max_wait ]; then
    echo "WARNING: Timeout waiting for models to be available after $max_wait seconds"
    echo "Continuing anyway..."
fi

# Additional health check
echo "Checking server health..."
HEALTH_RESPONSE=$(curl -v http://0.0.0.0:8000/health 2>&1)
echo "Health check response:"
echo "$HEALTH_RESPONSE"
echo "$HEALTH_RESPONSE" | tail -1 | python -m json.tool 2>/dev/null || echo "$HEALTH_RESPONSE" | tail -1

echo ""
echo "================================================"
echo "Testing thinking_mode behavior"
echo "================================================"

# Test 1: Try to disable thinking_mode
echo ""
echo "Test 1: Disable thinking_mode (enable_thinking=false)"
echo "======================================================"
echo "Sending request with verbose output..."
RESPONSE=$(curl -X POST -v http://0.0.0.0:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '\''{
        "model": "qwen",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful AI assistant."
            },
            {
                "role": "user",
                "content": "Please introduce Hangzhou in 10 words"
            }
        ],
        "chat_template_args": {
            "enable_thinking": false
        },
        "max_tokens": 100,
        "temperature": 0
    }'\'' 2>&1)

echo "Full curl output:"
echo "$RESPONSE"
echo ""
echo "Formatted response body:"
echo "$RESPONSE" | tail -1 | python -m json.tool 2>/dev/null || echo "Could not parse JSON response"

echo ""
echo "Analyzing response for thinking tags..."
echo ""

# Test 2: Enable thinking_mode explicitly
echo ""
echo "Test 2: Enable thinking_mode (enable_thinking=true)"
echo "===================================================="
echo "Sending request with verbose output..."
RESPONSE=$(curl -X POST -v http://0.0.0.0:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '\''{
        "model": "qwen",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful AI assistant."
            },
            {
                "role": "user",
                "content": "Please introduce Hangzhou in 10 words"
            }
        ],
        "chat_template_args": {
            "enable_thinking": true
        },
        "max_tokens": 100,
        "temperature": 0
    }'\'' 2>&1)

echo "Full curl output:"
echo "$RESPONSE"
echo ""
echo "Formatted response body:"
echo "$RESPONSE" | tail -1 | python -m json.tool 2>/dev/null || echo "Could not parse JSON response"

echo ""
echo "Analyzing response for thinking tags..."
echo ""

# Test 3: No thinking_mode parameter (default behavior)
echo ""
echo "Test 3: Default behavior (no enable_thinking specified)"
echo "========================================================"
echo "Sending request with verbose output..."
RESPONSE=$(curl -X POST -v http://0.0.0.0:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '\''{
        "model": "qwen",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful AI assistant."
            },
            {
                "role": "user",
                "content": "Please introduce Hangzhou in 10 words"
            }
        ],
        "max_tokens": 100,
        "temperature": 0
    }'\'' 2>&1)

echo "Full curl output:"
echo "$RESPONSE"
echo ""
echo "Formatted response body:"
echo "$RESPONSE" | tail -1 | python -m json.tool 2>/dev/null || echo "Could not parse JSON response"

echo ""
echo "================================================"
echo "Test completed!"
echo ""
echo "Expected issue per DYN-1650:"
echo "- When enable_thinking=false, response may still contain <thinking> or <reasoning> tags"
echo "- This indicates thinking_mode cannot be properly disabled in Dynamo+TRT-LLM"
echo "- The issue table shows this works correctly in TRT-LLM alone but fails with Dynamo"
echo "================================================"

# Cleanup will be triggered by trap
'