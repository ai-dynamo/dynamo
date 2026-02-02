#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# E2E Test Script for TensorRT-LLM Video Diffusion Worker
#
# Prerequisites:
#   - GPU with at least 8GB VRAM
#   - etcd running on localhost:2379
#   - NATS running on localhost:4222
#   - Python venv with dependencies installed
#
# Usage:
#   ./run_e2e_test.sh [--model MODEL_PATH]
#
# Example:
#   ./run_e2e_test.sh
#   ./run_e2e_test.sh --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "TRT-LLM Video Diffusion E2E Test"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Repo root: $REPO_ROOT"
echo ""

# Check GPU
echo "Checking GPU..."
if ! nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null; then
    echo "ERROR: GPU not available or nvidia-smi failed"
    exit 1
fi
echo ""

# Check services
echo "Checking etcd..."
if ! nc -z localhost 2379 2>/dev/null; then
    echo "ERROR: etcd not running on localhost:2379"
    echo "Start with: docker run -d -p 2379:2379 quay.io/coreos/etcd:v3.5.0 etcd --advertise-client-urls=http://0.0.0.0:2379 --listen-client-urls=http://0.0.0.0:2379"
    exit 1
fi
echo "  OK"

echo "Checking NATS..."
if ! nc -z localhost 4222 2>/dev/null; then
    echo "ERROR: NATS not running on localhost:4222"
    echo "Start with: docker run -d -p 4222:4222 nats:latest"
    exit 1
fi
echo "  OK"
echo ""

# Set environment
export ETCD_ENDPOINTS=localhost:2379
export NATS_URL=nats://localhost:4222
export PYTHONPATH="$REPO_ROOT/components/src:${PYTHONPATH:-}"

# Start worker in background
echo "Starting video worker..."
python -m dynamo.trtllm_diffusion \
    --model-path "$MODEL_PATH" \
    --disable-torch-compile \
    2>&1 | tee /tmp/trtllm_diffusion_worker.log &
WORKER_PID=$!

# Wait for worker to be ready
echo "Waiting for worker to be ready..."
MAX_WAIT=180
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if grep -q "serving endpoint" /tmp/trtllm_diffusion_worker.log 2>/dev/null; then
        echo "Worker is ready!"
        break
    fi
    if ! kill -0 $WORKER_PID 2>/dev/null; then
        echo "ERROR: Worker process died"
        cat /tmp/trtllm_diffusion_worker.log
        exit 1
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    echo "  Waiting... ($WAITED/$MAX_WAIT s)"
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "ERROR: Worker did not become ready within $MAX_WAIT seconds"
    kill $WORKER_PID 2>/dev/null || true
    cat /tmp/trtllm_diffusion_worker.log
    exit 1
fi
echo ""

# Send test request
echo "Sending test request..."
python -c "
import asyncio
import json
from dynamo.runtime import DistributedRuntime

async def test():
    loop = asyncio.get_running_loop()
    runtime = DistributedRuntime(loop, 'etcd', 'nats', True)

    try:
        client = await (
            runtime.namespace('dynamo')
            .component('trtllm_diffusion')
            .endpoint('generate')
            .client()
        )
        await asyncio.sleep(0.5)

        instances = client.instance_ids()
        print(f'Found {len(instances)} worker instance(s)')

        request = {
            'prompt': 'A cat playing piano',
            'model': '$MODEL_PATH',
            'num_frames': 9,  # Minimal for testing
            'num_inference_steps': 2,
        }

        print('Sending request...')

        async def get_response():
            iterator = await client.random(request)
            async for resp in iterator:
                return resp.data()
            return None

        response = await asyncio.wait_for(get_response(), timeout=300)

        # Response is already a dict or JSON string from .data()
        if isinstance(response, str):
            response = json.loads(response)
        print(f'Response: {json.dumps(response, indent=2)}')

        if response is None:
            print('ERROR: No response received')
            return 1
        elif response.get('status') == 'completed':
            print('')
            print('SUCCESS: Video generation completed!')
            return 0
        elif 'error' in response:
            print(f'ERROR: {response[\"error\"]}')
            return 1
        else:
            print('UNKNOWN: Unexpected response')
            return 1

    finally:
        runtime.shutdown()

exit(asyncio.run(test()))
"
TEST_RESULT=$?

# Cleanup
echo ""
echo "Stopping worker..."
kill $WORKER_PID 2>/dev/null || true
wait $WORKER_PID 2>/dev/null || true

if [ $TEST_RESULT -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "E2E TEST PASSED"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "E2E TEST FAILED"
    echo "========================================"
fi

exit $TEST_RESULT
