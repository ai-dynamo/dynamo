#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# LMCache Dynamo One-Click Test Script

MODEL_URL=${1:-"Qwen/Qwen3-0.6B"}
NUM_SUBJECTS=${2:-15}

echo "🧪 LMCache Dynamo Complete Test"
echo "==============================="
echo "Model: $MODEL_URL"
echo "Number of subjects: $NUM_SUBJECTS"
echo ""

# Function to cleanup processes
cleanup() {
    echo "🧹 Cleaning up running processes..."

    # Kill any remaining dynamo processes
    pkill -f "dynamo-run" || true
    pkill -f "components/main.py" || true

    # Stop docker services
    docker compose -f ../../deploy/metrics/docker-compose.yml down 2>/dev/null || true

    # Wait a moment for cleanup
    sleep 2
}

# Set trap for cleanup on exit
trap cleanup EXIT

# Check if data exists
if [ ! -d "data/test" ] || [ ! -d "data/dev" ]; then
    echo "📚 MMLU dataset not found, starting download..."

    # Check if Python dependencies are installed
    if ! python3 -c "import datasets, pandas" 2>/dev/null; then
        echo "📦 Installing Python dependencies..."
        pip install datasets pandas
    fi

    python3 download_mmlu.py

    if [ $? -ne 0 ]; then
        echo "❌ Data download failed, exiting"
        exit 1
    fi
else
    echo "✅ MMLU dataset already exists"
fi

echo ""
echo "🔬 Step 1: Baseline Test (LMCache disabled)"
echo "==========================================="

# Run baseline test
echo "🚀 Starting baseline dynamo..."
timeout 600 ./deploy-1-dynamo.sh "$MODEL_URL" &
DEPLOY_PID=$!

# Wait for server to be ready
echo "⏳ Waiting for server to be ready..."
sleep 30

# Check if server is responding
max_attempts=30
attempt=0
until curl -s http://localhost:8000/v1/models > /dev/null 2>&1; do
    attempt=$((attempt + 1))
    if [ $attempt -gt $max_attempts ]; then
        echo "❌ Server failed to start within timeout"
        kill $DEPLOY_PID 2>/dev/null || true
        exit 1
    fi
    echo "⏳ Waiting for server... (attempt $attempt/$max_attempts)"
    sleep 10
done

echo "📊 Running baseline MMLU test..."
python3 1-mmlu-dynamo.py --model "$MODEL_URL" --number-of-subjects $NUM_SUBJECTS

if [ $? -ne 0 ]; then
    echo "❌ Baseline test failed"
    kill $DEPLOY_PID 2>/dev/null || true
    exit 1
fi

echo "🛑 Stopping baseline services..."
kill $DEPLOY_PID 2>/dev/null || true
cleanup
sleep 5

echo ""
echo "🔬 Step 2: LMCache Test (LMCache enabled)"
echo "========================================="

# Run LMCache test
echo "🚀 Starting LMCache dynamo..."
timeout 600 ./deploy-2-dynamo.sh "$MODEL_URL" &
DEPLOY_PID=$!

# Wait for server to be ready
echo "⏳ Waiting for server to be ready..."
sleep 30

# Check if server is responding
attempt=0
until curl -s http://localhost:8000/v1/models > /dev/null 2>&1; do
    attempt=$((attempt + 1))
    if [ $attempt -gt $max_attempts ]; then
        echo "❌ Server failed to start within timeout"
        kill $DEPLOY_PID 2>/dev/null || true
        exit 1
    fi
    echo "⏳ Waiting for server... (attempt $attempt/$max_attempts)"
    sleep 10
done

echo "📊 Running LMCache MMLU test..."
python3 2-mmlu-dynamo.py --model "$MODEL_URL" --number-of-subjects $NUM_SUBJECTS

if [ $? -ne 0 ]; then
    echo "❌ LMCache test failed"
    kill $DEPLOY_PID 2>/dev/null || true
    exit 1
fi

echo "🛑 Stopping LMCache services..."
kill $DEPLOY_PID 2>/dev/null || true
cleanup

echo ""
echo "📈 Step 3: Result Analysis"
echo "========================="

# Analyze results
python3 summarize_scores_dynamo.py

echo ""
echo "🎉 Test Complete!"
echo "================"

# Check if result files exist
baseline_file=$(ls dynamo-baseline-*.jsonl 2>/dev/null | head -1)
lmcache_file=$(ls dynamo-lmcache-*.jsonl 2>/dev/null | head -1)

if [ -n "$baseline_file" ] && [ -n "$lmcache_file" ]; then
    echo "✅ Generated result files:"
    echo "   - Baseline test: $baseline_file"
    echo "   - LMCache test: $lmcache_file"
    echo ""
    echo "💡 If accuracy difference < 1%, LMCache functionality is correct"
else
    echo "⚠️ Complete result files not found, please check if there were errors during testing"
fi

echo ""
echo "🔧 To re-run:"
echo "   ./run_test.sh \"$MODEL_URL\" $NUM_SUBJECTS"