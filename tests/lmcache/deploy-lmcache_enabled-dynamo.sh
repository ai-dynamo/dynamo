#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ASSUMPTION: dynamo and its dependencies are properly installed
# i.e. nats and etcd are running

# Overview:
# This script deploys dynamo with LMCache enabled on port 8000
# Used for LMCache correctness testing
set -e
trap 'echo Cleaning up...; kill 0' EXIT
# Arguments:
MODEL_URL=$1

if [ -z "$MODEL_URL" ]; then
    echo "Usage: $0 <MODEL_URL>"
    echo "Example: $0 Qwen/Qwen3-0.6B"
    exit 1
fi

echo "🚀 Starting dynamo setup with LMCache:"
echo "   Model: $MODEL_URL"
echo "   Port: 8000"
echo "   !! Remmber to kill the old dynamo processes other wise the port will be busy !! "

# Kill any existing dynamo processes
echo "🧹 Cleaning up any existing dynamo processes..."
pkill -f "dynamo-run" || true
sleep 2

echo "🔧 Starting dynamo worker with LMCache enabled..."

python -m dynamo.frontend &
ENABLE_LMCACHE=1 \
  python3 -m dynamo.vllm --model $MODEL_URL