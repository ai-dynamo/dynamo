#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Single-machine 2-GPU test for multi-node TP with --headless flag.
#
# Launches frontend + head (node-rank=0, GPU 0) + headless worker (node-rank=1, GPU 1)
# on localhost to validate the headless code path without requiring multiple machines.

set -e
trap 'echo "Cleaning up..."; kill 0' EXIT

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"

echo "Starting Dynamo frontend..."
python3 -m dynamo.frontend &

echo "Starting dynamo.vllm head node (TP=2, nnodes=2, node-rank=0, GPU 0)..."
CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.vllm \
  --model "${MODEL}" \
  --tensor-parallel-size 2 \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr 127.0.0.1 \
  --enforce-eager &

echo "Starting dynamo.vllm headless worker (TP=2, nnodes=2, node-rank=1, GPU 1)..."
CUDA_VISIBLE_DEVICES=1 python3 -m dynamo.vllm \
  --model "${MODEL}" \
  --tensor-parallel-size 2 \
  --nnodes 2 \
  --node-rank 1 \
  --master-addr 127.0.0.1 \
  --enforce-eager \
  --headless &

wait
