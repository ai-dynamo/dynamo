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
WORKER_MODULE="dynamo.vllm"

# --unified switches both nodes to the unified backend entry point
# (dynamo.vllm.unified_main), exercising the unified headless code path.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --unified) WORKER_MODULE="dynamo.vllm.unified_main"; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

KV_BYTES="${_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES:-}"
GPU_MEM_ARGS=""
if [[ -n "$KV_BYTES" ]]; then
    GPU_MEM_ARGS="--kv-cache-memory-bytes $KV_BYTES --gpu-memory-utilization 0.01"
fi

echo "Starting Dynamo frontend..."
python3 -m dynamo.frontend &

echo "Starting ${WORKER_MODULE} head node (TP=2, nnodes=2, node-rank=0, GPU 0)..."
CUDA_VISIBLE_DEVICES=0 python3 -m "${WORKER_MODULE}" \
  --model "${MODEL}" \
  --tensor-parallel-size 2 \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr 127.0.0.1 \
  --enforce-eager \
  $GPU_MEM_ARGS &

echo "Starting ${WORKER_MODULE} headless worker (TP=2, nnodes=2, node-rank=1, GPU 1)..."
CUDA_VISIBLE_DEVICES=1 python3 -m "${WORKER_MODULE}" \
  --model "${MODEL}" \
  --tensor-parallel-size 2 \
  --nnodes 2 \
  --node-rank 1 \
  --master-addr 127.0.0.1 \
  --enforce-eager \
  $GPU_MEM_ARGS \
  --headless &

wait
