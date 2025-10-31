#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# run ingress with KV router
python -m dynamo.frontend --router-mode kv --http-port=8000 &

# run decode worker on GPU 0, without enabling LMCache
CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B &

# wait for decode worker to initialize
sleep 20

# run prefill worker on GPU 1 with LMCache
ENABLE_LMCACHE=1 \
LMCACHE_CHUNK_SIZE=256 \
LMCACHE_LOCAL_CPU=True \
LMCACHE_MAX_LOCAL_CPU_SIZE=20 \
CUDA_VISIBLE_DEVICES=1 \
  python3 -m dynamo.vllm \
    --model Qwen/Qwen3-0.6B \
    --is-prefill-worker \
    --connector lmcache nixl
