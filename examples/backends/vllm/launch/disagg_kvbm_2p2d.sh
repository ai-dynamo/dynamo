#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# run ingress with KV router
python -m dynamo.frontend --router-mode kv --http-port=8000 &

# run decode workers on GPU 0 and 1, without enabling KVBM
# NOTE: remove --enforce-eager for production use
CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B --connector nixl --enforce-eager &
DYN_VLLM_KV_EVENT_PORT=20081 \
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
CUDA_VISIBLE_DEVICES=1 python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B --connector nixl --enforce-eager &

# run prefill workers on GPU 2 and 3 with KVBM enabled using 20GB of CPU cache
# NOTE: use different barrier id prefixes for each prefill worker to avoid conflicts
# NOTE: remove --enforce-eager for production use
DYN_VLLM_KV_EVENT_PORT=20082 \
VLLM_NIXL_SIDE_CHANNEL_PORT=20098 \
DYN_KVBM_CPU_CACHE_GB=20 \
CUDA_VISIBLE_DEVICES=2 \
  python3 -m dynamo.vllm \
    --model Qwen/Qwen3-0.6B \
    --is-prefill-worker \
    --connector kvbm nixl \
    --enforce-eager &

DYN_VLLM_KV_EVENT_PORT=20083 \
VLLM_NIXL_SIDE_CHANNEL_PORT=20099 \
DYN_KVBM_LEADER_ZMQ_PUB_PORT=56003 \
DYN_KVBM_LEADER_ZMQ_ACK_PORT=56004 \
DYN_KVBM_CPU_CACHE_GB=20 \
CUDA_VISIBLE_DEVICES=3 \
  python3 -m dynamo.vllm \
    --model Qwen/Qwen3-0.6B \
    --is-prefill-worker \
    --connector kvbm nixl \
    --enforce-eager
