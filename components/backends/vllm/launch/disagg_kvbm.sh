#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# run ingress with KV router
python -m dynamo.frontend --router-mode kv --http-port=8000 &

# run decode worker on GPU 0, without enabling KVBM
# NOTE: remove --enforce-eager for production use
CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.vllm --model qwen/qwen3-0.6b --enforce-eager &

# wait for decode worker to initialize
sleep 20

# run prefill worker on GPU 1 with KVBM
# NOTE: remove --enforce-eager for production use
DYN_KVBM_CPU_CACHE_GB=4 \
CUDA_VISIBLE_DEVICES=1 \
  python3 -m dynamo.vllm \
    --model qwen/qwen3-0.6b \
    --is-prefill-worker \
    --connector kvbm nixl \
    --enforce-eager
