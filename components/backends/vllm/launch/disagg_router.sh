#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e

trap 'echo Cleaning up...; kill 0' EXIT

# run ingress
python -m dynamo.frontend --router-mode kv --http-port=8000 &

# routing will happen between the two decode workers
# --enforce-eager is added for quick deployment. for production use, need to remove this flag
CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B --enforce-eager &

CUDA_VISIBLE_DEVICES=1 python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B --enforce-eager &

CUDA_VISIBLE_DEVICES=2 python3 -m dynamo.vllm \
    --model Qwen/Qwen3-0.6B \
    --enforce-eager \
    --is-prefill-worker
