#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# run ingress
python -m dynamo.frontend --http-port=8000 &

# run worker with LMCache enabled
ENABLE_LMCACHE=1 \
LMCACHE_CHUNK_SIZE=256 \
LMCACHE_LOCAL_CPU=True \
LMCACHE_MAX_LOCAL_CPU_SIZE=20 \
  python -m dynamo.vllm --model Qwen/Qwen3-0.6B
