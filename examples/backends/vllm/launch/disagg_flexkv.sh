#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

MODEL="Qwen/Qwen3-0.6B"

# Run frontend
python -m dynamo.frontend &

# Run decode worker without FlexKV
CUDA_VISIBLE_DEVICES=0 python -m dynamo.vllm --model $MODEL --connector nixl &

# Run prefill worker with FlexKV
DYN_VLLM_KV_EVENT_PORT=20081 \
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
DYNAMO_USE_FLEXKV=1 \
FLEXKV_CPU_CACHE_GB=32 \
CUDA_VISIBLE_DEVICES=1 \
  python -m dynamo.vllm \
  --model $MODEL \
  --is-prefill-worker \
  --connector nixl flexkv