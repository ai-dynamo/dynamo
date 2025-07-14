#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# run ingress
dynamo run in=http out=dyn &

CUDA_VISIBLE_DEVICES=0 python3 components/main.py --model Qwen/Qwen3-0.6B --enforce-eager &

CUDA_VISIBLE_DEVICES=1 python3 components/main.py \
    --model Qwen/Qwen3-0.6B \
    --enforce-eager \
    --is-prefill-worker
