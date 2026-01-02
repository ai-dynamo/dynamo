#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT


# ---------------------------
# 1. Frontend (Ingress)
# ---------------------------
python -m dynamo.frontend --http-port=8000 &


# ---------------------------
# 2. Speculative Main Worker
# ---------------------------
# This runs the main model with EAGLE as the draft model for speculative decoding
DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=8081 \
CUDA_VISIBLE_DEVICES=0 python -m dynamo.vllm \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --enforce-eager \
    --speculative_config '{
        "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
        "draft_tensor_parallel_size": 1,
        "num_speculative_tokens": 2,
        "method": "eagle"
    }' \
    --connector none \
    --gpu-memory-utilization 0.8