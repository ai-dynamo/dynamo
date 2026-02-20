#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Set deterministic hash for KV event IDs
export PYTHONHASHSEED=0

# ============================================================
# Configuration — change these for your setup
# ============================================================
STRONG_MODEL="${STRONG_MODEL:-/data/models/gpt-oss-120b}"
WEAK_MODEL="${WEAK_MODEL:-/data/models/gpt-oss-20b}"
STRONG_GPUS="${STRONG_GPUS:-1,2,3,4}"
STRONG_TP="${STRONG_TP:-4}"
WEAK_GPUS="${WEAK_GPUS:-5}"
ROUTER_TYPE="${ROUTER_TYPE:-mf}"
THRESHOLD="${THRESHOLD:-0.5}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-routellm/mf_gpt4_augmented}"
FRONTEND_PORT=8000
PROXY_PORT=8080

# ============================================================
# 1. Start weak model worker (single GPU)
# ============================================================
echo "Starting weak model worker ($WEAK_MODEL) on GPU(s) $WEAK_GPUS..."
CUDA_VISIBLE_DEVICES=$WEAK_GPUS python3 -m dynamo.vllm \
    --model $WEAK_MODEL \
    --enforce-eager \
    --connector none &

# ============================================================
# 2. Start strong model worker (multi-GPU with tensor parallelism)
# ============================================================
echo "Starting strong model worker ($STRONG_MODEL) on GPU(s) $STRONG_GPUS (TP=$STRONG_TP)..."
CUDA_VISIBLE_DEVICES=$STRONG_GPUS python3 -m dynamo.vllm \
    --model $STRONG_MODEL \
    --tensor-parallel-size $STRONG_TP \
    --enforce-eager \
    --connector none &

# ============================================================
# 3. Start Dynamo Frontend (discovers both models via etcd)
# ============================================================
echo "Starting Dynamo Frontend on port $FRONTEND_PORT..."
python -m dynamo.frontend \
    --http-port $FRONTEND_PORT &

# ============================================================
# 4. Start RouteLLM Proxy (client-facing entrypoint)
# ============================================================
echo "Starting RouteLLM Proxy on port $PROXY_PORT..."
echo "  Strong model: $STRONG_MODEL"
echo "  Weak model:   $WEAK_MODEL"
echo "  Router:       $ROUTER_TYPE (threshold=$THRESHOLD)"
python -m dynamo.nemo_switchyard \
    --http-port $PROXY_PORT \
    --backend-url "http://localhost:$FRONTEND_PORT" \
    --strong-model "$STRONG_MODEL" \
    --weak-model "$WEAK_MODEL" \
    --router-type "$ROUTER_TYPE" \
    --threshold "$THRESHOLD" \
    --checkpoint-path "$CHECKPOINT_PATH"
