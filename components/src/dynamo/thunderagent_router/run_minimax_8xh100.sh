#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Bring up the thunderagent_router MiniMax-M2 eval stack on a single 8xH100
# node: two TP4 vLLM workers (GPUs 0-3 and 4-7) + the program-aware router +
# the frontend on :8100. Hardcoded for that one config -- nothing else.
#
# Pair it with the pi + Harbor client (see README "Reproducing the MiniMax-M2
# results"). First launch JIT-warms the FP8 kernels and can take several
# minutes before /v1/models responds.
set -euo pipefail
trap 'echo "Cleaning up..."; kill 0' EXIT

MODEL="MiniMaxAI/MiniMax-M2"
BLOCK_SIZE=16
HTTP_PORT=8100

export PYTHONHASHSEED=0  # deterministic KV-event block hashes

# Worker 0: GPUs 0-3.
DYN_SYSTEM_PORT=8081 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m dynamo.vllm \
    --model "$MODEL" --tensor-parallel-size 4 --block-size "$BLOCK_SIZE" \
    --dyn-tool-call-parser minimax_m2 \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}' &

# Worker 1: GPUs 4-7.
DYN_SYSTEM_PORT=8082 CUDA_VISIBLE_DEVICES=4,5,6,7 python -m dynamo.vllm \
    --model "$MODEL" --tensor-parallel-size 4 --block-size "$BLOCK_SIZE" \
    --dyn-tool-call-parser minimax_m2 \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}' &

# Program-aware router: registers the model handler and forwards the parser so
# MiniMax's <minimax:tool_call> XML reaches the agent as OpenAI tool_calls.
python -m dynamo.thunderagent_router \
    --endpoint dynamo.vllm.generate \
    --model-name "$MODEL" \
    --dyn-tool-call-parser minimax_m2 \
    --router-block-size "$BLOCK_SIZE" &

# Frontend (round-robin; the router owns scheduling and registered the model).
python -m dynamo.frontend \
    --http-port "$HTTP_PORT" \
    --router-mode round-robin \
    --router-reset-states &

wait
