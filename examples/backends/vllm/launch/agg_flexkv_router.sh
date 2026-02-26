#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

MODEL="Qwen/Qwen3-0.6B"

# Run frontend and KV router
python -m dynamo.frontend \
    --router-mode kv \
    --router-reset-states &

# Run worker with FlexKV
DYNAMO_USE_FLEXKV=1 \
FLEXKV_CPU_CACHE_GB=32 \
FLEXKV_SERVER_RECV_PORT="ipc:///tmp/flexkv_server_0" \
CUDA_VISIBLE_DEVICES=0 \
python -m dynamo.vllm \
    --model $MODEL \
    --kv-transfer-config '{"kv_connector":"FlexKVConnectorV1","kv_role":"kv_both"}' \
    --gpu-memory-utilization 0.2 \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}' &

DYNAMO_USE_FLEXKV=1 \
FLEXKV_CPU_CACHE_GB=32 \
FLEXKV_SERVER_RECV_PORT="ipc:///tmp/flexkv_server_1" \
CUDA_VISIBLE_DEVICES=1 \
python -m dynamo.vllm \
    --model $MODEL \
    --kv-transfer-config '{"kv_connector":"FlexKVConnectorV1","kv_role":"kv_both"}' \
    --gpu-memory-utilization 0.2 \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}'
