#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Set deterministic hash for KV event IDs
export PYTHONHASHSEED=0

# Common configuration
MODEL="Qwen/Qwen3-0.6B"
BLOCK_SIZE=64
VLLM_NIXL_DEVICE_TO_DEVICE=false
VLLM_SKIP_WARMUP=true
PT_HPU_LAZY_MODE=0
NIXL_BUFFER_DEVICE=cpu
VLLM_NIXL_BACKEND=UCX


# Start frontend with KV routing
# The frontend will automatically detect prefill workers and activate an internal prefill router
# edit --router-mode to random / round-robin / kv
python -m dynamo.frontend \
    --router-mode kv \
    --http-port 8000 \
    --router-reset-states &

# two decode workers
# --enforce-eager is added for quick deployment. for production use, need to remove this flag
VLLM_NIXL_SIDE_CHANNEL_PORT=20096 \
HABANA_VISIBLE_DEVICES=0 python3 -m dynamo.vllm \
    --model $MODEL \
    --block-size $BLOCK_SIZE \
    --kv-transfer-config "{\"kv_connector\": \"NixlConnector\", \"kv_role\": \"kv_both\", \"kv_buffer_device\": \"${NIXL_BUFFER_DEVICE}\", \"kv_connector_extra_config\": {\"backends\": [\"${VLLM_NIXL_BACKEND}\"]}}" \
    --connector none \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5556", "enable_kv_cache_events":true}' &

VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
HABANA_VISIBLE_DEVICES=1 python3 -m dynamo.vllm \
    --model $MODEL \
    --block-size $BLOCK_SIZE \
    --kv-transfer-config "{\"kv_connector\": \"NixlConnector\", \"kv_role\": \"kv_both\", \"kv_buffer_device\": \"${NIXL_BUFFER_DEVICE}\", \"kv_connector_extra_config\": {\"backends\": [\"${VLLM_NIXL_BACKEND}\"]}}" \
    --connector none \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557", "enable_kv_cache_events":true}' &

# two prefill workers
# When registered with --is-prefill-worker, these workers are automatically detected
# by the frontend, which activates an internal prefill router for KV-aware prefill routing
VLLM_NIXL_SIDE_CHANNEL_PORT=20098 \
HABANA_VISIBLE_DEVICES=2 python3 -m dynamo.vllm \
    --model $MODEL \
    --block-size $BLOCK_SIZE \
    --kv-transfer-config "{\"kv_connector\": \"NixlConnector\", \"kv_role\": \"kv_both\", \"kv_buffer_device\": \"${NIXL_BUFFER_DEVICE}\", \"kv_connector_extra_config\": {\"backends\": [\"${VLLM_NIXL_BACKEND}\"]}}" \
    --connector none \
    --is-prefill-worker \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5558", "enable_kv_cache_events":true}'&

VLLM_NIXL_SIDE_CHANNEL_PORT=20099 \
HABANA_VISIBLE_DEVICES=3 python3 -m dynamo.vllm \
    --model $MODEL \
    --block-size $BLOCK_SIZE \
    --kv-transfer-config "{\"kv_connector\": \"NixlConnector\", \"kv_role\": \"kv_both\", \"kv_buffer_device\": \"${NIXL_BUFFER_DEVICE}\", \"kv_connector_extra_config\": {\"backends\": [\"${VLLM_NIXL_BACKEND}\"]}}" \
    --connector none \
    --is-prefill-worker \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5559", "enable_kv_cache_events":true}'
