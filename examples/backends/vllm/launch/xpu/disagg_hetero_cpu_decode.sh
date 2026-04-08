
#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Set deterministic hash for KV event IDs
export PYTHONHASHSEED=0
# Common configuration
MODEL="Qwen/Qwen3-0.6B"
BLOCK_SIZE=64
VLLM_NIXL_DEVICE_TO_DEVICE=false
export UCX_TLS=tcp

# one decode worker on xeon cpu
VLLM_NIXL_SIDE_CHANNEL_PORT=20096 \
python3 -m dynamo.vllm \
    --model $MODEL \
    --block-size $BLOCK_SIZE \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device":"cpu","kv_connector_extra_config":{"enforce_handshake_compat": false}}' \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5556", "enable_kv_cache_events":true}'
