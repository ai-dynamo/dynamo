
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

# Start frontend with KV routing
# The frontend will automatically detect prefill workers and activate an internal prefill router
# edit --router-mode to random / round-robin / kv
python -m dynamo.frontend \
    --router-mode kv \
    --http-port 8000 \
    --router-reset-states &

# one prefill worker on xpu
# When registered with --is-prefill-worker, these workers are automatically detected
# by the frontend, which activates an internal prefill router for KV-aware prefill routing
VLLM_NIXL_SIDE_CHANNEL_PORT=20098 \
ZE_AFFINITY_MASK=0 python3 -m dynamo.vllm \
    --model $MODEL \
    --block-size $BLOCK_SIZE \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device":"cpu","kv_connector_extra_config":{"enforce_handshake_compat": false}}' \
    --is-prefill-worker \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5558", "enable_kv_cache_events":true}'
