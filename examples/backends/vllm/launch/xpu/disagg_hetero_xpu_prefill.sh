#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Launch the XPU side of a heterogeneous disaggregated vLLM setup.
#
# This script starts:
#   - dynamo.frontend on port 8000, acting as the HTTP frontend/proxy with KV routing
#   - one XPU prefill worker registered with --is-prefill-worker
#
# Run this together with disagg_hetero_cpu_decode.sh, which starts the CPU decode
# worker. The two workers exchange KV cache through NixlConnector over TCP, using
# CPU buffers for cross-device compatibility
set -e
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../../common/launch_utils.sh"
trap dynamo_exit_trap EXIT

# Set deterministic hash for KV event IDs
export PYTHONHASHSEED=0

# Common configuration
MODEL="Qwen/Qwen3-0.6B"
BLOCK_SIZE=64
VLLM_NIXL_DEVICE_TO_DEVICE=false
export UCX_TLS=tcp

# Start frontend with KV routing.
# The frontend will automatically detect prefill workers and activate an internal prefill router.
# Edit --router-mode to random / round-robin / kv.
python -m dynamo.frontend \
    --router-mode kv \
    --http-port 8000 \
    --router-reset-states &

# One prefill worker on XPU.
# When registered with --disaggregation-mode prefill, this worker is automatically detected
# by the frontend, which activates an internal prefill router for KV-aware prefill routing.
VLLM_NIXL_SIDE_CHANNEL_PORT=20098 \
ZE_AFFINITY_MASK=0 python3 -m dynamo.vllm \
    --model $MODEL \
    --block-size $BLOCK_SIZE \
    --enforce-eager \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device":"cpu","kv_connector_extra_config":{"enforce_handshake_compat": false}}' \
    --disaggregation-mode prefill \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5558", "enable_kv_cache_events":true}' &

wait_any_exit
