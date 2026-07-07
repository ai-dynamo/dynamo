#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Launch the CPU side of a heterogeneous disaggregated vLLM setup.
#
# This script starts one CPU decode worker only. It does not start the frontend.
# Run it together with disagg_hetero_xpu_prefill.sh, which starts the HTTP
# frontend/proxy and the XPU prefill worker.
#
# KV cache is exchanged with the XPU prefill worker through NixlConnector over TCP,
# using CPU buffers for cross-device compatibility.
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

# One decode worker on Xeon CPU.
VLLM_NIXL_SIDE_CHANNEL_PORT=20096 \
python3 -m dynamo.vllm \
    --model $MODEL \
    --block-size $BLOCK_SIZE \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device":"cpu","kv_connector_extra_config":{"enforce_handshake_compat": false}}' \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5556", "enable_kv_cache_events":true}' &

wait_any_exit
