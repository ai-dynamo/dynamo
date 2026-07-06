#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
trap 'rm -f "${READY_FILE:-}"; kill 0' EXIT

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 MODEL ENGINE_CONFIG STARTUP_TIMEOUT READY_FILE" >&2
    exit 2
fi

MODEL=$1
ENGINE_CONFIG=$2
STARTUP_TIMEOUT=$3
READY_FILE=$4

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
source "$SCRIPT_DIR/../../../examples/common/launch_utils.sh"

rm -f "$READY_FILE"
export DYN_REQUEST_PLANE=tcp
# Qwen3.5 uses a hybrid Mamba architecture. vLLM's three-read NIXL transfer
# path requires the convolution state dimension to be leading on both workers.
export VLLM_SSM_CONV_STATE_LAYOUT=DS

python -m dynamo.frontend &

# Load decode first. Co-resident large models otherwise contend for GPU memory
# during initialization and can make the second worker fail its free-memory
# check before either endpoint is ready.
DYN_SYSTEM_PORT=8081 \
VLLM_NIXL_SIDE_CHANNEL_PORT=20099 \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python -m dynamo.vllm \
  --disaggregation-mode decode \
  --enable-multimodal \
  --enable-mm-embeds \
  --model "$MODEL" \
  --engine-config-json "$ENGINE_CONFIG" \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20082","enable_kv_cache_events":true}' &

wait_for_ready "http://localhost:8081/health" "$STARTUP_TIMEOUT"

DYN_SYSTEM_PORT=8082 \
VLLM_NIXL_SIDE_CHANNEL_PORT=20098 \
DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT=60 \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python -m dynamo.vllm \
  --disaggregation-mode prefill \
  --enable-multimodal \
  --model "$MODEL" \
  --engine-config-json "$ENGINE_CONFIG" \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}' &

wait_for_ready "http://localhost:8082/health" "$STARTUP_TIMEOUT"

# Give frontend discovery one poll interval to replace the decode-only route
# with the active PrefillRouter before the harness sends its first request.
sleep 5
touch "$READY_FILE"

wait_any_exit
