#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Launch script for Aggregated Multimodal with native KV routing.
#
# Architecture:
#   Frontend (Rust + KV router)  -->  TRT-LLM Worker (aggregated multimodal)
#
# The Rust frontend forwards each image's routing hash to the worker as
# multi_modal_uuids; the worker's KV-event publisher normalizes image-token
# runs to the matching pad_value, so KV-aware routing works without a
# separate MM router sidecar.

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"   # build_trtllm_override_args_with_mem
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

export DYNAMO_HOME=${DYNAMO_HOME:-"/workspace"}
export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-VL-2B-Instruct"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"Qwen/Qwen3-VL-2B-Instruct"}
export AGG_ENGINE_ARGS=${AGG_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3-vl-2b-instruct/agg_kv_router.yaml"}
export BLOCK_SIZE=${BLOCK_SIZE:-32}

# Profiler/test-harness override: KvCacheConfig JSON when env var is set, empty otherwise.
TRTLLM_OVERRIDE_ARGS=()
OVERRIDE_JSON=$(build_trtllm_override_args_with_mem)
if [[ -n "$OVERRIDE_JSON" ]]; then
    TRTLLM_OVERRIDE_ARGS=(--override-engine-args "$OVERRIDE_JSON")
fi

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner --multimodal "Launching Aggregated Multimodal + KV Routing" "$MODEL_PATH" "$HTTP_PORT"

# Aggregated multimodal TRT-LLM worker: registers with the real model name.
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$AGG_ENGINE_ARGS" \
  --enable-multimodal \
  "${TRTLLM_OVERRIDE_ARGS[@]}" \
  --publish-events-and-metrics \
  --kv-block-size "$BLOCK_SIZE" &

# Frontend: KV routing over the worker's published KV-cache events.
# dynamo.frontend accepts either --http-port or the DYN_HTTP_PORT env var (default 8000).
python3 -m dynamo.frontend --http-port "$HTTP_PORT" --router-mode kv &

wait_any_exit
