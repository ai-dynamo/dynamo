#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"   # build_trtllm_override_args_with_mem
source "$SCRIPT_DIR/../../../common/launch_utils.sh" # print_launch_banner, wait_any_exit

# Environment variables with defaults
export DYNAMO_HOME=${DYNAMO_HOME:-"/workspace"}
export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-0.6B"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"Qwen/Qwen3-0.6B"}
export PREFILL_ENGINE_ARGS=${PREFILL_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3/prefill.yaml"}
export DECODE_ENGINE_ARGS=${DECODE_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3/decode.yaml"}
export PREFILL_CUDA_VISIBLE_DEVICES=${PREFILL_CUDA_VISIBLE_DEVICES:-"0"}
export DECODE_CUDA_VISIBLE_DEVICES=${DECODE_CUDA_VISIBLE_DEVICES:-"1"}

# Optional KV cap from _PROFILE_OVERRIDE_TRTLLM_* (empty when unset). It applies
# to each worker and bounds the KV region that NIXL registers on each GPU.
TRTLLM_OVERRIDE_ARGS=()
OVERRIDE_JSON=$(build_trtllm_override_args_with_mem)
if [[ -n "$OVERRIDE_JSON" ]]; then
    TRTLLM_OVERRIDE_ARGS=(--override-engine-args "$OVERRIDE_JSON")
fi

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Disaggregated + KV Routing (2 GPUs)" "$MODEL_PATH" "$HTTP_PORT"

# run frontend with KV routing for cache-aware optimization
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python3 -m dynamo.frontend --router-mode kv &

# run prefill worker
# Publishes KV events for router's cache-aware routing
# No next_endpoint needed - unified frontend handles routing
CUDA_VISIBLE_DEVICES=$PREFILL_CUDA_VISIBLE_DEVICES python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$PREFILL_ENGINE_ARGS" \
  --disaggregation-mode prefill \
  --publish-kv-events \
  "${TRTLLM_OVERRIDE_ARGS[@]}" &

# run decode worker
# No event publishing needed - prefill handles it
CUDA_VISIBLE_DEVICES=$DECODE_CUDA_VISIBLE_DEVICES python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$DECODE_ENGINE_ARGS" \
  --disaggregation-mode decode \
  "${TRTLLM_OVERRIDE_ARGS[@]}" &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
