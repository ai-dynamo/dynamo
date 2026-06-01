#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"   # build_trtllm_override_args_with_mem
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

# Environment variables with defaults
export DYNAMO_HOME=${DYNAMO_HOME:-"/workspace"}
export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2-VL-7B-Instruct"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"Qwen/Qwen2-VL-7B-Instruct"}
export AGG_ENGINE_ARGS=${AGG_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen2-vl-7b-instruct/agg.yaml"}
export MODALITY=${MODALITY:-"multimodal"}

# Profiler/test-harness override: when _PROFILE_OVERRIDE_TRTLLM_MAX_TOTAL_TOKENS or
# _PROFILE_OVERRIDE_TRTLLM_MAX_GPU_TOTAL_BYTES is set, build_trtllm_override_args_with_mem
# emits a KvCacheConfig JSON; otherwise empty. Empty when unset, so direct invocations
# are unchanged.
TRTLLM_OVERRIDE_ARGS=()
OVERRIDE_JSON=$(build_trtllm_override_args_with_mem)
if [[ -n "$OVERRIDE_JSON" ]]; then
    TRTLLM_OVERRIDE_ARGS=(--override-engine-args "$OVERRIDE_JSON")
fi

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner --multimodal "Launching Aggregated Multimodal Serving" "$MODEL_PATH" "$HTTP_PORT"

# run frontend
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python3 -m dynamo.frontend --router-mode kv &

# run worker
python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$AGG_ENGINE_ARGS" \
  --modality "$MODALITY" \
  "${TRTLLM_OVERRIDE_ARGS[@]}" \
  --publish-events-and-metrics &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
