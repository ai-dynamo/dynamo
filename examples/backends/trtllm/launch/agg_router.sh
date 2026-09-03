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
export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-0.6B"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"Qwen/Qwen3-0.6B"}
export AGG_ENGINE_ARGS=${AGG_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3/agg.yaml"}

OVERRIDE_JSON=$(build_trtllm_override_args_with_mem)
TRTLLM_OVERRIDE_ARGS=()
if [[ -n "$OVERRIDE_JSON" ]]; then
    TRTLLM_OVERRIDE_ARGS=(--override-engine-args "$OVERRIDE_JSON")
fi

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Aggregated + KV Routing" "$MODEL_PATH" "$HTTP_PORT"

# run frontend
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python3 -m dynamo.frontend --router-mode kv &

# run worker
python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$AGG_ENGINE_ARGS" \
  --publish-kv-events \
  "${TRTLLM_OVERRIDE_ARGS[@]}" &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
