#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Launch script for Aggregated Multimodal with MM Router Worker
#
# Architecture:
#   Frontend  -->  MM Router Worker  -->  TRT-LLM Worker
#                  (KV-aware routing)     (aggregated multimodal)
#
# The MM Router Worker sits between frontend and TRT-LLM, computing
# mm_hash for images and routing to the best worker based on KV cache overlap.

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

export DYNAMO_HOME=${DYNAMO_HOME:-"/workspace"}
export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-VL-2B-Instruct"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"Qwen/Qwen3-VL-2B-Instruct"}
export AGG_ENGINE_ARGS=${AGG_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3-vl-2b-instruct/agg.yaml"}
export MODALITY=${MODALITY:-"multimodal"}
export MODEL_TYPE=${MODEL_TYPE:-"qwen3_vl"}
export BLOCK_SIZE=${BLOCK_SIZE:-32}

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner --multimodal "Launching Aggregated Multimodal + MM Router" "$MODEL_PATH" "$HTTP_PORT"

# TRT-LLM worker: "__internal" suffix hides it from frontend discovery.
python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "${SERVED_MODEL_NAME}__internal" \
  --extra-engine-args "$AGG_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --publish-events-and-metrics \
  --kv-block-size "$BLOCK_SIZE" &

# MM Router Worker: registers with the real model name; does KV-aware routing internally.
(cd "$DYNAMO_HOME" && python3 -m examples.backends.trtllm.mm_router_worker \
  --model "$MODEL_PATH" \
  --model-type "$MODEL_TYPE" \
  --namespace dynamo \
  --component mm_router \
  --endpoint generate \
  --downstream-component tensorrt_llm \
  --downstream-endpoint generate \
  --block-size "$BLOCK_SIZE") &

# Frontend: round-robin to mm_router (KV routing happens inside mm_router, not here).
python3 -m dynamo.frontend --router-mode round-robin &

wait_any_exit
