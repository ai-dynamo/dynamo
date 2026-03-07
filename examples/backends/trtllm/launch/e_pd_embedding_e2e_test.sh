#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# E2E test: Encode + PD with embedding cache for llava-v1.6-mistral-7b-hf.
# GPU 0: Encode worker, GPU 1: PD worker.

export DYNAMO_HOME=${DYNAMO_HOME:-"/workspace"}
export MODEL_PATH=${MODEL_PATH:-"llava-hf/llava-v1.6-mistral-7b-hf"}
export MODEL_REVISION=${MODEL_REVISION:-"52320fb52229"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"$MODEL_PATH"}

# Resolve HF repo ID to local snapshot for the pinned revision
if [[ -n "$MODEL_REVISION" && "$MODEL_PATH" != /* ]]; then
  echo "Resolving $MODEL_PATH revision=$MODEL_REVISION to local cache..."
  RESOLVED_PATH=$(python3 -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$MODEL_PATH', revision='$MODEL_REVISION', local_files_only=True))
" 2>/dev/null)
  if [[ -n "$RESOLVED_PATH" && -d "$RESOLVED_PATH" ]]; then
    echo "Using cached model: $RESOLVED_PATH"
    MODEL_PATH="$RESOLVED_PATH"
  else
    echo "WARNING: pinned revision not cached locally, using MODEL_PATH=$MODEL_PATH as-is"
  fi
fi

EXTRA_PD_ARGS=("$@")

trap 'kill $DYNAMO_PID $ENCODE_PID $PD_PID 2>/dev/null; wait 2>/dev/null' EXIT INT TERM

python3 -m dynamo.frontend &
DYNAMO_PID=$!

CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$DYNAMO_HOME/examples/backends/trtllm/engine_configs/llava-v1.6-mistral-7b-hf/encode.yaml" \
  --modality multimodal \
  --allowed-local-media-path /tmp \
  --custom-jinja-template "$DYNAMO_HOME/examples/backends/trtllm/templates/llava_multimodal.jinja" \
  --disaggregation-mode encode &
ENCODE_PID=$!

CUDA_VISIBLE_DEVICES=1 python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$DYNAMO_HOME/examples/backends/trtllm/engine_configs/llava-v1.6-mistral-7b-hf/agg.yaml" \
  --modality multimodal \
  --encode-endpoint "dyn://dynamo.tensorrt_llm_encode.generate" \
  --custom-jinja-template "$DYNAMO_HOME/examples/backends/trtllm/templates/llava_multimodal.jinja" \
  --disaggregation-mode prefill_and_decode \
  "${EXTRA_PD_ARGS[@]}" &
PD_PID=$!

wait $DYNAMO_PID
