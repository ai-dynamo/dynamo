#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# 1E+3PD: 1 encoder + 3 aggregated prefill-decode workers.
# Matches the vLLM EPD blog's architecture for fair comparison.
#
# GPU layout:
#   GPU 0: encoder (vision encoder + projector only)
#   GPU 1: PD worker 1 (prefill + decode, receives embeddings from encoder)
#   GPU 2: PD worker 2
#   GPU 3: PD worker 3
#
# Usage:
#   bash launch_1e3pd.sh --model-path Qwen/Qwen3-VL-30B-A3B-Instruct

export DYNAMO_HOME=${DYNAMO_HOME:-"/workspace"}
export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-VL-30B-A3B-Instruct"}
export AGG_ENGINE_ARGS=${AGG_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3-vl-30b-a3b-instruct/agg.yaml"}
export ENCODE_ENGINE_ARGS=${ENCODE_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3-vl-30b-a3b-instruct/encode.yaml"}
export ENCODE_ENDPOINT=${ENCODE_ENDPOINT:-"dyn://dynamo.tensorrt_llm_encode.generate"}
export MODALITY=${MODALITY:-"multimodal"}
export ALLOWED_LOCAL_MEDIA_PATH=${ALLOWED_LOCAL_MEDIA_PATH:-"/tmp"}
export MAX_FILE_SIZE_MB=${MAX_FILE_SIZE_MB:-50}

# Remaining arguments forwarded from sweep orchestrator
EXTRA_ARGS=("$@")

# Parse --model-path from extra args to keep SERVED_MODEL_NAME in sync
for (( i=0; i<${#EXTRA_ARGS[@]}; i++ )); do
    if [[ "${EXTRA_ARGS[$i]}" == "--model-path" || "${EXTRA_ARGS[$i]}" == "--model" ]]; then
        MODEL_PATH="${EXTRA_ARGS[$((i+1))]}"
        break
    fi
done
export SERVED_MODEL_NAME="$MODEL_PATH"

echo "1E+3PD config: 1 encoder + 3 PD workers (4 GPUs total)"

# Collect PIDs for cleanup
PIDS=()

cleanup() {
    echo "Cleaning up background processes..."
    kill "${PIDS[@]}" 2>/dev/null || true
    wait "${PIDS[@]}" 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# Frontend with KV-aware routing across PD workers
python3 -m dynamo.frontend --router-mode kv &
PIDS+=($!)
DYNAMO_PID=$!

# Encoder on GPU 0
CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$ENCODE_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
  --max-file-size-mb "$MAX_FILE_SIZE_MB" \
  --disaggregation-mode encode \
  "${EXTRA_ARGS[@]}" &
PIDS+=($!)
echo "  Encoder on GPU 0 (PID $!)"

# 3 PD workers on GPUs 1, 2, 3
# Each is an aggregated worker that delegates encoding to the external encoder
for gpu in 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gpu python3 -m dynamo.trtllm \
      --model-path "$MODEL_PATH" \
      --served-model-name "$SERVED_MODEL_NAME" \
      --extra-engine-args "$AGG_ENGINE_ARGS" \
      --modality "$MODALITY" \
      --encode-endpoint "$ENCODE_ENDPOINT" \
      --publish-events-and-metrics \
      "${EXTRA_ARGS[@]}" &
    PIDS+=($!)
    echo "  PD worker on GPU $gpu (PID $!)"
done

wait $DYNAMO_PID
