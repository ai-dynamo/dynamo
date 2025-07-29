#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Environment variables with defaults
# Verifier variables
# TODO: remove the local model path
export MODEL_PATH=${MODEL_PATH:-"meta-llama/Llama-3.3-70B-Instruct"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"meta-llama/Llama-3.3-70B-Instruct"}
export VERIFIER_ENGINE_ARGS=${VERIFIER_ENGINE_ARGS:-"engine_configs/verifier.yaml"}
export VERIFIER_CUDA_VISIBLE_DEVICES=${VERIFIER_CUDA_VISIBLE_DEVICES:-"0,1,2,3"}

# Drafter variables
export NUM_DRAFTERS=${NUM_DRAFTERS:-1}
export DRAFTER_MODEL_PATH=${DRAFTER_MODEL_PATH:-"meta-llama/Llama-3.2-1B-Instruct"}
export DRAFTER_MODEL_NAME=${DRAFTER_MODEL_NAME:-"meta-llama/Llama-3.2-1B-Instruct"}
export DRAFTER_ENGINE_ARGS=${DRAFTER_ENGINE_ARGS:-"engine_configs/drafter.yaml"}
export DRAFTER_CUDA_VISIBLE_DEVICES=${DRAFTER_CUDA_VISIBLE_DEVICES:-"4,5,6,7"}

# Check enough GPUs for drafters
IFS=',' read -ra CUDA_DEVICES <<< "$DRAFTER_CUDA_VISIBLE_DEVICES"
if [ ${#CUDA_DEVICES[@]} -lt $NUM_DRAFTERS ]; then
    echo "Error: Not enough CUDA devices specified for drafters. Need $NUM_DRAFTERS devices, but only ${#CUDA_DEVICES[@]} provided."
    exit 1
fi

# Check num drafters >= 1
if [[ $NUM_DRAFTERS -lt 1 ]]; then
    echo "Error: NUM_DRAFTERS must be >= 1, got: $NUM_DRAFTERS"
    exit 1
fi

# Setup cleanup trap
cleanup() {
    echo "Cleaning up background processes..."
    kill $DYNAMO_PID $VERIFIER_PID "${DRAFTER_PIDS[@]}" 2>/dev/null || true
    wait $DYNAMO_PID $VERIFIER_PID "${DRAFTER_PIDS[@]}" 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# run clear_namespace
python3 utils/clear_namespace.py --namespace dynamo

# run frontend
python3 -m dynamo.frontend --http-port 8000 &
DYNAMO_PID=$!

# run verifier worker with speculative decoding
CUDA_VISIBLE_DEVICES=$VERIFIER_CUDA_VISIBLE_DEVICES \
python3 -m dynamo.verifier \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$VERIFIER_ENGINE_ARGS" &
VERIFIER_PID=$!

# run drafter workers without speculative decoding
start_drafter() {
    local cuda_device=$1
    CUDA_VISIBLE_DEVICES=$cuda_device \
    python3 -m dynamo.drafter \
      --endpoint "dyn://dynamo.tensorrt_llm.generate_draft" \
      --model-path "$DRAFTER_MODEL_PATH" \
      --served-model-name "$DRAFTER_MODEL_NAME" \
      --extra-engine-args "$DRAFTER_ENGINE_ARGS" &
    DRAFTER_PIDS+=($!)
}

for ((i=0; i<$NUM_DRAFTERS-1; i++)); do
    start_drafter ${CUDA_DEVICES[$i]}
done

CUDA_VISIBLE_DEVICES=${CUDA_DEVICES[$NUM_DRAFTERS-1]} \
python3 -m dynamo.drafter \
    --endpoint "dyn://dynamo.tensorrt_llm.generate_draft" \
    --model-path "$DRAFTER_MODEL_PATH" \
    --served-model-name "$DRAFTER_MODEL_NAME" \
    --extra-engine-args "$DRAFTER_ENGINE_ARGS"

