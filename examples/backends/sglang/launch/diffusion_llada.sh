#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Diffusion language model (LLaDA2.0). Text generation via iterative refinement.
# GPUs: 1

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

# Model configuration
MODEL_PATH="inclusionAI/LLaDA2.0-mini-preview"

# Diffusion algorithm configuration
DLLM_ALGORITHM="${DLLM_ALGORITHM:-LowConfidence}"
DLLM_ALGORITHM_CONFIG="${DLLM_ALGORITHM_CONFIG:-}"  # Optional: path to YAML config file

# Dynamo configuration
NAMESPACE="${NAMESPACE:-dynamo}"
COMPONENT="${COMPONENT:-backend}"
ENDPOINT="${ENDPOINT:-generate}"
HTTP_PORT="${HTTP_PORT:-8001}"
TP_SIZE="${TP_SIZE:-1}"

print_launch_banner --no-curl "Launching Diffusion LM Worker (LLaDA2.0)" "$MODEL_PATH" "$HTTP_PORT" \
    "Namespace:   $NAMESPACE" \
    "Component:   $COMPONENT" \
    "TP Size:     $TP_SIZE" \
    "Diffusion Algorithm: ${DLLM_ALGORITHM:-LowConfidence}" \
    "Algorithm Config: ${DLLM_ALGORITHM_CONFIG:-default}"

print_curl_footer <<CURL
  curl http://localhost:${HTTP_PORT}/v1/chat/completions \\
    -H 'Content-Type: application/json' \\
    -d '{
      "model": "${MODEL_PATH}",
      "messages": [{"role": "user", "content": "${EXAMPLE_PROMPT}"}],
      "temperature": 0.7,
      "max_tokens": 512
    }'
CURL

# Launch frontend (OpenAI-compatible API server)
echo "Starting Dynamo Frontend on port $HTTP_PORT..."
python -m dynamo.frontend \
    --http-port "$HTTP_PORT" &

# Wait for frontend to start
sleep 2

# Launch diffusion worker
echo "Starting Diffusion LM Worker..."

# Build the command with required arguments
export CUDA_VISIBLE_DEVICES=0
CMD="python -m dynamo.sglang \
    --model-path $MODEL_PATH \
    --tp-size $TP_SIZE \
    --skip-tokenizer-init \
    --trust-remote-code \
    --endpoint dyn://${NAMESPACE}.${COMPONENT}.${ENDPOINT} \
    --enable-metrics \
    --disable-cuda-graph \
    --disable-overlap-schedule \
    --attention-backend triton \
    --dllm-algorithm $DLLM_ALGORITHM"

# Add optional algorithm config if provided
if [ -n "$DLLM_ALGORITHM_CONFIG" ]; then
    CMD="$CMD --dllm-algorithm-config $DLLM_ALGORITHM_CONFIG"
fi

# Execute the command
eval $CMD &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
