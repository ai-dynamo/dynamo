#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Setup cleanup trap
cleanup() {
    echo "Cleaning up background processes..."
    kill $DYNAMO_PID $PREFILL_PID $PREFILL_ROUTER_PID 2>/dev/null || true
    wait $DYNAMO_PID $PREFILL_PID $PREFILL_ROUTER_PID 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# run ingress
python3 -m dynamo.frontend \
 --http-port=8000 \
 --router-mode kv \
 --kv-overlap-score-weight 0 \
 --router-reset-states &
DYNAMO_PID=$!

# run prefill router
python3 -m dynamo.router \
  --endpoint dynamo.prefill.generate \
  --block-size 64 \
  --router-reset-states \
  --no-track-active-blocks &
PREFILL_ROUTER_PID=$!

# run prefill worker
python3 -m dynamo.sglang \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --served-model-name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --page-size 64 \
  --tp 1 \
  --trust-remote-code \
  --disaggregation-mode prefill \
  --host 0.0.0.0 \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}' \
  --disaggregation-transfer-backend nixl &
PREFILL_PID=$!

# run prefill worker
CUDA_VISIBLE_DEVICES=1 python3 -m dynamo.sglang \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --served-model-name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --page-size 64 \
  --tp 1 \
  --trust-remote-code \
  --disaggregation-mode prefill \
  --host 0.0.0.0 \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5558"}' \
  --disaggregation-transfer-backend nixl &
PREFILL_PID=$!

# run decode worker
CUDA_VISIBLE_DEVICES=3 python3 -m dynamo.sglang \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --served-model-name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --page-size 64 \
  --tp 1 \
  --trust-remote-code \
  --disaggregation-mode decode \
  --host 0.0.0.0 \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5560"}' \
  --disaggregation-transfer-backend nixl &
PREFILL_PID=$!

# run decode worker
CUDA_VISIBLE_DEVICES=2 python3 -m dynamo.sglang \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --served-model-name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --page-size 64 \
  --tp 1 \
  --trust-remote-code \
  --disaggregation-mode decode \
  --host 0.0.0.0 \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5559"}' \
  --disaggregation-transfer-backend nixl
