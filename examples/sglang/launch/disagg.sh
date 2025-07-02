#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Setup cleanup trap
cleanup() {
    echo "Cleaning up background processes..."
    kill $DYNAMO_PID $PREFILL_PID 2>/dev/null || true
    wait $DYNAMO_PID $PREFILL_PID 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# run ingress
dynamo run in=http out=dyn &
DYNAMO_PID=$!

# run prefill worker
python3 components/worker_inc.py \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --served-model-name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --page-size 16 \
  --tp 1 \
  --trust-remote-code \
  --skip-tokenizer-init \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend nixl &
PREFILL_PID=$!

# run decode worker
CUDA_VISIBLE_DEVICES=1 python3 components/decode_worker_inc.py \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --served-model-name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --page-size 16 \
  --tp 1 \
  --trust-remote-code \
  --skip-tokenizer-init \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend nixl