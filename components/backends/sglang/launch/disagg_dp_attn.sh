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
python3 -m dynamo.frontend --http-port=8000 &
DYNAMO_PID=$!

# run prefill worker
python3 -m dynamo.sglang \
  --model-path silence09/DeepSeek-R1-Small-2layers \
  --served-model-name silence09/DeepSeek-R1-Small-2layers \
  --tp 2 \
  --dp-size 2 \
  --page-size 16 \
  --enable-dp-attention \
  --trust-remote-code \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend nixl \
  --load-balance-method round_robin \
  --port 30000 &
PREFILL_PID=$!

# run decode worker
CUDA_VISIBLE_DEVICES=2,3 python3 -m dynamo.sglang \
  --model-path silence09/DeepSeek-R1-Small-2layers \
  --served-model-name silence09/DeepSeek-R1-Small-2layers \
  --tp 2 \
  --dp-size 2 \
  --page-size 16 \
  --enable-dp-attention \
  --trust-remote-code \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend nixl \
  --prefill-round-robin-balance \
  --port 31000
