#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Setup cleanup trap
cleanup() {
    echo "Cleaning up background processes..."
    kill $DYNAMO_PID $PREFILL_PID $HTTP_SERVER_PID 2>/dev/null || true
    wait $DYNAMO_PID $PREFILL_PID $HTTP_SERVER_PID 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# run clear_namespace
python3 -m dynamo.sglang.utils.clear_namespace --namespace dynamo

# run ingress
python3 -m dynamo.frontend --http-port=8000 &
DYNAMO_PID=$!

# run http server
python3 -m dynamo.sglang.utils.sgl_http_server --namespace dynamo &
HTTP_SERVER_PID=$!

# Set the expert distribution recording directory
mkdir -p /tmp/sglang_expert_distribution_record
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/tmp/sglang_expert_distribution_record

# run prefill worker
python3 -m dynamo.sglang.worker \
  --model-path silence09/DeepSeek-R1-Small-2layers \
  --served-model-name silence09/DeepSeek-R1-Small-2layers \
  --tp 2 \
  --dp-size 2 \
  --enable-dp-attention \
  --trust-remote-code \
  --skip-tokenizer-init \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend nixl \
  --expert-distribution-recorder-mode stat \
  --port 30000 &
PREFILL_PID=$!

# run decode worker
CUDA_VISIBLE_DEVICES=2,3 python3 -m dynamo.sglang.decode_worker \
  --model-path silence09/DeepSeek-R1-Small-2layers \
  --served-model-name silence09/DeepSeek-R1-Small-2layers \
  --tp 2 \
  --dp-size 2 \
  --enable-dp-attention \
  --trust-remote-code \
  --skip-tokenizer-init \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend nixl \
  --expert-distribution-recorder-mode stat \
  --port 31000
