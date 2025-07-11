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

# run clear_namespace
python3 utils/clear_namespace.py --namespace dynamo

# run ingress
dynamo run in=http out=dyn --http-port=8000 &
DYNAMO_PID=$!

# run prefill worker
python3 components/worker.py \
  --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --served-model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --extra-engine-args  configs/prefill.yaml \
  --disaggregation-mode prefill \
  --disaggregation-strategy prefill_first&
PREFILL_PID=$!

# run decode worker
python3 components/worker.py \
  --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --served-model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --extra-engine-args  configs/decode.yaml \
  --disaggregation-mode decode \
  --disaggregation-strategy prefill_first