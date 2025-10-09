#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

PREFILL_TP=1
DECODE_TP=1
PREFILL_PP=1

while [[ $# -gt 0 ]]; do
  case $1 in
    --prefill-tp)
      PREFILL_TP="$2"
      shift 2
      ;;
    --prefill-pp)
      PREFILL_PP="$2"
      shift 2
      ;;
    --decode-tp)
      DECODE_TP="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--prefill-tp <size>] [--prefill-pp <size>] [--decode-tp <size>]"
      exit 1
      ;;
  esac
done

echo "Starting disaggregated deployment with Prefill TP=$PREFILL_TP, Prefill PP=$PREFILL_PP, Decode TP=$DECODE_TP"

# Setup cleanup trap
cleanup() {
    echo "Cleaning up background processes..."
    kill $DYNAMO_PID $PREFILL_PID 2>/dev/null || true
    wait $DYNAMO_PID $PREFILL_PID 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# run clear_namespace
python3 -m dynamo.sglang.clear_namespace --namespace dynamo

# run ingress
python3 -m dynamo.frontend --http-port=8000 &
DYNAMO_PID=$!

# run prefill worker
python3 -m dynamo.sglang \
  --model-path Qwen/Qwen3-0.6B \
  --served-model-name Qwen/Qwen3-0.6B \
  --page-size 16 \
  --pp $PREFILL_PP \
  --tp $PREFILL_TP \
  --trust-remote-code \
  --disaggregation-mode prefill \
  --disaggregation-bootstrap-port 12345 \
  --host 0.0.0.0 \
  --disaggregation-transfer-backend nixl &
PREFILL_PID=$!

# run decode worker
CUDA_VISIBLE_DEVICES=1 python3 -m dynamo.sglang \
  --model-path Qwen/Qwen3-0.6B \
  --served-model-name Qwen/Qwen3-0.6B \
  --page-size 16 \
  --tp $DECODE_TP \
  --trust-remote-code \
  --disaggregation-mode decode \
  --disaggregation-bootstrap-port 12345 \
  --host 0.0.0.0 \
  --disaggregation-transfer-backend nixl
