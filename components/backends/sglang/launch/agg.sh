#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Setup cleanup trap
cleanup() {
    echo "Cleaning up background processes..."
    kill ${DYN_PID:-$DYNAMO_PID} 2>/dev/null || true
    wait ${DYN_PID:-$DYNAMO_PID} 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# run clear_namespace
python3 -m dynamo.sglang.utils.clear_namespace --namespace dynamo

# run ingress
python3 -m dynamo.frontend --http-port=8000 &
DYN_PID=$!

# run worker
python3 -m dynamo.sglang \
  --model-path Qwen/Qwen3-0.6B \
  --served-model-name Qwen/Qwen3-0.6B \
  --page-size 16 \
  --tp 1 \
  --trust-remote-code \
  --skip-tokenizer-init
