#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cleanup() {
    echo "Cleaning up background processes..."
    kill $DYNAMO_PID 2>/dev/null || true
    wait $DYNAMO_PID 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# run ingress
dynamo run in=http out=dyn &
DYNAMO_PID=$!

# run worker
python3 worker.py \
    --model-path Qwen/Qwen3-0.6B \
    --extra-engine-args args.json
