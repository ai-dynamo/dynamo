#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Setup cleanup trap
cleanup() {
    echo "Cleaning up background processes..."
    kill $DYNAMO_PID 2>/dev/null || true
    wait $DYNAMO_PID 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# Path to the custom template
# The script runs from /workspace/tests/serve directory
TEMPLATE_PATH="fixtures/custom_template.jinja"

# run clear_namespace
python3 -m dynamo.sglang.utils.clear_namespace --namespace dynamo

# run ingress
python3 -m dynamo.frontend --http-port=8000 &
DYNAMO_PID=$!

# run worker with CUSTOM TEMPLATE
python3 -m dynamo.sglang \
  --model-path "Qwen/Qwen3-0.6B" \
  --served-model-name "Qwen/Qwen3-0.6B" \
  --trust-remote-code \
  --skip-tokenizer-init \
  --custom-jinja-template "$TEMPLATE_PATH"
