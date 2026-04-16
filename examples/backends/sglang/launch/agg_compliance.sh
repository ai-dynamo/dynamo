#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated launcher for Responses / Anthropic compliance tests.
# Frontend exposes /v1/responses, /v1/messages, and /v1/chat/completions on
# the same port so the three compliance suites (OpenResponses bun tests,
# codex exec smoke, claude -p smoke) run sequentially against one server.
# GPUs: 1.

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
SYSTEM_PORT="${DYN_SYSTEM_PORT:-8081}"
GPU_MEM_ARGS=$(build_sglang_gpu_mem_args)

print_launch_banner "Launching Aggregated Compliance Serving" "$MODEL" "$HTTP_PORT"

OTEL_SERVICE_NAME=dynamo-frontend \
python3 -m dynamo.frontend \
  --enable-anthropic-api \
  --http-port "$HTTP_PORT" &

OTEL_SERVICE_NAME=dynamo-worker DYN_SYSTEM_PORT="$SYSTEM_PORT" \
python3 -m dynamo.sglang \
  --model-path "$MODEL" \
  --served-model-name "$MODEL" \
  --page-size 16 \
  --tp 1 \
  --trust-remote-code \
  --skip-tokenizer-init \
  --enable-metrics \
  --disable-piecewise-cuda-graph \
  --dyn-reasoning-parser qwen3 \
  --dyn-tool-call-parser qwen3_coder \
  $GPU_MEM_ARGS &

wait_any_exit
