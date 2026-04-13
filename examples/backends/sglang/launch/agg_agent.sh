#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated serving with agent controller: session control, sticky routing,
# KV event tracking, and reasoning/tool-call parsing.
# GPUs: 2

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"   # build_gpu_mem_args
source "$SCRIPT_DIR/../../../common/launch_utils.sh" # print_launch_banner, wait_any_exit

MODEL="zai-org/GLM-4.7-Flash"

GPU_MEM_FRACTION=$(build_gpu_mem_args sglang --model "$MODEL")

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Aggregated + Agent Controller" "$MODEL" "$HTTP_PORT"

# Frontend with KV routing and state reset
# Session control activates automatically when requests carry nvext.session_control
python3 -m dynamo.frontend \
  --router-mode kv \
  --router-reset-states &

# Worker with streaming sessions, KV events, and metrics
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
python3 -m dynamo.sglang \
  --model-path "$MODEL" \
  --served-model-name "$MODEL" \
  --page-size 16 \
  --tp 2 \
  --trust-remote-code \
  --skip-tokenizer-init \
  --enable-streaming-session \
  --dyn-reasoning-parser glm45 \
  --dyn-tool-call-parser glm47 \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}' \
  --enable-metrics \
  ${GPU_MEM_FRACTION:+--mem-fraction-static "$GPU_MEM_FRACTION"} &

wait_any_exit
