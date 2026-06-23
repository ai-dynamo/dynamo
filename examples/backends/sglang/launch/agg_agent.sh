#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated agent serving with trajectory radix tagging, KV event tracking,
# and reasoning/tool-call parsing.
# GPUs: 2 (default model uses --tp 2)

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"   # build_gpu_mem_args
source "$SCRIPT_DIR/../../../common/launch_utils.sh" # print_launch_banner, wait_any_exit

# Default values
MODEL="zai-org/GLM-4.7-Flash"
TP=2

# Parse command line arguments
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL="$2"
            shift 2
            ;;
        --tp)
            TP="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model-path <name>  Specify model (default: $MODEL)"
            echo "  --tp <n>             Tensor parallelism (default: $TP)"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Additional SGLang/Dynamo flags can be passed and will be forwarded"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

GPU_MEM_FRACTION=$(build_sglang_gpu_mem_args)

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
DYN_REQUEST_TRACE="${DYN_REQUEST_TRACE:-1}"
DYN_REQUEST_TRACE_SINKS="${DYN_REQUEST_TRACE_SINKS:-jsonl}"
DYN_REQUEST_TRACE_OUTPUT_PATH="${DYN_REQUEST_TRACE_OUTPUT_PATH:-/tmp/dynamo-request-trace-$(date +%Y%m%d-%H%M%S)-$$.jsonl}"
DYNAMO_API_KEY="${DYNAMO_API_KEY:-dummy}"
export DYN_REQUEST_TRACE DYN_REQUEST_TRACE_SINKS DYN_REQUEST_TRACE_OUTPUT_PATH DYNAMO_API_KEY

print_launch_banner "Launching Aggregated Agent Serving" "$MODEL" "$HTTP_PORT"
echo "Request trace output: $DYN_REQUEST_TRACE_OUTPUT_PATH"

# Frontend with KV routing and state reset
python3 -m dynamo.frontend \
  --router-mode kv \
  --router-reset-states \
  --enable-anthropic-api &

# Requires SGLang from source until --enable-session-radix-cache is released.
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
python3 -m dynamo.sglang \
  --model-path "$MODEL" \
  --served-model-name "$MODEL" \
  --page-size 16 \
  --tp "$TP" \
  --trust-remote-code \
  --enable-session-radix-cache \
  --radix-eviction-policy priority \
  --skip-tokenizer-init \
  --dyn-reasoning-parser glm45 \
  --dyn-tool-call-parser glm47 \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}' \
  --enable-metrics \
  "${EXTRA_ARGS[@]}" &

wait_any_exit
