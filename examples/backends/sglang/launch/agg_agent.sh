#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated agent serving with priority-based radix eviction, KV event tracking,
# and reasoning/tool-call parsing.
# GPUs: 2 (default model uses --tp 2)

set -euo pipefail
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"   # build_gpu_mem_args
source "$SCRIPT_DIR/../../../common/launch_utils.sh" # print_launch_banner, wait_any_exit

# Default values
MODEL="zai-org/GLM-4.7-Flash"
TP=2
SERVED_MODEL_NAME=""
TOOL_CALL_PARSER="glm47"
REASONING_PARSER="glm45"
DEFAULT_THINKING_MODE="enabled"

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
        --served-model-name)
            SERVED_MODEL_NAME="$2"
            shift 2
            ;;
        --tool-call-parser)
            TOOL_CALL_PARSER="$2"
            shift 2
            ;;
        --reasoning-parser)
            REASONING_PARSER="$2"
            shift 2
            ;;
        --default-thinking-mode)
            DEFAULT_THINKING_MODE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model-path <name>  Specify model (default: $MODEL)"
            echo "  --tp <n>             Tensor parallelism (default: $TP)"
            echo "  --served-model-name <name>  API model alias (default: model path)"
            echo "  --tool-call-parser <name|none>  Dynamo tool parser (default: $TOOL_CALL_PARSER)"
            echo "  --reasoning-parser <name|none>  Dynamo reasoning parser (default: $REASONING_PARSER)"
            echo "  --default-thinking-mode <mode|none>  Default thinking mode (default: $DEFAULT_THINKING_MODE)"
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

if [[ -z "$SERVED_MODEL_NAME" ]]; then
    SERVED_MODEL_NAME="$MODEL"
fi

PARSER_ARGS=()
if [[ "$TOOL_CALL_PARSER" != "none" ]]; then
    PARSER_ARGS+=(--dyn-tool-call-parser "$TOOL_CALL_PARSER")
fi
if [[ "$REASONING_PARSER" != "none" ]]; then
    PARSER_ARGS+=(--dyn-reasoning-parser "$REASONING_PARSER")
fi
if [[ "$DEFAULT_THINKING_MODE" != "none" ]]; then
    PARSER_ARGS+=(--dyn-default-thinking-mode "$DEFAULT_THINKING_MODE")
fi

GPU_MEM_FRACTION=$(build_sglang_gpu_mem_args)

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
DYN_REQUEST_TRACE="${DYN_REQUEST_TRACE:-1}"
DYN_REQUEST_TRACE_SINKS="${DYN_REQUEST_TRACE_SINKS:-jsonl}"
DYN_REQUEST_TRACE_OUTPUT_PATH="${DYN_REQUEST_TRACE_OUTPUT_PATH:-/tmp/dynamo-request-trace-$(date +%Y%m%d-%H%M%S)-$$.jsonl}"
DYNAMO_API_KEY="${DYNAMO_API_KEY:-dummy}"
export DYN_REQUEST_TRACE DYN_REQUEST_TRACE_SINKS DYN_REQUEST_TRACE_OUTPUT_PATH DYNAMO_API_KEY

print_launch_banner "Launching Aggregated Agent Serving" "$MODEL" "$HTTP_PORT"
echo "Request trace output: $DYN_REQUEST_TRACE_OUTPUT_PATH"

# Frontend with KV routing
python3 -m dynamo.frontend \
  --router-mode kv \
  --enable-anthropic-api &

# Use priority-based radix eviction for agent requests.
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
python3 -m dynamo.sglang \
  --model-path "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --page-size 16 \
  --tp "$TP" \
  --trust-remote-code \
  --radix-eviction-policy priority \
  "${PARSER_ARGS[@]}" \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}' \
  --enable-metrics \
  "${EXTRA_ARGS[@]}" &

wait_any_exit
