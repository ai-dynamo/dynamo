#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Multimodal E/PD: separate vision encoder (GPU 0) + combined PD worker (GPU 1).
# GPUs: 2

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

# Default values
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
CHAT_TEMPLATE="qwen2-vl"
PROVIDED_CHAT_TEMPLATE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME=$2
            shift 2
            ;;
        --served-model-name)
            SERVED_MODEL_NAME=$2
            shift 2
            ;;
        --chat-template)
            PROVIDED_CHAT_TEMPLATE=$2
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model <model_name> Specify the model to use (default: $MODEL_NAME)"
            echo "  --served-model-name <served_model_name> Specify the served model name to use (default: empty)"
            echo "  --chat-template <template> Specify the SGLang chat template to use (default: $CHAT_TEMPLATE)"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set CHAT_TEMPLATE if provided
if [[ -n "$PROVIDED_CHAT_TEMPLATE" ]]; then
    CHAT_TEMPLATE="$PROVIDED_CHAT_TEMPLATE"
fi

# Prepare served-model-name argument if provided
SERVED_MODEL_ARG=""
if [[ -n "$SERVED_MODEL_NAME" ]]; then
    SERVED_MODEL_ARG="--served-model-name $SERVED_MODEL_NAME"
fi

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner --multimodal "Launching Multimodal E/PD Workers" "$MODEL_NAME" "$HTTP_PORT"

# run ingress
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python3 -m dynamo.frontend &

# run SGLang multimodal processor
python3 -m dynamo.sglang --multimodal-processor --model-path "$MODEL_NAME" $SERVED_MODEL_ARG --chat-template "$CHAT_TEMPLATE" &

# run SGLang multimodal encode worker
CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.sglang --multimodal-encode-worker --model-path "$MODEL_NAME" $SERVED_MODEL_ARG --chat-template "$CHAT_TEMPLATE" &

# run SGLang multimodal inference worker
# TODO: Remove disable-radix-cache once the issue is fixed.
# See https://github.com/sgl-project/sglang/pull/11203.
CUDA_VISIBLE_DEVICES=1 python3 -m dynamo.sglang \
  --multimodal-worker \
  --model-path "$MODEL_NAME" \
  $SERVED_MODEL_ARG \
  --page-size 16 \
  --tp 1 \
  --trust-remote-code \
  --skip-tokenizer-init \
  --disable-radix-cache \
  --disaggregation-transfer-backend nixl &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
