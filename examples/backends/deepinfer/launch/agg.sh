#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated serving with the experimental DeepInfer backend.
# GPUs: --context-parallel-size (default 8), SM100 only

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh" # print_launch_banner, wait_any_exit

MODEL="${MODEL:-deepseek-ai/DeepSeek-V4.1-Flash-1017}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model <id-or-dir>  Model to serve (default: $MODEL)"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Any additional options are passed through to dynamo.deepinfer, e.g."
            echo "  --expert-parallel-size 4 --context-parallel-size 4"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching DeepInfer Aggregated Serving" "$MODEL" "$HTTP_PORT"

# run frontend
python3 -m dynamo.frontend &

# run DeepInfer worker
python3 -m dynamo.deepinfer \
  --model "$MODEL" \
  "${EXTRA_ARGS[@]}" &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
