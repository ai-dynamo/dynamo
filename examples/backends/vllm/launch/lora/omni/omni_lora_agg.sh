#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated vLLM-Omni image serving with dynamic LoRA support.

set -euo pipefail
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../../../common/launch_utils.sh"

MODEL="${DYN_MODEL_NAME:-Qwen/Qwen-Image}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
SYSTEM_PORT="${DYN_SYSTEM_PORT:-8081}"
TP_SIZE="${DYN_TENSOR_PARALLEL_SIZE:-2}"
MEDIA_OUTPUT_FS_URL="${DYN_MEDIA_OUTPUT_FS_URL:-file:///tmp/dynamo_media}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        -h|--help)
            cat <<USAGE
Usage: $0 [--model <model_name>] [-- EXTRA_OMNI_ARGS]

Options:
  --model <model_name>  Base omni model to serve (default: Qwen/Qwen-Image)
  -h, --help            Show help

Environment variables:
  DYN_MODEL_NAME            Base model name (default: Qwen/Qwen-Image)
  DYN_HTTP_PORT             Frontend HTTP port (default: 8000)
  DYN_SYSTEM_PORT           Omni system/admin port (default: 8081)
  DYN_TENSOR_PARALLEL_SIZE  Tensor parallel size (default: 1)
  DYN_MEDIA_OUTPUT_FS_URL   Media output filesystem URL (default: file:///tmp/dynamo_media)

Any arguments after '--' are passed through to python -m dynamo.vllm.omni.
USAGE
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

print_launch_banner --no-curl "Launching vLLM-Omni Image + LoRA (aggregated)" "$MODEL" "$HTTP_PORT"

echo ""
echo "Configuration:"
echo "  model:        $MODEL"
echo "  frontend:     http://localhost:$HTTP_PORT"
echo "  system API:   http://localhost:$SYSTEM_PORT"
echo "  tensor-par:   $TP_SIZE"
echo ""
echo "LoRA API examples:"
echo "  curl -s -X POST http://localhost:${SYSTEM_PORT}/v1/loras \\\"
echo "    -H 'Content-Type: application/json' \\\"
echo "    -d '{\"lora_name\": \"my-lora\", \"source\": {\"uri\": \"file:///path/to/adapter\"}}' | jq ."
echo "  curl -s http://localhost:${SYSTEM_PORT}/v1/loras | jq ."
echo "  curl -s -X DELETE http://localhost:${SYSTEM_PORT}/v1/loras/my-lora | jq ."

echo ""
echo "Starting frontend..."
python -m dynamo.frontend &
FRONTEND_PID=$!

sleep 2

echo "Starting Omni worker..."
DYN_SYSTEM_ENABLED=true \
DYN_SYSTEM_PORT="$SYSTEM_PORT" \
    python -m dynamo.vllm.omni \
    --model "$MODEL" \
    --output-modalities image \
    --enable-lora \
    --media-output-fs-url "$MEDIA_OUTPUT_FS_URL" \
    --tensor-parallel-size "$TP_SIZE" \
    "${EXTRA_ARGS[@]}" &

wait_any_exit
