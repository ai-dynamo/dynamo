#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated serving with LoRA support (SGLang backend).
# GPUs: 1
# Prerequisites: ./setup_minio.sh (starts MinIO, uploads LoRA)

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../../common/launch_utils.sh"

# S3/MinIO credentials
export AWS_ENDPOINT=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export AWS_REGION=us-east-1
export AWS_ALLOW_HTTP=true

# Dynamo LoRA configuration
export DYN_LORA_ENABLED=true
export DYN_LORA_PATH=/tmp/dynamo_loras_minio
mkdir -p "$DYN_LORA_PATH"

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
LORA_NAME="${LORA_NAME:-codelion/Qwen3-0.6B-accuracy-recovery-lora}"
SYSTEM_PORT="${DYN_SYSTEM_PORT:-8081}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
GPU_MEM_ARGS=$(build_sglang_gpu_mem_args)

print_launch_banner --no-curl "Launching Aggregated Serving + LoRA (1 GPU)" "$MODEL" "$HTTP_PORT"
echo ""
echo "Once running, test with:"
echo "  curl -s -X POST http://localhost:${SYSTEM_PORT}/v1/loras \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"lora_name\": \"${LORA_NAME}\", \"source\": {\"uri\": \"s3://my-loras/${LORA_NAME}\"}}' | jq ."
echo ""
echo "  curl http://localhost:${HTTP_PORT}/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"${LORA_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"What is deep learning?\"}], \"max_tokens\": 300}' | jq ."
echo "=========================================="

# Frontend
python3 -m dynamo.frontend &

# Worker
DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=${SYSTEM_PORT} \
python3 -m dynamo.sglang \
  --model-path "$MODEL" \
  --served-model-name "$MODEL" \
  --page-size 16 \
  --tp 1 \
  --trust-remote-code \
  --skip-tokenizer-init \
  --enable-lora \
  --max-lora-rank 32 \
  --lora-target-modules all \
  $GPU_MEM_ARGS &

wait_any_exit
