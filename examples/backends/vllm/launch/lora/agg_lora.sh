#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

export AWS_ENDPOINT=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export AWS_REGION=us-east-1
export AWS_ALLOW_HTTP=true

# Dynamo LoRA Configuration
export DYN_LORA_ENABLED=true
export DYN_LORA_PATH=/tmp/dynamo_loras_minio

mkdir -p $DYN_LORA_PATH

MODEL="Qwen/Qwen3-0.6B"
SYSTEM_PORT="${DYN_SYSTEM_PORT1:-8081}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
echo "=========================================="
echo "Launching Aggregated Serving + LoRA (1 GPU)"
echo "=========================================="
echo "Model:       $MODEL"
echo "Frontend:    http://localhost:$HTTP_PORT"
echo "=========================================="
echo ""
echo "Once running, test with:"
echo ""
echo "  # Check available models"
echo "  curl http://localhost:${HTTP_PORT}/v1/models | jq ."
echo ""
echo "  # Load LoRA (using S3 URI)"
echo "  curl -s -X POST http://localhost:${SYSTEM_PORT}/v1/loras \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"lora_name\": \"codelion/Qwen3-0.6B-accuracy-recovery-lora\","
echo "         \"source\": {\"uri\": \"s3://my-loras/codelion/Qwen3-0.6B-accuracy-recovery-lora\"}}' | jq ."
echo ""
echo "  # Test LoRA inference"
echo "  curl http://localhost:${HTTP_PORT}/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"codelion/Qwen3-0.6B-accuracy-recovery-lora\","
echo "         \"messages\": [{\"role\": \"user\", \"content\": \"What is deep learning?\"}],"
echo "         \"max_tokens\": 300, \"temperature\": 0.0}' | jq ."
echo ""
echo "  # Test base model inference (for comparison)"
echo "  curl http://localhost:${HTTP_PORT}/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"${MODEL}\","
echo "         \"messages\": [{\"role\": \"user\", \"content\": \"What is deep learning?\"}],"
echo "         \"max_tokens\": 300, \"temperature\": 0.0}' | jq ."
echo ""
echo "  # Unload LoRA"
echo "  curl -X DELETE http://localhost:${SYSTEM_PORT}/v1/loras/codelion/Qwen3-0.6B-accuracy-recovery-lora"
echo ""
echo "=========================================="

# run ingress
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var.
python -m dynamo.frontend &

# run worker
# --enforce-eager is added for quick deployment. for production use, need to remove this flag
GPU_MEM_ARGS=()
if [[ -n "${DYN_GPU_MEMORY_FRACTION_OVERRIDE:-}" ]]; then
    GPU_MEM_ARGS=("--gpu-memory-utilization" "$DYN_GPU_MEMORY_FRACTION_OVERRIDE")
fi

DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=${SYSTEM_PORT} \
    python -m dynamo.vllm --model "$MODEL" --enforce-eager \
    "${GPU_MEM_ARGS[@]}" \
    --enable-lora \
    --max-lora-rank 64
