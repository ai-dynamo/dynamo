#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../../common/launch_utils.sh"

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
print_launch_banner --no-curl "Launching Aggregated Serving + LoRA (1 GPU)" "$MODEL" "$HTTP_PORT"
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

# ---- Tunable (override via env vars) ----
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"

# Default KV cache cap from profiling (2x safety over min=471 MiB); ~4.0 GiB peak VRAM
# Profiler/test framework overrides via env
: "${_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES:=941712000}"
GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)
DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=${SYSTEM_PORT} \
    python -m dynamo.vllm --model "$MODEL" --enforce-eager \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_CONCURRENT_SEQS" \
    $GPU_MEM_ARGS \
    --enable-lora \
    --max-lora-rank 64 &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
