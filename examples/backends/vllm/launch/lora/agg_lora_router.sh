#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
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

# Set deterministic hash for KV event IDs
export PYTHONHASHSEED=0

# Common configuration
MODEL="Qwen/Qwen3-0.6B"
BLOCK_SIZE=64

SYSTEM_PORT1="${DYN_SYSTEM_PORT1:-8081}"
SYSTEM_PORT2="${DYN_SYSTEM_PORT2:-8082}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner --no-curl "Launching Aggregated + LoRA + KV Routing (2 GPUs)" "$MODEL" "$HTTP_PORT"
echo ""
echo "Once running, test with:"
echo ""
echo "  # Check available models"
echo "  curl http://localhost:${HTTP_PORT}/v1/models | jq ."
echo ""
echo "  # Load LoRA to both instances (using S3 URI)"
echo "  curl -s -X POST http://localhost:${SYSTEM_PORT1}/v1/loras \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"lora_name\": \"codelion/Qwen3-0.6B-accuracy-recovery-lora\","
echo "         \"source\": {\"uri\": \"s3://my-loras/codelion/Qwen3-0.6B-accuracy-recovery-lora\"}}' | jq ."
echo ""
echo "  curl -s -X POST http://localhost:${SYSTEM_PORT2}/v1/loras \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"lora_name\": \"codelion/Qwen3-0.6B-accuracy-recovery-lora\","
echo "         \"source\": {\"uri\": \"s3://my-loras/codelion/Qwen3-0.6B-accuracy-recovery-lora\"}}' | jq ."
echo ""
echo "  # Test LoRA inference"
echo "  curl http://localhost:${HTTP_PORT}/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"codelion/Qwen3-0.6B-accuracy-recovery-lora\","
echo "         \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}],"
echo "         \"max_tokens\": 32}' | jq ."
echo ""
echo "=========================================="

# run frontend + KV router
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python -m dynamo.frontend \
    --router-mode kv \
    --router-reset-states &

# run workers
# --enforce-eager is added for quick deployment. for production use, need to remove this flag
# TODO: use build_vllm_gpu_mem_args to measure VRAM instead of relying on vLLM defaults
DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=${SYSTEM_PORT1} \
CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.vllm \
    --model $MODEL \
    --block-size $BLOCK_SIZE \
    --enforce-eager \
    --enable-lora \
    --max-lora-rank 64 \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}' &

DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=${SYSTEM_PORT2} \
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
CUDA_VISIBLE_DEVICES=1 python3 -m dynamo.vllm \
    --model $MODEL \
    --block-size $BLOCK_SIZE \
    --enforce-eager \
    --enable-lora \
    --max-lora-rank 64 \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}' &

# Sample output after running LoRA inference curl request twice.
# usage.prompt_tokens_details.cached_tokens is the number of tokens that were cached from the previous request.
: <<'SAMPLE_OUTPUT'
{
  "id": "chatcmpl-0cf880c2-fe98-45c4-9c76-84c3ad1a56cc",
  "choices": [
    {
      "index": 0,
      "message": {
        "content": "<think>\nOkay, so I need to develop a character background for a character named Elara. Let me start by understanding the requirements. The user wants",
        "role": "assistant",
        "reasoning_content": null
      },
      "finish_reason": "length"
    }
  ],
  "created": 1765230243,
  "model": "codelion/Qwen3-0.6B-accuracy-recovery-lora",
  "object": "chat.completion",
  "usage": {
    "prompt_tokens": 196,
    "completion_tokens": 30,
    "total_tokens": 226,
    "prompt_tokens_details": {
      "audio_tokens": null,
      "cached_tokens": 192              # tokens that were cached from the previous request.
    }
  },
  "nvext": {
    "worker_id": {
      "prefill_worker_id": 7587891281668871552,
      "decode_worker_id": 7587891281668871552
    }
  }
}
SAMPLE_OUTPUT

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
