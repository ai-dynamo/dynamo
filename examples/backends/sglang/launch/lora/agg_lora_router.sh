#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated serving with LoRA support and KV-aware routing (SGLang backend).
# GPUs: 2
# Prerequisites: ./setup_minio.sh (starts MinIO, uploads LoRA)
#
# The workers publish KV events in the "dynamo" format, which carries the LoRA
# adapter name so the router can tell adapter blocks from base-model blocks.

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

# Set deterministic hash for KV event IDs
export PYTHONHASHSEED=0

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
LORA_NAME="${LORA_NAME:-codelion/Qwen3-0.6B-accuracy-recovery-lora}"
PAGE_SIZE=16
SYSTEM_PORT1="${DYN_SYSTEM_PORT1:-8081}"
SYSTEM_PORT2="${DYN_SYSTEM_PORT2:-8082}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
GPU_MEM_ARGS=$(build_sglang_gpu_mem_args)

print_launch_banner --no-curl "Launching Aggregated + LoRA + KV Routing (2 GPUs)" "$MODEL" "$HTTP_PORT"
echo ""
echo "Once running, test with:"
echo ""
echo "  # Load LoRA to both workers"
for port in "$SYSTEM_PORT1" "$SYSTEM_PORT2"; do
  echo "  curl -s -X POST http://localhost:${port}/v1/loras \\"
  echo "    -H 'Content-Type: application/json' \\"
  echo "    -d '{\"lora_name\": \"${LORA_NAME}\", \"source\": {\"uri\": \"s3://my-loras/${LORA_NAME}\"}}' | jq ."
done
echo ""
echo "  # Send the same LoRA request twice; the second one goes to the same worker"
echo "  curl http://localhost:${HTTP_PORT}/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"${LORA_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"What is deep learning?\"}], \"max_tokens\": 32}' | jq ."
echo "=========================================="

# Frontend + KV router
python3 -m dynamo.frontend --router-mode kv &

# Workers
run_worker() {
  local gpu=$1 system_port=$2 kv_port=$3
  DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=${system_port} \
  CUDA_VISIBLE_DEVICES=${gpu} \
  python3 -m dynamo.sglang \
    --model-path "$MODEL" \
    --served-model-name "$MODEL" \
    --page-size "$PAGE_SIZE" \
    --tp 1 \
    --trust-remote-code \
    --skip-tokenizer-init \
    --enable-lora \
    --max-lora-rank 64 \
    --lora-target-modules all \
    --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${kv_port}\",\"format\":\"dynamo\"}" \
    $GPU_MEM_ARGS &
}

run_worker 0 "$SYSTEM_PORT1" 5557
run_worker 1 "$SYSTEM_PORT2" 5558

wait_any_exit
