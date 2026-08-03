#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../../common/launch_utils.sh"
trap 'dynamo_exit_trap' EXIT

export DYN_LORA_ENABLED=true
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
HF_LORA_REPO="${HF_LORA_REPO:-codelion/Qwen3-0.6B-accuracy-recovery-lora}"
LORA_NAME="${LORA_NAME:-${HF_LORA_REPO%%@*}}"
LORA_URI="${LORA_URI:-hf://$HF_LORA_REPO}"
SYSTEM_PORT="${DYN_SYSTEM_PORT:-8081}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"

: "${_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES:=941712000}"
GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)

print_launch_banner --no-curl "Launching Aggregated Serving + Hugging Face LoRA (1 GPU)" "$MODEL" "$HTTP_PORT" \
    "LoRA:       $LORA_URI" \
    "System API: http://localhost:$SYSTEM_PORT"

cat <<CURL_EOF

After the worker reports ready, load the adapter:

  curl --fail-with-body -sS -X POST http://localhost:${SYSTEM_PORT}/v1/loras \\
    -H 'Content-Type: application/json' \\
    -d '{"lora_name":"${LORA_NAME}","source":{"uri":"${LORA_URI}"}}'

Wait for the adapter to appear in /v1/models, then run inference:

  curl --fail-with-body -sS -X POST http://localhost:${HTTP_PORT}/v1/chat/completions \\
    -H 'Content-Type: application/json' \\
    -d '{"model":"${LORA_NAME}","messages":[{"role":"user","content":"What is deep learning?"}],"max_tokens":32}'

The first LoRA inference can take longer while vLLM initializes its LoRA kernels.
==========================================
CURL_EOF

python3 -m dynamo.frontend &

# shellcheck disable=SC2086 # GPU_MEM_ARGS intentionally contains multiple CLI arguments.
DYN_SYSTEM_PORT="$SYSTEM_PORT" \
python3 -m dynamo.vllm \
    --model "$MODEL" \
    --enforce-eager \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_CONCURRENT_SEQS" \
    $GPU_MEM_ARGS \
    --enable-lora \
    --max-lora-rank 64 &

wait_any_exit
