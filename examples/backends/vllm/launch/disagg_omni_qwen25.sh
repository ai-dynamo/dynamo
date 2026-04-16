#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# 3-stage disaggregated Qwen2.5-Omni text-to-text+audio generation.
# Stage 0: thinker  (GPU 0) — comprehension / text generation (AR)
# Stage 1: talker   (GPU 1) — enhanced text generation (AR)
# Stage 2: code2wav (GPU 0) — audio generation
# Router: orchestrates the 3-stage pipeline, formats response
set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MODEL="${MODEL:-Qwen/Qwen2.5-Omni-7B}"

# Resolve vllm-omni's built-in Qwen2.5-Omni stage config
if [ -z "$STAGE_CONFIG" ]; then
    STAGE_CONFIG="$(python -c "import vllm_omni, os; print(os.path.join(os.path.dirname(vllm_omni.__file__), 'model_executor/stage_configs/qwen2_5_omni.yaml'))" 2>/dev/null | tail -1)"
fi

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --stage-configs-path) STAGE_CONFIG="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
if [ -z "${DYN_NAMESPACE:-}" ]; then
    export DYN_NAMESPACE="dynamo-omni-qwen25-$(date +%s)"
fi
echo "Namespace:   ${DYN_NAMESPACE}"
echo "Stage config: ${STAGE_CONFIG}"
print_launch_banner --no-curl "Disaggregated Qwen2.5-Omni (3-stage, 2 GPUs)" "$MODEL" "$HTTP_PORT"
print_curl_footer <<CURL
curl -s http://localhost:${HTTP_PORT}/v1/chat/completions \\
  -H 'Content-Type: application/json' \\
  -d '{
    "model": "${MODEL}",
    "messages": [{"role": "user", "content": "Hello, say something short."}],
    "stream": false,
    "max_tokens": 64
  }' | jq
CURL

export FLASHINFER_DISABLE_VERSION_CHECK=1
export DYN_DISCOVERY_BACKEND="${DYN_DISCOVERY_BACKEND:-file}"

# Stage 0: thinker (GPU 0) — comprehension / text generation
echo "Starting Stage 0 (thinker)..."
CUDA_VISIBLE_DEVICES=0 DYN_SYSTEM_PORT=8081 \
    python -m dynamo.vllm.omni \
    --model "$MODEL" \
    --stage-id 0 \
    --stage-configs-path "$STAGE_CONFIG" \
    "${EXTRA_ARGS[@]}" &
sleep 30

# Stage 1: talker (GPU 1) — enhanced text generation
echo "Starting Stage 1 (talker)..."
CUDA_VISIBLE_DEVICES=1 DYN_SYSTEM_PORT=8082 \
    python -m dynamo.vllm.omni \
    --model "$MODEL" \
    --stage-id 1 \
    --stage-configs-path "$STAGE_CONFIG" \
    "${EXTRA_ARGS[@]}" &
sleep 30

# Stage 2: code2wav (GPU 1) — audio generation
echo "Starting Stage 2 (code2wav)..."
CUDA_VISIBLE_DEVICES=1 DYN_SYSTEM_PORT=8083 \
    python -m dynamo.vllm.omni \
    --model "$MODEL" \
    --stage-id 2 \
    --stage-configs-path "$STAGE_CONFIG" \
    "${EXTRA_ARGS[@]}" &
sleep 20

# Router — discovers stage workers, orchestrates pipeline, formats response
echo "Starting Router..."
DYN_SYSTEM_PORT=8084 \
    python -m dynamo.vllm.omni \
    --model "$MODEL" \
    --omni-router \
    --stage-configs-path "$STAGE_CONFIG" \
    "${EXTRA_ARGS[@]}" &
sleep 5

# Frontend
echo "Starting Frontend..."
python -m dynamo.frontend &

wait_any_exit
