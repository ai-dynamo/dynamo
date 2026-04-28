#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Disaggregated Qwen3-Omni-MoE launch (3 stage workers + router + frontend).
# Stage 0 (thinker, GPU 0) -> text out
# Stage 1 (talker,  GPU 1) -> RVQ codec codes
# Stage 2 (code2wav, GPU 1) -> audio out
# For DYN-2581 agg-vs-disagg benchmarking.
set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MODEL="${MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
STAGE_CONFIG="${STAGE_CONFIG:-$SCRIPT_DIR/stage_configs/qwen3_omni_moe.yaml}"

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
    export DYN_NAMESPACE="dynamo-omni-qwen3-$(date +%s)"
fi
echo "Namespace:    ${DYN_NAMESPACE}"
echo "Stage config: ${STAGE_CONFIG}"

print_launch_banner --no-curl "Disaggregated Qwen3-Omni (3-stage thinker/talker/code2wav, 2 GPUs)" "$MODEL" "$HTTP_PORT"
print_curl_footer <<CURL
# chat (text out — stops at thinker)
curl -s http://localhost:${HTTP_PORT}/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "${MODEL}",
  "messages": [{"role":"user","content":"Hello in one sentence."}],
  "max_tokens": 64
}' | jq

# audio (full pipeline — thinker -> talker -> code2wav)
curl -s http://localhost:${HTTP_PORT}/v1/audio/speech -H 'Content-Type: application/json' -d '{
  "model": "${MODEL}",
  "input": "Hello, this is Qwen3-Omni speaking through disaggregated Dynamo.",
  "voice": "ethan"
}' -o dynamo-audio.wav
CURL

export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Stage 0: Thinker (GPU 0)
echo "Starting Stage 0 (Thinker)..."
CUDA_VISIBLE_DEVICES=0 DYN_SYSTEM_PORT=8081 \
    python -m dynamo.vllm.omni \
    --model "$MODEL" \
    --stage-id 0 \
    --stage-configs-path "$STAGE_CONFIG" \
    --output-modalities text,audio \
    --media-output-fs-url file:///tmp/dynamo_media \
    --trust-remote-code \
    "${EXTRA_ARGS[@]}" &
sleep 20

# Stage 1: Talker (GPU 1)
echo "Starting Stage 1 (Talker)..."
CUDA_VISIBLE_DEVICES=1 DYN_SYSTEM_PORT=8082 \
    python -m dynamo.vllm.omni \
    --model "$MODEL" \
    --stage-id 1 \
    --stage-configs-path "$STAGE_CONFIG" \
    --output-modalities text,audio \
    --media-output-fs-url file:///tmp/dynamo_media \
    --trust-remote-code \
    "${EXTRA_ARGS[@]}" &
sleep 20

# Stage 2: Code2Wav (GPU 1, shares GPU with talker)
echo "Starting Stage 2 (Code2Wav)..."
CUDA_VISIBLE_DEVICES=1 DYN_SYSTEM_PORT=8083 \
    python -m dynamo.vllm.omni \
    --model "$MODEL" \
    --stage-id 2 \
    --stage-configs-path "$STAGE_CONFIG" \
    --output-modalities text,audio \
    --media-output-fs-url file:///tmp/dynamo_media \
    --trust-remote-code \
    "${EXTRA_ARGS[@]}" &
sleep 20

# Router
echo "Starting Router..."
DYN_SYSTEM_PORT=8084 \
    python -m dynamo.vllm.omni \
    --model "$MODEL" \
    --omni-router \
    --stage-configs-path "$STAGE_CONFIG" \
    --output-modalities text,audio \
    --media-output-fs-url file:///tmp/dynamo_media \
    --trust-remote-code \
    "${EXTRA_ARGS[@]}" &
sleep 5

# Frontend
echo "Starting Frontend..."
python -m dynamo.frontend &

wait_any_exit
