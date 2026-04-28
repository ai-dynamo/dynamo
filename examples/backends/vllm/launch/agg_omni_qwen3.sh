#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated Qwen3-Omni-MoE launch (single AsyncOmni instance, all stages colocated).
# Modalities served from one endpoint: text out (multimodal in -> text) and audio out
# (text in -> audio). For DYN-2581 agg-vs-disagg benchmarking.
set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MODEL="${MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner --no-curl "Aggregated Qwen3-Omni (text+audio out, 1-2 GPUs)" "$MODEL" "$HTTP_PORT"
print_curl_footer <<CURL
# chat (text out)
curl -s http://localhost:${HTTP_PORT}/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "${MODEL}",
  "messages": [{"role":"user","content":"Hello in one sentence."}],
  "max_tokens": 64
}' | jq

# audio (TTS-style)
curl -s http://localhost:${HTTP_PORT}/v1/audio/speech -H 'Content-Type: application/json' -d '{
  "model": "${MODEL}",
  "input": "Hello, this is Qwen3-Omni speaking through Dynamo.",
  "voice": "ethan"
}' -o dynamo-audio.wav
CURL

export FLASHINFER_DISABLE_VERSION_CHECK=1
# Upstream Qwen3-Omni stage configs use long max_model_len on the talker stage.
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

python -m dynamo.frontend &
sleep 2

echo "Starting Aggregated Omni worker..."
DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT:-8081}" \
    python -m dynamo.vllm.omni \
    --model "$MODEL" \
    --output-modalities text,audio \
    --media-output-fs-url file:///tmp/dynamo_media \
    --trust-remote-code \
    "${EXTRA_ARGS[@]}" &

wait_any_exit
