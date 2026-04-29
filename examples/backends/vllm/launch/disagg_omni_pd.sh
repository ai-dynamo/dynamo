#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# 4-stage PD disaggregated Qwen2.5-Omni text-to-text+audio generation.
# Stage 0: thinker_prefill (GPU 0) — processes prompt, exports KV via NixlConnector
# Stage 1: thinker_decode  (GPU 1) — resumes from KV, generates tokens
# Stage 2: talker           (GPU 1) — enhanced text generation (AR)
# Stage 3: code2wav         (GPU 1) — audio generation
# Router: orchestrates the 4-stage pipeline, formats response
set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MODEL="${MODEL:-Qwen/Qwen2.5-Omni-7B}"
STAGE_CONFIG="${STAGE_CONFIG:-$SCRIPT_DIR/stage_configs/qwen2_5_omni_pd.yaml}"

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
    export DYN_NAMESPACE="dynamo-omni-pd-$(date +%s)"
fi
echo "Namespace:   ${DYN_NAMESPACE}"
echo "Stage config: ${STAGE_CONFIG}"
print_launch_banner --no-curl "PD Disaggregated Qwen2.5-Omni (4-stage, 2 GPUs)" "$MODEL" "$HTTP_PORT"
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
export DYN_REQUEST_PLANE="${DYN_REQUEST_PLANE:-tcp}"
export DYN_EVENT_PLANE="${DYN_EVENT_PLANE:-zmq}"

# Stage 0: thinker prefill (GPU 0) — processes prompt, exports KV
echo "Starting Stage 0 (thinker_prefill)..."
CUDA_VISIBLE_DEVICES=0 DYN_SYSTEM_PORT=8081 VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
    python -m dynamo.vllm.omni \
    --model "$MODEL" \
    --stage-id 0 \
    --stage-configs-path "$STAGE_CONFIG" \
    "${EXTRA_ARGS[@]}" &
sleep 30

# Stage 1: thinker decode (GPU 1) — resumes from prefill KV
echo "Starting Stage 1 (thinker_decode)..."
CUDA_VISIBLE_DEVICES=1 DYN_SYSTEM_PORT=8082 VLLM_NIXL_SIDE_CHANNEL_PORT=5610 \
    python -m dynamo.vllm.omni \
    --model "$MODEL" \
    --stage-id 1 \
    --stage-configs-path "$STAGE_CONFIG" \
    "${EXTRA_ARGS[@]}" &
sleep 30

# Stage 2: talker (GPU 1)
echo "Starting Stage 2 (talker)..."
CUDA_VISIBLE_DEVICES=1 DYN_SYSTEM_PORT=8083 \
    python -m dynamo.vllm.omni \
    --model "$MODEL" \
    --stage-id 2 \
    --stage-configs-path "$STAGE_CONFIG" \
    "${EXTRA_ARGS[@]}" &
sleep 30

# Stage 3: code2wav (GPU 1)
echo "Starting Stage 3 (code2wav)..."
CUDA_VISIBLE_DEVICES=1 DYN_SYSTEM_PORT=8084 \
    python -m dynamo.vllm.omni \
    --model "$MODEL" \
    --stage-id 3 \
    --stage-configs-path "$STAGE_CONFIG" \
    "${EXTRA_ARGS[@]}" &
sleep 20

# Router — discovers stage workers, orchestrates pipeline, formats response
echo "Starting Router..."
DYN_SYSTEM_PORT=8085 \
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
