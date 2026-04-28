#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# vLLM-Omni standalone baseline for Qwen3-Omni-MoE.
# Runs upstream `vllm-omni serve` (no Dynamo frontend / router) so we can A/B
# the Dynamo agg/disagg paths against the same model on the same hardware.
#
# Prereqs (cluster image already has these; listed for repro on a fresh box):
#   pip install vllm-omni
#   pip install vllm  # whatever version vllm-omni pins
#
# For DYN-2581 agg-vs-disagg benchmarking.
set -e
trap 'echo Cleaning up...; kill 0' EXIT

MODEL="${MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
PORT="${PORT:-8000}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

echo "================================================="
echo " vLLM-Omni standalone baseline"
echo " Model:  $MODEL"
echo " Port:   $PORT"
echo "================================================="
echo "After ready, smoke test with:"
cat <<CURL
  curl -s http://localhost:${PORT}/v1/chat/completions -H 'Content-Type: application/json' -d '{
    "model": "${MODEL}",
    "messages": [{"role":"user","content":"Hello in one sentence."}],
    "max_tokens": 64
  }' | jq
  curl -s http://localhost:${PORT}/v1/audio/speech -H 'Content-Type: application/json' -d '{
    "model": "${MODEL}",
    "input": "Hello, this is Qwen3-Omni from vllm-omni standalone.",
    "voice": "ethan"
  }' -o vllm-omni-audio.wav
CURL

export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# `vllm-omni serve` is the upstream entrypoint; it auto-discovers the model's
# stage pipeline (thinker/talker/code2wav for Qwen3-Omni) and exposes the same
# OpenAI-compatible endpoints (/v1/chat/completions, /v1/audio/speech, ...).
exec vllm-omni serve \
    "$MODEL" \
    --port "$PORT" \
    --output-modalities text,audio \
    --trust-remote-code \
    "${EXTRA_ARGS[@]}"
