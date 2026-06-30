#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Launch the cascaded voice pipeline workers (ASR, TTS, vLLM LLM, orchestrator).
# Run from this directory inside the dynamo-riva image, with etcd/nats and the
# RIVA NIMs already reachable. The dynamo.frontend is launched separately
# (see README.md).
#
#   LLM_MODEL        HF model the vLLM worker serves (required)
#   ASR_RIVA_SERVER  ASR NIM gRPC host:port (default: localhost:50051)
#   TTS_RIVA_SERVER  TTS NIM gRPC host:port (default: localhost:50052)
#   VLLM_EXTRA_ARGS  Extra args for the vLLM worker (optional, e.g. "--tensor-parallel-size 2")

set -euo pipefail

ASR_RIVA_SERVER="${ASR_RIVA_SERVER:-localhost:50051}"
TTS_RIVA_SERVER="${TTS_RIVA_SERVER:-localhost:50052}"
LLM_MODEL="${LLM_MODEL:-}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

if [[ -z "${LLM_MODEL}" ]]; then
  echo "LLM_MODEL is required (the HF model the vLLM worker serves)." >&2
  exit 1
fi

ASR_ENDPOINT="dynamo.riva-asr.generate"
TTS_ENDPOINT="dynamo.riva-tts.generate"
LLM_ENDPOINT="dynamo.backend.generate"

pids=()
cleanup() { kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup EXIT

echo "Starting ASR worker (riva=${ASR_RIVA_SERVER}) ..."
python -m riva_nim.asr_worker --riva-server "${ASR_RIVA_SERVER}" --endpoint "${ASR_ENDPOINT}" &
pids+=($!)

echo "Starting TTS worker (riva=${TTS_RIVA_SERVER}) ..."
python -m riva_nim.tts_worker --riva-server "${TTS_RIVA_SERVER}" --endpoint "${TTS_ENDPOINT}" &
pids+=($!)

echo "Starting vLLM LLM worker (model=${LLM_MODEL}) ..."
# Word-splitting on VLLM_EXTRA_ARGS is intentional so callers can pass flags.
# shellcheck disable=SC2086
python -m dynamo.vllm --model "${LLM_MODEL}" --use-vllm-tokenizer ${VLLM_EXTRA_ARGS} &
pids+=($!)

echo "Starting orchestrator ..."
python -m riva_nim.orchestrator \
  --asr-endpoint "${ASR_ENDPOINT}" \
  --llm-endpoint "${LLM_ENDPOINT}" \
  --tts-endpoint "${TTS_ENDPOINT}" \
  --llm-model "${LLM_MODEL}" &
pids+=($!)

echo "Workers started. Launch dynamo.frontend separately. Ctrl-C to stop."
wait
