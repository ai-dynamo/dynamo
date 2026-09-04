#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Launch the full cascaded voice pipeline (ASR, TTS, vLLM LLM, orchestrator,
# and the dynamo.frontend). Run from a checkout / mounted worktree inside the
# dynamo-riva image, with etcd/nats and the RIVA NIMs already reachable.
#
#   LLM_MODEL           HF model the vLLM worker serves (required)
#   ASR_RIVA_SERVER     ASR NIM gRPC host:port (default: localhost:50051)
#   TTS_RIVA_SERVER     TTS NIM gRPC host:port (default: localhost:50052)
#   VLLM_EXTRA_ARGS     Extra args for the vLLM worker (optional, e.g. "--tensor-parallel-size 2")
#   FRONTEND_EXTRA_ARGS Extra args for dynamo.frontend (optional, e.g. "--http-port 8000")

set -euo pipefail

# Shared launch helper; wait_any_exit tears everything down as soon as any
# worker exits, so a crash surfaces immediately instead of hanging.
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../common/launch_utils.sh" # wait_any_exit

ASR_RIVA_SERVER="${ASR_RIVA_SERVER:-localhost:50051}"
TTS_RIVA_SERVER="${TTS_RIVA_SERVER:-localhost:50052}"
LLM_MODEL="${LLM_MODEL:-}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
FRONTEND_EXTRA_ARGS="${FRONTEND_EXTRA_ARGS:-}"

if [[ -z "${LLM_MODEL}" ]]; then
  echo "LLM_MODEL is required (the HF model the vLLM worker serves)." >&2
  exit 1
fi

ASR_ENDPOINT="dynamo.riva-asr.generate"
TTS_ENDPOINT="dynamo.riva-tts.generate"
LLM_ENDPOINT="dynamo.backend.generate"

echo "Starting ASR worker (riva=${ASR_RIVA_SERVER}) ..."
python -m riva_nim.asr_worker --riva-server "${ASR_RIVA_SERVER}" --endpoint "${ASR_ENDPOINT}" &

echo "Starting TTS worker (riva=${TTS_RIVA_SERVER}) ..."
python -m riva_nim.tts_worker --riva-server "${TTS_RIVA_SERVER}" --endpoint "${TTS_ENDPOINT}" &

echo "Starting vLLM LLM worker (model=${LLM_MODEL}) ..."
# Word-splitting on VLLM_EXTRA_ARGS is intentional so callers can pass flags.
# shellcheck disable=SC2086
python -m dynamo.vllm --model "${LLM_MODEL}" --use-vllm-tokenizer ${VLLM_EXTRA_ARGS} &

echo "Starting orchestrator ..."
python -m riva_nim.orchestrator \
  --asr-endpoint "${ASR_ENDPOINT}" \
  --llm-endpoint "${LLM_ENDPOINT}" \
  --tts-endpoint "${TTS_ENDPOINT}" \
  --llm-model "${LLM_MODEL}" &

echo "Starting dynamo.frontend (realtime WS at /v1/realtime) ..."
# Word-splitting on FRONTEND_EXTRA_ARGS is intentional so callers can pass flags.
# shellcheck disable=SC2086
python -m dynamo.frontend ${FRONTEND_EXTRA_ARGS} &

echo "All components started (ASR, TTS, vLLM, orchestrator, frontend). Ctrl-C to stop."
wait_any_exit
