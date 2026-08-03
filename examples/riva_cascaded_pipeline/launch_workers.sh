#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Launch Dynamo model endpoints for the Nemotron Voice Agent Blueprint.
# ASR and TTS NIMs must already be reachable over Riva gRPC.

set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../common/gpu_utils.sh"
source "$SCRIPT_DIR/../common/launch_utils.sh"

ASR_RIVA_SERVER="${ASR_RIVA_SERVER:-localhost:50152}"
TTS_RIVA_SERVER="${TTS_RIVA_SERVER:-localhost:50151}"
ASR_MODEL_NAME="${ASR_MODEL_NAME:-nemotron-asr-streaming}"
TTS_MODEL_NAME="${TTS_MODEL_NAME:-nvidia/magpie-tts-multilingual}"
LLM_MODEL_PATH="${LLM_MODEL_PATH:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8}"
LLM_MODEL_NAME="${LLM_MODEL_NAME:-nvidia/nemotron-3-nano}"
LLM_TP_SIZE="${LLM_TP_SIZE:-2}"
LLM_GPU_DEVICES="${LLM_GPU_DEVICES:-2,3}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
FRONTEND_EXTRA_ARGS="${FRONTEND_EXTRA_ARGS:-}"
GPU_MEM_ARGS="$(build_vllm_gpu_mem_args)"
export VLLM_USE_FLASHINFER_MOE_FP8="${VLLM_USE_FLASHINFER_MOE_FP8:-1}"

print_launch_banner --no-curl \
  "Launching Blueprint-compatible Dynamo model endpoints" "$LLM_MODEL_NAME" "$HTTP_PORT" \
  "Realtime ASR: ws://localhost:${HTTP_PORT}/v1/realtime" \
  "LLM:          http://localhost:${HTTP_PORT}/v1/chat/completions" \
  "TTS:          http://localhost:${HTTP_PORT}/v1/audio/speech"

python -m riva_nim.asr_worker \
  --riva-server "$ASR_RIVA_SERVER" --model-name "$ASR_MODEL_NAME" &

python -m riva_nim.tts_worker \
  --riva-server "$TTS_RIVA_SERVER" --model-name "$TTS_MODEL_NAME" \
  --sample-rate-hz 24000 &

# Word splitting is intentional for optional CLI argument strings.
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="$LLM_GPU_DEVICES" python -m dynamo.vllm \
  --model "$LLM_MODEL_PATH" \
  --served-model-name "$LLM_MODEL_NAME" \
  --tensor-parallel-size "$LLM_TP_SIZE" \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3 \
  --kv-cache-dtype fp8 \
  --max-model-len 32768 \
  --max-num-seqs 256 \
  --use-vllm-tokenizer \
  ${GPU_MEM_ARGS} ${VLLM_EXTRA_ARGS} &

# shellcheck disable=SC2086
python -m dynamo.frontend --http-port "$HTTP_PORT" ${FRONTEND_EXTRA_ARGS} &

wait_any_exit
