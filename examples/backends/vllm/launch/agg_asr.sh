#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated speech recognition (ASR) serving with Qwen3-ASR
#
# Architecture: Single-worker PD (Prefill-Decode)
# - Frontend: vLLM chat processor (--dyn-chat-processor vllm). The original
#   Qwen3-ASR checkpoints ship vocab.json and merges.txt but no tokenizer.json,
#   which the default Rust preprocessor requires.
# - Worker: Standard vLLM worker with --enable-multimodal
#
# Clients send an audio_url content part to /v1/chat/completions. The model
# replies with "language <Language><asr_text><transcript>".
#
# The stock runtime image decodes WAV/FLAC audio but omits PyAV, which vLLM
# uses to resample audio to the model's 16 kHz rate. To accept other sample
# rates, first run: python -m dynamo.common.utils.install_media_decoders vllm
#
# For streaming transcription over /v1/realtime, see
# agg_realtime_transcription.sh.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MODEL_NAME="${DYN_MODEL_NAME:-Qwen/Qwen3-ASR-1.7B}"
CHAT_PROCESSOR="${DYN_CHAT_PROCESSOR:-vllm}"

# Extra arguments are passed through to the vLLM worker
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME=$2
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] [EXTRA_VLLM_ARGS]"
            echo "Options:"
            echo "  --model <model_name>   Qwen3-ASR checkpoint to serve (default: $MODEL_NAME)"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Any additional arguments are passed through to the vLLM worker."
            echo "Example: $0 --model Qwen/Qwen3-ASR-0.6B --max-model-len 8192"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

trap dynamo_exit_trap EXIT

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)

print_launch_banner --no-curl "Launching Aggregated ASR Serving" "$MODEL_NAME" "$HTTP_PORT" \
    "Chat proc:   $CHAT_PROCESSOR" \
    "Backend:     dynamo.vllm --enable-multimodal" \
    "Media:       audio_url (16 kHz unless PyAV is installed)"

print_curl_footer <<CURL
  curl http://localhost:${HTTP_PORT}/v1/chat/completions \\
    -H 'Content-Type: application/json' \\
    -d '{
      "model": "${MODEL_NAME}",
      "messages": [{"role": "user", "content": [
        {"type": "audio_url", "audio_url": {"url": "https://raw.githubusercontent.com/yuekaizhang/Triton-ASR-Client/main/datasets/mini_en/wav/1221-135766-0002.wav"}}
      ]}],
      "max_tokens": 256
    }'
CURL

# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python -m dynamo.frontend --dyn-chat-processor "$CHAT_PROCESSOR" &

# Extra args from command line come last to allow overrides
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    python -m dynamo.vllm --enable-multimodal --model "$MODEL_NAME" \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

# Exit on first process failure; the EXIT trap tears down the rest
wait_any_exit
