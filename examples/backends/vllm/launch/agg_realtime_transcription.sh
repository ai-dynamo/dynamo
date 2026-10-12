#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MODEL="mistralai/Voxtral-Mini-4B-Realtime-2602"
HF_OVERRIDES=""

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --hf-overrides)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "Error: --hf-overrides requires a value." >&2
                exit 2
            fi
            HF_OVERRIDES="$2"
            shift 2
            ;;
        --hf-overrides=*)
            HF_OVERRIDES="${1#*=}"
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$HF_OVERRIDES" ]]; then
    echo "Error: select the model's realtime architecture explicitly with --hf-overrides." >&2
    echo "Usage: $0 [--model MODEL] --hf-overrides JSON [vLLM args...]" >&2
    exit 2
fi

trap 'echo Cleaning up...; kill 0' EXIT

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)
print_launch_banner --no-curl "Launching vLLM Realtime Transcription" "$MODEL" "$HTTP_PORT"
print_curl_footer <<TEST
  # Stream an audio file and print its transcription:
  python ${SCRIPT_DIR}/realtime_audio_client.py \\
    --session-type transcription \\
    --url ws://localhost:${HTTP_PORT}/v1/realtime \\
    --model "${MODEL}"
TEST

python -m dynamo.frontend &

echo "Starting Realtime Transcription worker..."
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    python -m dynamo.vllm \
    --realtime \
    --model "$MODEL" \
    --enforce-eager \
    --hf-overrides "$HF_OVERRIDES" \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

wait_any_exit
