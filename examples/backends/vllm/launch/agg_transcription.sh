#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated speech-to-text model serving.
# GPUs: 1

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MODEL="openai/whisper-tiny"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--model <name>] [vLLM options]"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)
print_launch_banner --no-curl "Launching Transcription Worker (1 GPU)" "$MODEL" "$HTTP_PORT"
print_curl_footer <<CURL
  curl http://localhost:${HTTP_PORT}/v1/audio/transcriptions \
    -F file=@sample.wav \
    -F model="${MODEL}" \
    -F response_format=json
CURL

python3 -m dynamo.frontend &

DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    python3 -m dynamo.vllm \
    --transcription-worker \
    --model "$MODEL" \
    --no-enable-prefix-caching \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

wait_any_exit
