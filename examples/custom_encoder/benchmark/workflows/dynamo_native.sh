#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
trap 'kill 0 2>/dev/null || true' EXIT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
source "$REPO_ROOT/examples/common/launch_utils.sh"

MODEL=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"; shift 2 ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done
if [[ -z "$MODEL" ]]; then
    echo "ERROR: --model is required" >&2
    exit 2
fi

export DYN_REQUEST_PLANE=tcp
export DYN_TCP_MAX_MESSAGE_SIZE=209715200
export DYN_HTTP_BODY_LIMIT_MB=200

python -m dynamo.frontend &
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT:-8081}" \
python -m dynamo.vllm \
    --model "$MODEL" \
    --enable-multimodal \
    --max-model-len "${DYN_MAX_MODEL_LEN:-2048}" \
    --max-num-seqs "${DYN_MAX_NUM_SEQS:-64}" \
    --gpu-memory-utilization "${DYN_VLLM_GPU_MEMORY_UTILIZATION:-0.7}" \
    --enable-prefix-caching \
    "${EXTRA_ARGS[@]}" &

wait_any_exit
