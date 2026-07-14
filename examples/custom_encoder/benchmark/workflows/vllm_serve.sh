#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

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

exec vllm serve "$MODEL" \
    --max-model-len "${DYN_MAX_MODEL_LEN:-2048}" \
    --max-num-seqs "${DYN_MAX_NUM_SEQS:-64}" \
    --gpu-memory-utilization "${DYN_VLLM_GPU_MEMORY_UTILIZATION:-0.7}" \
    --enable-prefix-caching \
    "${EXTRA_ARGS[@]}"
