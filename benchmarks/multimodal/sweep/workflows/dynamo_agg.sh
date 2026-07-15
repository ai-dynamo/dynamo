#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
trap 'kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../../examples/common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../../examples/common/launch_utils.sh"

MODEL=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "ERROR: --model is required" >&2
    exit 1
fi

MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
: "${_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES:=4294967296}"
GPU_MEM_ARGS="$(build_vllm_gpu_mem_args)"

python -m dynamo.frontend &

DYN_SYSTEM_PORT="${DYN_WORKER_SYSTEM_PORT:-8082}" \
    python -m dynamo.vllm \
        --model "$MODEL" \
        --disaggregation-mode agg \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --enable-prefix-caching \
        $GPU_MEM_ARGS \
        "${EXTRA_ARGS[@]}" &

wait_any_exit
