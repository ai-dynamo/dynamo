#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal vllm serve wrapper for benchmark sweeps.
# Unlike vllm_serve.sh, this does NOT inject embedding cache or KV cache overrides.
# Launched by the sweep orchestrator via: bash vllm_serve_clean.sh --model <model> [extra_args...]

MODEL=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"; shift 2 ;;
        --multimodal-embedding-cache-capacity-gb)
            shift 2 ;;  # silently consume — not used in clean mode
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "ERROR: --model is required" >&2
    exit 1
fi

if [[ "${DYN_PERF_PATCHES:-}" == "1" ]]; then
    exec python -m benchmarks.multimodal.sweep.vllm_serve_patched \
        serve "$MODEL" \
        --max-model-len 16384 \
        "${EXTRA_ARGS[@]}"
else
    exec vllm serve "$MODEL" \
        --max-model-len 16384 \
        "${EXTRA_ARGS[@]}"
fi
