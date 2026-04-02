#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shared GPU utility functions for launch scripts (source, don't execute).
#
# Usage:
#   source "$(dirname "$(readlink -f "$0")")/../common/gpu_utils.sh"
#   # or with SCRIPT_DIR already set:
#   source "$SCRIPT_DIR/../common/gpu_utils.sh"
#
# Functions (all return via stdout):
#   build_gpu_mem_args <engine> [--workers-per-gpu N]
#       Returns engine-specific CLI args for GPU memory control based on
#       environment variable overrides. Empty if no overrides.
#
#       vLLM:   _PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES   → --kv-cache-memory-bytes N --gpu-memory-utilization 0.01
#       SGLang: _PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS → --max-total-tokens N
#
# Usage:
#   GPU_MEM_ARGS=$(build_gpu_mem_args sglang)
#   python -m dynamo.sglang --model-path "$MODEL" $GPU_MEM_ARGS &
#
#   GPU_MEM_ARGS=$(build_gpu_mem_args vllm)
#   python -m dynamo.vllm --model "$MODEL" $GPU_MEM_ARGS &
build_gpu_mem_args() {
    local engine="${1:?usage: build_gpu_mem_args <engine> [--workers-per-gpu N]}"
    shift

    local workers_per_gpu=1
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --workers-per-gpu) workers_per_gpu="$2"; shift 2 ;;
            *) echo "build_gpu_mem_args: unknown option '$1'" >&2; return 1 ;;
        esac
    done

    # --- SGLang: token-based KV cache cap ---
    if [[ "$engine" == "sglang" && -n "${_PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS:-}" ]]; then
        echo "--max-total-tokens ${_PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS}"
        return 0
    fi

    # --- vLLM: byte-based KV cache cap ---
    # --gpu-memory-utilization 0.01 prevents vLLM's startup check from rejecting
    # the launch when co-resident tests use >10% of VRAM (vLLM checks free memory
    # against the fraction *before* applying the byte cap).
    if [[ "$engine" == "vllm" && -n "${_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES:-}" ]]; then
        local kv_bytes="$_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES"
        if [[ "$workers_per_gpu" -gt 1 ]]; then
            kv_bytes=$(awk -v b="$kv_bytes" -v n="$workers_per_gpu" 'BEGIN { printf "%d", b / n }')
        fi
        echo "--kv-cache-memory-bytes $kv_bytes --gpu-memory-utilization 0.01"
        return 0
    fi

    # No override — engine uses its default allocation
    echo ""
}


# ---------------------------------------------------------------------------
# Self-test: bash gpu_utils.sh --self-test
# ---------------------------------------------------------------------------
_gpu_utils_self_test() {
    local pass=0 fail=0
    _assert() {
        local label="$1" expected="$2" actual="$3"
        if [[ "$expected" == "$actual" ]]; then
            ((pass++))
            echo "  PASS  $label"
        else
            ((fail++))
            echo "  FAIL  $label  (expected='$expected'  actual='$actual')"
        fi
    }

    local result

    echo "=== vLLM: kv bytes override ==="
    result=$(_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES=942054000 \
        build_gpu_mem_args vllm)
    _assert "kv bytes" "--kv-cache-memory-bytes 942054000 --gpu-memory-utilization 0.01" "$result"

    echo ""
    echo "=== vLLM: kv bytes with --workers-per-gpu 2 ==="
    result=$(_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES=942054000 \
        build_gpu_mem_args vllm --workers-per-gpu 2)
    _assert "kv bytes / 2" "--kv-cache-memory-bytes 471027000 --gpu-memory-utilization 0.01" "$result"

    echo ""
    echo "=== vLLM: no override = empty ==="
    result=$(build_gpu_mem_args vllm)
    _assert "empty (engine default)" "" "$result"

    echo ""
    echo "=== vLLM: sglang token env ignored ==="
    result=$(_PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS=23824 \
        build_gpu_mem_args vllm)
    _assert "vllm ignores token cap" "" "$result"

    echo ""
    echo "=== sglang: token cap env ==="
    result=$(_PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS=1024 \
        build_gpu_mem_args sglang)
    _assert "token cap" "--max-total-tokens 1024" "$result"

    echo ""
    echo "=== sglang: no override = empty ==="
    result=$(build_gpu_mem_args sglang)
    _assert "empty (engine default)" "" "$result"

    echo ""
    echo "=== sglang: vllm kv bytes env ignored ==="
    result=$(_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES=942054000 \
        build_gpu_mem_args sglang)
    _assert "sglang ignores kv bytes" "" "$result"

    echo ""
    echo "=== missing engine ==="
    (build_gpu_mem_args 2>/dev/null)
    _assert "missing engine exits non-zero" "1" "$?"

    echo ""
    echo "=========================================="
    echo "Results: $pass passed, $fail failed"
    echo "=========================================="
    [[ "$fail" -eq 0 ]]
}

# Self-test: source this file then call _gpu_utils_self_test
if [[ "${BASH_SOURCE[0]}" == "$0" && "${1:-}" == "--self-test" ]]; then
    _gpu_utils_self_test
    exit $?
fi
