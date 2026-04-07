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
#       Supported engines: vllm, sglang
#
#       vLLM:   _PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES      → --kv-cache-memory-bytes N --gpu-memory-utilization 0.01
#       SGLang: _PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS → --max-total-tokens N
#
#       Note: TensorRT-LLM uses build_trtllm_override_args_with_mem() instead (requires JSON merging)
#
#       TODO: Split into build_vllm_gpu_mem_args and build_sglang_gpu_mem_args
#
# Usage:
#   # vLLM / SGLang
#   GPU_MEM_ARGS=$(build_gpu_mem_args sglang)
#   python -m dynamo.sglang --model-path "$MODEL" $GPU_MEM_ARGS &
#
#   GPU_MEM_ARGS=$(build_gpu_mem_args vllm)
#   python -m dynamo.vllm --model "$MODEL" $GPU_MEM_ARGS &
build_gpu_mem_args() {
    local engine="${1:?usage: build_gpu_mem_args <engine> [--workers-per-gpu N]}"
    shift

    # TensorRT-LLM uses build_trtllm_override_args_with_mem instead
    if [[ "$engine" == "trtllm" ]]; then
        echo "build_gpu_mem_args: TensorRT-LLM not supported. Use build_trtllm_override_args_with_mem instead." >&2
        return 1
    fi

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
# build_trtllm_override_args_with_mem [--merge-with-json JSON]
#   TensorRT-LLM-specific: builds JSON for --override-engine-args with GPU memory config.
#   Returns ONLY the bare JSON value (no --override-engine-args flag, no quotes).
#
#   Separate function because TRT-LLM requires JSON merging for --override-engine-args
#   (unlike vLLM/SGLang which use direct CLI flags).
#
#   Environment variables:
#     _PROFILE_OVERRIDE_TRTLLM_MAX_TOTAL_TOKENS        → {"kv_cache_config": {"max_tokens": N}}
#     _PROFILE_OVERRIDE_TRTLLM_MAX_GPU_TOTAL_BYTES → {"kv_cache_config": {"max_gpu_total_bytes": N}}
#
#   If --merge-with-json is provided, merges GPU config with the existing JSON.
#
# Usage:
#   # TensorRT-LLM: simple case (no existing overrides)
#   JSON=$(build_trtllm_override_args_with_mem)
#   python -m dynamo.trtllm --model-path "$MODEL" ${JSON:+--override-engine-args "$JSON"} &
#
#   # TensorRT-LLM: merge with existing JSON
#   EXISTING='{"return_perf_metrics": true}'
#   JSON=$(build_trtllm_override_args_with_mem --merge-with-json "$EXISTING")
#   python -m dynamo.trtllm --model-path "$MODEL" --override-engine-args "$JSON" &
# ---------------------------------------------------------------------------
build_trtllm_override_args_with_mem() {
    local merge_json=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --merge-with-json)
                merge_json="$2"
                shift 2
                ;;
            *) echo "build_trtllm_override_args_with_mem: unknown option '$1'" >&2; return 1 ;;
        esac
    done

    local gpu_mem_json=""

    # Token-based (preferred, simpler to reason about)
    if [[ -n "${_PROFILE_OVERRIDE_TRTLLM_MAX_TOTAL_TOKENS:-}" ]]; then
        gpu_mem_json='"kv_cache_config": {"max_tokens": '"${_PROFILE_OVERRIDE_TRTLLM_MAX_TOTAL_TOKENS}"'}'
    # Byte-based (alternative, more precise)
    elif [[ -n "${_PROFILE_OVERRIDE_TRTLLM_MAX_GPU_TOTAL_BYTES:-}" ]]; then
        gpu_mem_json='"kv_cache_config": {"max_gpu_total_bytes": '"${_PROFILE_OVERRIDE_TRTLLM_MAX_GPU_TOTAL_BYTES}"'}'
    fi

    if [[ -n "$gpu_mem_json" ]]; then
        if [[ -n "$merge_json" ]]; then
            # Merge: GPU mem config first, then existing config
            # Strip outer braces from existing JSON
            local existing="${merge_json#\{}"
            existing="${existing%\}}"
            if [[ -n "${existing//[[:space:]]/}" ]]; then
                echo "{${gpu_mem_json}, ${existing}}"
            else
                echo "{${gpu_mem_json}}"
            fi
        else
            # Just GPU mem config
            echo "{${gpu_mem_json}}"
        fi
    elif [[ -n "$merge_json" ]]; then
        # No GPU override, return existing JSON as-is
        echo "$merge_json"
    fi

    # No output if both are empty (engine uses default)
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
    echo "=== trtllm: token cap env ==="
    result=$(_PROFILE_OVERRIDE_TRTLLM_MAX_TOTAL_TOKENS=4096 \
        build_trtllm_override_args_with_mem)
    _assert "trtllm token cap" '{"kv_cache_config": {"max_tokens": 4096}}' "$result"

    echo ""
    echo "=== trtllm: byte cap env ==="
    result=$(_PROFILE_OVERRIDE_TRTLLM_MAX_GPU_TOTAL_BYTES=1073741824 \
        build_trtllm_override_args_with_mem)
    _assert "trtllm byte cap" '{"kv_cache_config": {"max_gpu_total_bytes": 1073741824}}' "$result"

    echo ""
    echo "=== trtllm: no override = empty ==="
    result=$(build_trtllm_override_args_with_mem)
    _assert "empty (engine default)" "" "$result"

    echo ""
    echo "=== trtllm: token cap takes precedence over byte cap ==="
    result=$(_PROFILE_OVERRIDE_TRTLLM_MAX_TOTAL_TOKENS=2048 _PROFILE_OVERRIDE_TRTLLM_MAX_GPU_TOTAL_BYTES=999999 \
        build_trtllm_override_args_with_mem)
    _assert "trtllm token precedence" '{"kv_cache_config": {"max_tokens": 2048}}' "$result"

    echo ""
    echo "=== trtllm: merge with existing JSON ==="
    result=$(_PROFILE_OVERRIDE_TRTLLM_MAX_TOTAL_TOKENS=2048 \
        build_trtllm_override_args_with_mem --merge-with-json '{"return_perf_metrics": true, "otlp_traces_endpoint": "http://localhost:4317"}')
    _assert "trtllm merged" '{"kv_cache_config": {"max_tokens": 2048}, "return_perf_metrics": true, "otlp_traces_endpoint": "http://localhost:4317"}' "$result"

    echo ""
    echo "=== trtllm: merge with empty JSON object ==="
    result=$(_PROFILE_OVERRIDE_TRTLLM_MAX_TOTAL_TOKENS=2048 \
        build_trtllm_override_args_with_mem --merge-with-json '{}')
    _assert "trtllm merge empty obj" '{"kv_cache_config": {"max_tokens": 2048}}' "$result"

    echo ""
    echo "=== trtllm: no GPU override, but pass through existing JSON ==="
    result=$(build_trtllm_override_args_with_mem --merge-with-json '{"return_perf_metrics": true}')
    _assert "trtllm passthrough" '{"return_perf_metrics": true}' "$result"

    echo ""
    echo "=== missing engine ==="
    (build_gpu_mem_args 2>/dev/null)
    _assert "missing engine exits non-zero" "1" "$?"

    echo ""
    echo "=== trtllm rejected (use build_trtllm_override_args_with_mem) ==="
    (build_gpu_mem_args trtllm 2>/dev/null)
    _assert "trtllm rejected" "1" "$?"

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
