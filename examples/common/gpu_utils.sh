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
#
#   build_vllm_gpu_mem_args
#       vLLM:   _PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES → --kv-cache-memory-bytes N --gpu-memory-utilization 0.01
#
#   build_sglang_gpu_mem_args
#       SGLang: _PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS → --max-total-tokens N
#
#       Note: TensorRT-LLM uses build_trtllm_override_args_with_mem() instead (requires JSON merging)
#
# Usage:
#   GPU_MEM_ARGS=$(build_sglang_gpu_mem_args)
#   python -m dynamo.sglang --model-path "$MODEL" $GPU_MEM_ARGS &
#
#   GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)
#   python -m dynamo.vllm --model "$MODEL" $GPU_MEM_ARGS &


# ---------------------------------------------------------------------------
# build_vllm_gpu_mem_args
#   Returns vLLM CLI args for GPU memory control.
#   Empty if _PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES is not set.
#
#   --kv-cache-memory-bytes is per-process: each vLLM worker gets the same
#   value, even in multi-worker-per-GPU setups (e.g. disagg_same_gpu.sh).
#   The profiler finds the per-worker budget directly.
#
#   --gpu-memory-utilization 0.01 prevents vLLM's startup check from rejecting
#   the launch when co-resident tests use >10% of VRAM (vLLM checks free memory
#   against the fraction *before* applying the byte cap).
# ---------------------------------------------------------------------------
build_vllm_gpu_mem_args() {
    if [[ -n "${_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES:-}" ]]; then
        echo "--kv-cache-memory-bytes ${_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES} --gpu-memory-utilization 0.01"
        return 0
    fi

    echo ""
}


# ---------------------------------------------------------------------------
# build_sglang_gpu_mem_args
#   Returns SGLang CLI args for GPU memory control.
#   Empty if _PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS is not set.
# ---------------------------------------------------------------------------
build_sglang_gpu_mem_args() {
    if [[ -n "${_PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS:-}" ]]; then
        # --mem-fraction-static 0.9: SGLang sizes the KV cache pool using this
        # fraction of pre-model-load memory BEFORE applying --max-total-tokens.
        # The default (~0.65) can request more pool memory than is available on
        # smaller GPUs (e.g. L4 24 GiB) even when the token cap would fit.
        # Setting 0.9 maximises the pool headroom; the token cap controls the
        # actual allocation.
        echo "--max-total-tokens ${_PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS} --mem-fraction-static 0.9"
        return 0
    fi

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

# get_model_params <model_name>
#
# Prints "params_b weight_bytes layers kv_heads head_dim" to stdout.
# Returns 1 (prints nothing) if the model is unknown.
#
# Fields:
#   params_b       Total parameters in billions (all experts for MoE)
#   weight_bytes   Bytes per weight element (2=BF16/FP16, 1=FP8)
#   layers         Number of transformer layers
#   kv_heads       Number of key-value heads (GQA groups)
#   head_dim       Dimension per attention head
#
# KV cache is assumed BF16 (2 bytes per element) regardless of weight dtype,
# since FP8 KV cache (--kv-cache-dtype fp8) is opt-in and not the default.
#
# To add a model:
#   1. Find config.json at  https://huggingface.co/<model>/raw/main/config.json
#      For VL/multimodal models, architecture params are under text_config.
#   2. Map fields:
#        layers    ← num_hidden_layers
#        kv_heads  ← num_key_value_heads
#        head_dim  ← head_dim  (or hidden_size / num_attention_heads)
#   3. params_b: total parameter count in billions.  Derive from:
#        - safetensors file size:  size_bytes / weight_bytes / 1e9
#          (single file: ls -l model.safetensors; sharded: metadata.total_size
#          in model.safetensors.index.json)
#        - or the model card / paper
#      For MoE: params_b is the TOTAL count (all experts loaded into VRAM).
#   4. weight_bytes: 2 for BF16/FP16, 1 for FP8/INT8.
#
# Usage:
#   read -r pb wb layers kvh hd <<< "$(get_model_params "Qwen/Qwen3-0.6B")"
#   echo "$layers layers, $kvh KV heads"
get_model_params() {
    local model="${1:?usage: get_model_params <model_name>}"
    local pb wb layers kvh hd
    case "$model" in
        # https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json
        Qwen/Qwen3-0.6B)
            pb=0.6;  wb=2; layers=28; kvh=8;  hd=128 ;;
        # https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/raw/main/config.json  (text_config)
        # params_b from model.safetensors.index.json metadata.total_size / 2 / 1e9
        Qwen/Qwen2-VL-2B-Instruct)
            pb=2.2;  wb=2; layers=28; kvh=2;  hd=128 ;;
        # https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/raw/main/config.json  (text_config)
        Qwen/Qwen2.5-VL-7B-Instruct)
            pb=8.3;  wb=2; layers=28; kvh=4;  hd=128 ;;
        # https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/raw/main/config.json  (text_config)
        # params_b from model.safetensors size / 2 / 1e9
        Qwen/Qwen3-VL-2B-Instruct)
            pb=2.1;  wb=2; layers=28; kvh=8;  hd=128 ;;
        # https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/raw/main/config.json  (text_config)
        Qwen/Qwen3-VL-8B-Instruct)
            pb=9.2;  wb=2; layers=36; kvh=8;  hd=128 ;;
        # https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json
        Qwen/Qwen3-30B-A3B|\
        Qwen/Qwen3-30B-A3B-Instruct)
            pb=30.5; wb=2; layers=48; kvh=4;  hd=128 ;;
        # Same architecture as Qwen3-30B-A3B but FP8 quantized (1 byte per weight)
        Qwen/Qwen3-VL-30B-A3B-Instruct-FP8)
            pb=30.5; wb=1; layers=48; kvh=4;  hd=128 ;;
        # https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/raw/main/config.json
        meta-llama/Meta-Llama-3.1-8B-Instruct)
            pb=8.0;  wb=2; layers=32; kvh=8;  hd=128 ;;
        # https://huggingface.co/deepseek-ai/deepseek-llm-7b-base/raw/main/config.json
        # MHA (not GQA): num_key_value_heads == num_attention_heads == 32
        deepseek-ai/deepseek-llm-7b-base)
            pb=6.9;  wb=2; layers=30; kvh=32; hd=128 ;;
        # https://huggingface.co/Qwen/Qwen3-Embedding-4B/raw/main/config.json
        # params_b from model.safetensors.index.json metadata.total_size / 2 / 1e9
        # head_dim = hidden_size(2560) / num_attention_heads(32) = 80
        Qwen/Qwen3-Embedding-4B)
            pb=4.0;  wb=2; layers=36; kvh=8;  hd=80 ;;
        # https://huggingface.co/llava-hf/llava-1.5-7b-hf/raw/main/config.json  (text_config)
        # MHA: num_key_value_heads == num_attention_heads == 32
        llava-hf/llava-1.5-7b-hf)
            pb=7.1;  wb=2; layers=32; kvh=32; hd=128 ;;
        *)
            echo "get_model_params: unknown model '$model'" >&2
            echo "Add it to get_model_params() in gpu_utils.sh" >&2
            return 1 ;;
    esac
    echo "$pb $wb $layers $kvh $hd"
}

# estimate_worker_vram <model> [max_model_len] [max_concurrent_seqs] [engine_or_overhead]
#
# Prints "weights_gib kv_gib overhead_gib total_gib" to stdout.
# Returns 1 (prints nothing) if the model is unknown to get_model_params.
#
# Formula:
#   weights = params_b * 1e9 * weight_bytes
#   kv      = 2 * layers * kv_heads * head_dim * 2(BF16) * seq_len * seqs
#   total   = weights + kv + overhead
#
# Arguments:
#   model               HuggingFace model name (required)
#   max_model_len       Max tokens per sequence (default: 4096)
#   max_concurrent_seqs Concurrent sequences to budget for (default: 2)
#   engine_or_overhead  Engine name OR explicit GiB value (default: 2.0)
#
# If the 4th argument is an engine name (vllm, sglang, trtllm), overhead is
# auto-computed from model parameters:
#   overhead = base + scale * sqrt(params_b)
#
# Per-engine constants (calibrated from measurements on RTX 6000 Ada 48 GiB):
#   vllm:   base=1.2, scale=1.0  → 0.6B≈2.0, 8B≈4.0, 30B≈6.7
#   sglang: base=1.5, scale=1.0  → 0.6B≈2.3, 8B≈4.3, 30B≈7.0
#   trtllm: base=2.0, scale=1.2  → 0.6B≈2.9, 8B≈5.4, 30B≈8.6
#
# sglang overhead was re-calibrated via profile_pytest.py bisection on
# RTX 6000 Ada 48 GiB. Observed CUDA overhead (outside --mem-fraction-static):
#   Qwen3-0.6B: ~1.8 GiB. Previous coefficients (2.5, 1.5) over-estimated by ~2x.
#
# If the 4th argument is a number, it's used directly (backward compatible).
# If omitted, defaults to 2.0 (backward compatible).
#
# See examples/common/gpu_utils.md for the full derivation.
#
# Usage:
#   read -r w kv oh total <<< "$(estimate_worker_vram "Qwen/Qwen3-0.6B" 4096 2 vllm)"
#   echo "$total GiB (w=$w kv=$kv oh=$oh)"
estimate_worker_vram() {
    local model="${1:?usage: estimate_worker_vram <model> [seq_len] [seqs] [engine_or_overhead]}"
    local seqlen="${2:-4096}"
    local seqs="${3:-2}"
    local engine_or_overhead="${4:-2.0}"

    local mp_out
    mp_out=$(get_model_params "$model") || return 1
    local pb wb layers kvh hd
    read -r pb wb layers kvh hd <<< "$mp_out"

    local overhead
    case "$engine_or_overhead" in
        vllm)   overhead=$(awk -v p="$pb" 'BEGIN { printf "%.1f", 1.2 + 1.0 * sqrt(p) }') ;;
        sglang) overhead=$(awk -v p="$pb" 'BEGIN { printf "%.1f", 1.5 + 1.0 * sqrt(p) }') ;;
        trtllm) overhead=$(awk -v p="$pb" 'BEGIN { printf "%.1f", 2.0 + 1.2 * sqrt(p) }') ;;
        *)      overhead="$engine_or_overhead" ;;
    esac

    awk -v pb="$pb" -v wbytes="$wb" \
        -v layers="$layers" -v heads="$kvh" -v dim="$hd" \
        -v seqlen="$seqlen" -v seqs="$seqs" -v overhead="$overhead" \
        'BEGIN {
            gib = 1024 * 1024 * 1024
            w   = pb * 1e9 * wbytes / gib
            kv  = 2 * layers * heads * dim * 2 * seqlen * seqs / gib
            printf "%.1f %.1f %.1f %.1f", w, kv, overhead, w + kv + overhead
        }'
}

# Query GPU memory from the vendor tool available in the current environment.
#
# Prints memory in MiB to stdout.
# Returns 1 if:
#   - the metric is invalid,
#   - the selected GPU cannot be queried,
#   - or neither nvidia-smi nor xpu-smi is installed.
#
# Backend selection:
#   nvidia-smi  Preferred when available. Queries CSV output directly.
#   xpu-smi     Used as a fallback for Intel XPU environments. Queries JSON
#               output and extracts the relevant byte field, then converts it
#               to MiB.
#
# Supported metrics:
#   total   Physical device memory size.
#   free    Currently free device memory.
#
# Notes:
#   - The function prefers nvidia-smi when both tools are present.
#   - xpu-smi currently uses `discovery -j`, so the JSON field names are part
#     of the contract here: total -> memory_physical_size_byte,
#     free -> memory_free_size_byte.
#   - The output is intended for downstream fraction calculations in
#     gpu_gb_to_total_fraction and gpu_gb_to_free_fraction.
#
# Usage:
#   _gpu_query_memory_mib total        # total MiB on GPU 0
#   _gpu_query_memory_mib free 1       # free MiB on GPU 1
_gpu_query_memory_mib() {
    local metric="${1:?usage: _gpu_query_memory_mib <total|free> [gpu_index]}"
    local gpu_idx="${2:-0}"

    if command -v nvidia-smi >/dev/null 2>&1; then
        case "$metric" in
            total)
                nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$gpu_idx" 2>/dev/null
                return
                ;;
            free)
                nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_idx" 2>/dev/null
                return
                ;;
            *)
                echo "_gpu_query_memory_mib: unknown metric '$metric'" >&2
                return 1
                ;;
        esac
    fi

    if command -v xpu-smi >/dev/null 2>&1; then
        local json_key
        case "$metric" in
            total) json_key="memory_physical_size_byte" ;;
            free)  json_key="memory_free_size_byte" ;;
            *)
                echo "_gpu_query_memory_mib: unknown metric '$metric'" >&2
                return 1
                ;;
        esac

        xpu-smi discovery -d "$gpu_idx" -j 2>/dev/null | awk -F: -v key="$json_key" '
            $1 ~ "\"" key "\"" {
                gsub(/[",[:space:]]/, "", $2)
                printf "%d\n", int($2 / (1024 * 1024))
                found = 1
                exit
            }
            END { if (!found) exit 1 }
        '
        return
    fi

    echo "_gpu_query_memory_mib: neither nvidia-smi nor xpu-smi is available" >&2
    return 1
}

# gpu_worker_fraction <engine> <total_gib> <kv_gib> [gpu_index]
#
# Convert estimated GiB into the engine-appropriate GPU memory fraction.
#
# Engine semantics (see examples/common/gpu_utils.md):
#   vllm/sglang  — fraction of TOTAL VRAM (uses total_gib).
#   trtllm       — fraction of FREE VRAM after model load (uses kv_gib).
#
# Usage:
#   gpu_worker_fraction vllm   4.0 0.9      # fraction of total
#   gpu_worker_fraction trtllm 4.0 0.9      # fraction of free
#   gpu_worker_fraction trtllm 4.0 0.9 1    # query GPU index 1
gpu_worker_fraction() {
    local engine="${1:?usage: gpu_worker_fraction <engine> <total_gib> <kv_gib> [gpu_index]}"
    local total_gib="${2:?usage: gpu_worker_fraction <engine> <total_gib> <kv_gib>}"
    local kv_gib="${3:?usage: gpu_worker_fraction <engine> <total_gib> <kv_gib>}"
    local gpu_idx="${4:-0}"
    case "$engine" in
        vllm|sglang)
            gpu_gb_to_total_fraction "$total_gib" "$gpu_idx" ;;
        trtllm)
            gpu_gb_to_free_fraction "$kv_gib" "$gpu_idx" ;;
        *)
            echo "gpu_worker_fraction: unknown engine '$engine'" >&2
            echo "Supported: vllm, sglang, trtllm" >&2
            return 1 ;;
    esac
}

# gpu_peak_to_engine_fraction <engine> <peak_gib> [gpu_index]
#
# Convert a measured/profiled GPU peak (total VRAM including CUDA context,
# activations, etc.) into the engine-specific memory fraction flag.
#
# Each engine's fraction controls only a SUBSET of GPU memory (e.g. vLLM's
# --gpu-memory-utilization covers weights + KV cache but not CUDA context).
# This function subtracts the engine-specific overhead so the fraction
# targets the right internal budget, keeping the real peak stable across
# re-profiles.
#
# Overhead constants (GiB outside the engine's budget):
#   vllm   2.0   CUDA ctx ~0.6 + activations/sampler ~0.5 + PyTorch alloc ~0.5
#   sglang 2.0   (assumed same as vllm; refine when profiled)
#   trtllm 0.0   free-fraction is measured after model load, no subtraction needed
#
# Usage:
#   gpu_peak_to_engine_fraction vllm 8.6       # on 48 GiB → 0.14
#   gpu_peak_to_engine_fraction vllm 20.9      # on 48 GiB → 0.40
#   gpu_peak_to_engine_fraction vllm 8.6 1     # query GPU index 1
gpu_peak_to_engine_fraction() {
    local engine=${1:?usage: gpu_peak_to_engine_fraction <engine> <peak_gib> [gpu_index]}
    local peak_gib=${2:?usage: gpu_peak_to_engine_fraction <engine> <peak_gib> [gpu_index]}
    local gpu_idx=${3:-0}

    local overhead
    case "$engine" in
        vllm|sglang) overhead=2.0 ;;
        trtllm)      overhead=0.0 ;;
        *)
            echo "gpu_peak_to_engine_fraction: unknown engine '$engine'" >&2
            echo "Supported: vllm, sglang, trtllm" >&2
            return 1 ;;
    esac

    local budget
    budget=$(awk -v g="$peak_gib" -v oh="$overhead" \
        'BEGIN { b = g - oh; if (b < 1) b = 1; printf "%.1f", b }')

    case "$engine" in
        vllm|sglang) gpu_gb_to_total_fraction "$budget" "$gpu_idx" ;;
        trtllm)      gpu_gb_to_free_fraction  "$budget" "$gpu_idx" ;;
    esac
}

# gpu_gb_to_total_fraction <gib> [gpu_index]
#
# For vLLM / sglang: --gpu-memory-utilization is a fraction of TOTAL GPU memory.
# The engine budgets model weights + KV cache + activations within that limit.
#
# Prints the fraction of total GPU VRAM that <gib> GiB represents.
# Useful for converting portable absolute memory requirements to
# engine-specific fraction parameters (--gpu-memory-utilization, etc).
#
# Examples:
#   gpu_gb_to_total_fraction 4        # on 48 GiB GPU → 0.09
#   gpu_gb_to_total_fraction 16       # on 48 GiB GPU → 0.34
#   gpu_gb_to_total_fraction 4 1      # query GPU index 1 instead of 0
#
# The result is ceil-rounded to 2 decimal places with a minimum of 0.05
# and a maximum of 0.95.
gpu_gb_to_total_fraction() {
    local gib=${1:?usage: gpu_gb_to_total_fraction <gib> [gpu_index]}
    local gpu_idx=${2:-0}

    local total_mib
    total_mib=$(_gpu_query_memory_mib total "$gpu_idx")
    if [[ -z "$total_mib" || "$total_mib" -eq 0 ]]; then
        echo "gpu_gb_to_total_fraction: failed to query GPU $gpu_idx total memory" >&2
        return 1
    fi

    local total_gib
    total_gib=$(awk -v t="$total_mib" 'BEGIN { printf "%.1f", t / 1024 }')

    if awk -v gib="$gib" -v total="$total_mib" 'BEGIN { exit (gib * 1024 > total) ? 0 : 1 }'; then
        echo "" >&2
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
        echo "WARNING: Requested ${gib} GiB but GPU $gpu_idx only has ${total_gib} GiB total." >&2
        echo "The model likely won't fit. Consider a GPU with more VRAM" >&2
        echo "or reduce the model size (quantization, smaller model, etc)." >&2
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
        echo "" >&2
    fi

    # fraction = gib * 1024 / total_mib, ceil to 2 decimals, clamp [0.05, 0.95]
    awk -v gib="$gib" -v total="$total_mib" 'BEGIN {
        frac = (gib * 1024) / total
        # ceil to 2 decimal places
        frac = int(frac * 100 + 0.99) / 100
        if (frac < 0.05) frac = 0.05
        if (frac > 0.95) frac = 0.95
        printf "%.2f\n", frac
    }'
}

# gpu_gb_to_free_fraction <gib> [gpu_index]
#
# For TensorRT-LLM: --free-gpu-memory-fraction (CLI) and
# kv_cache_config.free_gpu_memory_fraction (YAML) are fractions of FREE
# memory AFTER model weights are loaded — NOT fractions of total VRAM.
# The engine loads model weights first, queries remaining free memory,
# then allocates  fraction * free_after_model  for the KV cache.
#
# Why gpu_gb_to_total_fraction won't work for TensorRT-LLM:
#   gpu_gb_to_total_fraction(10) on a 48 GiB GPU → 0.21 (fraction of total).
#   Passing 0.21 as free_gpu_memory_fraction after a 5 GiB model loads
#   would allocate 0.21 * 43 GiB ≈ 9 GiB — close but not exact.
#   For larger models the error grows: a 30 GiB model leaves 18 GiB free,
#   so 0.21 * 18 ≈ 3.8 GiB — far less than the 10 GiB intended.
#
# This function queries CURRENT free memory from the available GPU SMI
# backend (nvidia-smi or xpu-smi) and computes gib / free_mib. The result
# is a best-effort estimate: TensorRT-LLM will see less free memory than
# we measure here (model weights haven't loaded yet), so the actual KV
# cache allocation will be smaller than <gib>.
# For rough sizing this is fine; for precise control use the YAML config
# with a known model size.
#
# For disagg_same_gpu (two workers sharing one GPU), launch workers
# sequentially: start the first, wait for it to finish loading (poll
# nvidia-smi, xpu-smi, or logs), then query free memory again and compute
# the fraction for the second worker. This gives predictable per-worker
# KV cache sizes on any GPU.
#
# Override at launch via CLI or env var:
#   --override-engine-args '{"kv_cache_config":{"free_gpu_memory_fraction": 0.15}}'
#   DYN_TRTLLM_OVERRIDE_ENGINE_ARGS='{"kv_cache_config":{"free_gpu_memory_fraction": 0.15}}'
#
# GOTCHA: overriding any field inside kv_cache_config REPLACES the entire
# sub-dict from the YAML. You must re-include all fields you care about
# (e.g. enable_block_reuse, dtype) or they'll be lost.
#
# Examples:
#   gpu_gb_to_free_fraction 10       # on 48 GiB GPU with 46 GiB free → 0.22
#   gpu_gb_to_free_fraction 10 1     # query GPU index 1 instead of 0
#
# The result is ceil-rounded to 2 decimal places, clamped [0.01, 0.95].
# The floor is 0.01 (not 0.05 like gpu_gb_to_total_fraction) because this
# fraction only controls KV cache, so small values are valid.
gpu_gb_to_free_fraction() {
    local gib=${1:?usage: gpu_gb_to_free_fraction <gib> [gpu_index]}
    local gpu_idx=${2:-0}

    local free_mib
    free_mib=$(_gpu_query_memory_mib free "$gpu_idx")
    if [[ -z "$free_mib" || "$free_mib" -eq 0 ]]; then
        echo "gpu_gb_to_free_fraction: failed to query GPU $gpu_idx free memory" >&2
        return 1
    fi

    local free_gib
    free_gib=$(awk -v f="$free_mib" 'BEGIN { printf "%.1f", f / 1024 }')

    if awk -v gib="$gib" -v free="$free_mib" 'BEGIN { exit (gib * 1024 > free) ? 0 : 1 }'; then
        echo "" >&2
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
        echo "WARNING: Requested ${gib} GiB KV cache but GPU $gpu_idx only has ${free_gib} GiB free." >&2
        echo "After model loading, even less will be available." >&2
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
        echo "" >&2
    fi

    # fraction = gib * 1024 / free_mib, ceil to 2 decimals, clamp [0.01, 0.95]
    awk -v gib="$gib" -v free="$free_mib" 'BEGIN {
        frac = (gib * 1024) / free
        frac = int(frac * 100 + 0.99) / 100
        if (frac < 0.01) frac = 0.01
        if (frac > 0.95) frac = 0.95
        printf "%.2f\n", frac
    }'
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

    # --- build_vllm_gpu_mem_args (direct) ---

    echo "=== vLLM: kv bytes override ==="
    result=$(_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES=942054000 \
        build_vllm_gpu_mem_args)
    _assert "kv bytes" "--kv-cache-memory-bytes 942054000 --gpu-memory-utilization 0.01" "$result"

    echo ""
    echo "=== vLLM: no override = empty ==="
    result=$(build_vllm_gpu_mem_args)
    _assert "empty (engine default)" "" "$result"

    echo ""
    echo "=== vLLM: sglang token env ignored ==="
    result=$(_PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS=23824 \
        build_vllm_gpu_mem_args)
    _assert "vllm ignores token cap" "" "$result"

    # --- build_sglang_gpu_mem_args (direct) ---

    echo ""
    echo "=== sglang: token cap env ==="
    result=$(_PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS=1024 \
        build_sglang_gpu_mem_args)
    _assert "token cap" "--max-total-tokens 1024 --mem-fraction-static 0.9" "$result"

    echo ""
    echo "=== sglang: no override = empty ==="
    result=$(build_sglang_gpu_mem_args)
    _assert "empty (engine default)" "" "$result"

    echo ""
    echo "=== sglang: vllm kv bytes env ignored ==="
    result=$(_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES=942054000 \
        build_sglang_gpu_mem_args)
    _assert "sglang ignores kv bytes" "" "$result"


    # --- build_trtllm_override_args_with_mem ---

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
