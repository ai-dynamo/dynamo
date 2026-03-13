#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shared GPU utility functions for launch scripts.
#
# Usage:
#   source "$(dirname "$(readlink -f "$0")")/../common/gpu_utils.sh"
#   # or with SCRIPT_DIR already set:
#   source "$SCRIPT_DIR/../common/gpu_utils.sh"
#
# Functions:
#   get_model_params <model>           Set _MP_* vars for a known model's architecture
#   estimate_worker_vram <model> ...   Set _EW_* vars with per-worker VRAM estimate
#   gpu_worker_fraction <engine>       Convert _EW_* estimate → engine-appropriate fraction
#   gpu_gb_to_total_fraction <gib>     Convert absolute GiB → fraction of TOTAL VRAM (vLLM/sglang)
#   gpu_gb_to_free_fraction <gib>      Convert absolute GiB → fraction of FREE VRAM (TensorRT-LLM)

# get_model_params <model_name>
#
# Sets _MP_* variables for a known model's architecture:
#   _MP_PARAMS_B       Total parameters in billions (all experts for MoE)
#   _MP_WEIGHT_BYTES   Bytes per weight element (2=BF16/FP16, 1=FP8)
#   _MP_LAYERS         Number of transformer layers
#   _MP_KV_HEADS       Number of key-value heads (GQA groups)
#   _MP_HEAD_DIM       Dimension per attention head
#
# KV cache is assumed BF16 (2 bytes per element) regardless of weight dtype,
# since FP8 KV cache (--kv-cache-dtype fp8) is opt-in and not the default.
#
# To add a model: look up config.json on HuggingFace for num_hidden_layers,
# num_key_value_heads, and head_dim. For VL/multimodal models, use the
# text_config section. For MoE, _MP_PARAMS_B is the TOTAL param count
# (all experts are loaded into VRAM).
#
# Usage:
#   get_model_params "Qwen/Qwen3-0.6B"
#   echo "$_MP_LAYERS layers, $_MP_KV_HEADS KV heads"
get_model_params() {
    local model="${1:?usage: get_model_params <model_name>}"
    case "$model" in
        Qwen/Qwen3-0.6B)
            _MP_PARAMS_B=0.6;  _MP_WEIGHT_BYTES=2
            _MP_LAYERS=28;  _MP_KV_HEADS=8;   _MP_HEAD_DIM=128 ;;
        Qwen/Qwen2.5-VL-7B-Instruct)
            _MP_PARAMS_B=8.3;  _MP_WEIGHT_BYTES=2
            _MP_LAYERS=28;  _MP_KV_HEADS=4;   _MP_HEAD_DIM=128 ;;
        Qwen/Qwen3-VL-8B-Instruct)
            _MP_PARAMS_B=9.2;  _MP_WEIGHT_BYTES=2
            _MP_LAYERS=36;  _MP_KV_HEADS=8;   _MP_HEAD_DIM=128 ;;
        Qwen/Qwen3-30B-A3B|\
        Qwen/Qwen3-30B-A3B-Instruct)
            _MP_PARAMS_B=30.5; _MP_WEIGHT_BYTES=2
            _MP_LAYERS=48;  _MP_KV_HEADS=4;   _MP_HEAD_DIM=128 ;;
        Qwen/Qwen3-VL-30B-A3B-Instruct-FP8)
            _MP_PARAMS_B=30.5; _MP_WEIGHT_BYTES=1
            _MP_LAYERS=48;  _MP_KV_HEADS=4;   _MP_HEAD_DIM=128 ;;
        meta-llama/Meta-Llama-3.1-8B-Instruct)
            _MP_PARAMS_B=8.0;  _MP_WEIGHT_BYTES=2
            _MP_LAYERS=32;  _MP_KV_HEADS=8;   _MP_HEAD_DIM=128 ;;
        llava-hf/llava-1.5-7b-hf)
            _MP_PARAMS_B=7.1;  _MP_WEIGHT_BYTES=2
            _MP_LAYERS=32;  _MP_KV_HEADS=32;  _MP_HEAD_DIM=128 ;;
        *)
            echo "get_model_params: unknown model '$model'" >&2
            echo "Add it to get_model_params() in gpu_utils.sh" >&2
            return 1 ;;
    esac
}

# estimate_worker_vram <model> [max_model_len] [max_concurrent_seqs] [engine_or_overhead]
#
# Calls get_model_params, then sets:
#   _EW_WEIGHTS_GIB    Estimated model weight memory
#   _EW_KV_GIB         Estimated KV cache memory
#   _EW_OVERHEAD_GIB   Overhead used (auto-computed or explicit)
#   _EW_TOTAL_GIB      Estimated total per-worker VRAM (weights + kv + overhead)
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
#   sglang: base=2.5, scale=1.5  → 0.6B≈3.7, 8B≈6.7, 30B≈10.8
#   trtllm: base=2.0, scale=1.2  → 0.6B≈2.9, 8B≈5.4, 30B≈8.6
#
# If the 4th argument is a number, it's used directly (backward compatible).
# If omitted, defaults to 2.0 (backward compatible).
#
# See examples/common/gpu_utils.md for the full derivation.
#
# Usage:
#   estimate_worker_vram "Qwen/Qwen3-0.6B" 4096 2 vllm      # auto overhead
#   estimate_worker_vram "Qwen/Qwen3-0.6B" 4096 2 trtllm     # auto overhead
#   estimate_worker_vram "Qwen/Qwen3-0.6B" 4096 2 3.5        # explicit 3.5 GiB
#   estimate_worker_vram "Qwen/Qwen3-0.6B" 4096 2            # default 2.0 GiB
#   echo "$_EW_TOTAL_GIB GiB (w=$_EW_WEIGHTS_GIB kv=$_EW_KV_GIB oh=$_EW_OVERHEAD_GIB)"
estimate_worker_vram() {
    local model="${1:?usage: estimate_worker_vram <model> [seq_len] [seqs] [engine_or_overhead]}"
    local seqlen="${2:-4096}"
    local seqs="${3:-2}"
    local engine_or_overhead="${4:-2.0}"

    get_model_params "$model" || return 1

    local overhead
    case "$engine_or_overhead" in
        vllm)   overhead=$(awk -v p="$_MP_PARAMS_B" 'BEGIN { printf "%.1f", 1.2 + 1.0 * sqrt(p) }') ;;
        sglang) overhead=$(awk -v p="$_MP_PARAMS_B" 'BEGIN { printf "%.1f", 2.5 + 1.5 * sqrt(p) }') ;;
        trtllm) overhead=$(awk -v p="$_MP_PARAMS_B" 'BEGIN { printf "%.1f", 2.0 + 1.2 * sqrt(p) }') ;;
        *)      overhead="$engine_or_overhead" ;;
    esac

    _EW_OVERHEAD_GIB="$overhead"
    read -r _EW_WEIGHTS_GIB _EW_KV_GIB _EW_TOTAL_GIB <<< "$(awk \
        -v pb="$_MP_PARAMS_B" -v wbytes="$_MP_WEIGHT_BYTES" \
        -v layers="$_MP_LAYERS" -v heads="$_MP_KV_HEADS" -v dim="$_MP_HEAD_DIM" \
        -v seqlen="$seqlen" -v seqs="$seqs" -v overhead="$overhead" \
        'BEGIN {
            gib = 1024 * 1024 * 1024
            w   = pb * 1e9 * wbytes / gib
            kv  = 2 * layers * heads * dim * 2 * seqlen * seqs / gib
            printf "%.1f %.1f %.1f", w, kv, w + kv + overhead
        }')"
}

# gpu_worker_fraction <engine> [gpu_index]
#
# Unified fraction calculator for all engines.  Reads the _EW_* variables
# set by estimate_worker_vram and returns the engine-appropriate fraction.
#
# Engine semantics (see examples/common/gpu_utils.md):
#   vllm/sglang  — fraction of TOTAL VRAM.  The engine budgets weights + KV +
#                  activations inside this limit.  We pass _EW_TOTAL_GIB.
#   trtllm       — fraction of FREE VRAM (after model load).  The engine uses
#                  this only for KV cache.  We pass _EW_KV_GIB.
#
# This lets every launch script use the same pattern:
#   estimate_worker_vram "$MODEL" "$SEQ_LEN" "$CONCURRENCY" "$OVERHEAD_GIB"
#   GPU_MEM_FRACTION=$(gpu_worker_fraction "<engine>")
#
# Usage:
#   gpu_worker_fraction vllm        # uses _EW_TOTAL_GIB, fraction of total
#   gpu_worker_fraction sglang      # same as vllm
#   gpu_worker_fraction trtllm      # uses _EW_KV_GIB, fraction of free
#   gpu_worker_fraction trtllm 1    # query GPU index 1
gpu_worker_fraction() {
    local engine="${1:?usage: gpu_worker_fraction <engine> [gpu_index]}"
    local gpu_idx="${2:-0}"
    case "$engine" in
        vllm|sglang)
            gpu_gb_to_total_fraction "$_EW_TOTAL_GIB" "$gpu_idx" ;;
        trtllm)
            gpu_gb_to_free_fraction "$_EW_KV_GIB" "$gpu_idx" ;;
        *)
            echo "gpu_worker_fraction: unknown engine '$engine'" >&2
            echo "Supported: vllm, sglang, trtllm" >&2
            return 1 ;;
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
    total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$gpu_idx" 2>/dev/null)
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
# This function queries CURRENT free memory from nvidia-smi and computes
# gib / free_mib. The result is a best-effort estimate: TensorRT-LLM will
# see less free memory than we measure here (model weights haven't loaded
# yet), so the actual KV cache allocation will be smaller than <gib>.
# For rough sizing this is fine; for precise control use the YAML config
# with a known model size.
#
# For disagg_same_gpu (two workers sharing one GPU), launch workers
# sequentially: start the first, wait for it to finish loading (poll
# nvidia-smi or logs), then query free memory again and compute the
# fraction for the second worker. This gives predictable per-worker
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
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_idx" 2>/dev/null)
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

