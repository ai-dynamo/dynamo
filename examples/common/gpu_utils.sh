#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shared GPU utility functions for launch scripts.
#
# CLI:
#   ./gpu_utils.sh <engine> --model <name> [options...]   Print GPU fraction
#   ./gpu_utils.sh --self-test                            Run self-test suite
#
# Source:
#   source "$(dirname "$(readlink -f "$0")")/../common/gpu_utils.sh"
#   # or with SCRIPT_DIR already set:
#   source "$SCRIPT_DIR/../common/gpu_utils.sh"
#
# Functions (all return via stdout — no hidden globals):
#   build_gpu_mem_args <engine> <model> ...     Prints fraction (or empty)
#   get_model_params <model>                    Prints "pb wb layers kvh hd"
#   estimate_worker_vram <model> ...            Prints "w_gib kv_gib oh_gib total_gib"
#   gpu_worker_fraction <engine> <total> <kv>   Prints engine-appropriate fraction
#   gpu_peak_to_engine_fraction <engine> <peak> Prints fraction (subtracts engine overhead)
#   gpu_gb_to_total_fraction <gib>              Prints fraction of TOTAL VRAM (vLLM/sglang)
#   gpu_gb_to_free_fraction <gib>               Prints fraction of FREE VRAM (TensorRT-LLM)

# build_gpu_mem_args <engine> [options...]
#
# Prints the computed memory fraction to stdout (empty line if none).
# Callers capture with:  GPU_MEM_FRACTION=$(build_gpu_mem_args ...)
#
# Priority:
#   1. _PROFILE_PYTEST_VRAM_FRAC_OVERRIDE  (profiler binary search)
#   2. Engine flag passed to this function  (user already chose a value)
#   3. estimate_worker_vram + gpu_worker_fraction  (model architecture)
#   4. Empty  (let engine use its own default)
#
# Options (each flag accepts engine-specific aliases):
#   --model NAME                 Model name (required).
#     aliases: --model-path        (sglang, trtllm)
#   --max-model-len N            Max tokens per sequence (default: 4096).
#     aliases: --context-length    (sglang)
#              --max-seq-len       (trtllm)
#   --max-num-seqs N             Concurrent sequences to budget for (default: 2).
#     aliases: --max-running-requests (sglang)
#              --max-batch-size       (trtllm)
#   --gpu-memory-utilization F   User override (vllm flag name).  Skipped when empty.
#   --mem-fraction-static F      User override (sglang flag name).
#   --workers-per-gpu N          Divide the fraction by N (for shared-GPU disagg).
#
# Usage:
#   # Simple single-worker (agg.sh)
#   GPU_MEM_FRACTION=$(build_gpu_mem_args vllm \
#       --model "$MODEL" --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_CONCURRENT_SEQS")
#   python -m dynamo.vllm --model "$MODEL" \
#       ${GPU_MEM_FRACTION:+--gpu-memory-utilization "$GPU_MEM_FRACTION"} &
#
#   # Two workers sharing one GPU (disagg_same_gpu.sh)
#   GPU_MEM_FRACTION=$(build_gpu_mem_args vllm --model "$MODEL" --workers-per-gpu 2)
#   python -m dynamo.vllm ... --gpu-memory-utilization "${GPU_MEM_FRACTION}" &
#
#   # sglang
#   GPU_MEM_FRACTION=$(build_gpu_mem_args sglang --model "$MODEL" --workers-per-gpu 2)
#   python -m dynamo.sglang ... --mem-fraction-static "${GPU_MEM_FRACTION}" &
#
#   # trtllm (fraction goes into JSON, not CLI)
#   GPU_MEM_FRACTION=$(build_gpu_mem_args trtllm --model "$MODEL" --workers-per-gpu 2)
#   OVERRIDE_ARGS=(--override-engine-args "{\"kv_cache_config\":{\"free_gpu_memory_fraction\":${GPU_MEM_FRACTION}}}")
build_gpu_mem_args() {
    local engine="${1:?usage: build_gpu_mem_args <engine> --model <name> [options...]}"
    shift

    local model=""
    local max_model_len="4096"
    local max_seqs="2"
    local workers_per_gpu=1
    local user_frac=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model|--model-path)
                                model="$2";           shift 2 ;;
            --max-model-len|--context-length|--max-seq-len)
                                max_model_len="$2";   shift 2 ;;
            --max-num-seqs|--max-running-requests|--max-batch-size)
                                max_seqs="$2";        shift 2 ;;
            --gpu-memory-utilization|--mem-fraction-static)
                                user_frac="$2";       shift 2 ;;
            --workers-per-gpu)  workers_per_gpu="$2"; shift 2 ;;
            *) echo "build_gpu_mem_args: unknown option '$1'" >&2; return 1 ;;
        esac
    done

    if [[ -z "$model" ]]; then
        echo "build_gpu_mem_args: --model is required" >&2
        return 1
    fi

    local frac=""
    local from_estimator=false
    local est_w="" est_kv="" est_oh="" est_total=""
    if [[ -n "${_PROFILE_PYTEST_VRAM_FRAC_OVERRIDE:-}" ]]; then
        frac="$_PROFILE_PYTEST_VRAM_FRAC_OVERRIDE"
    elif [[ -n "$user_frac" ]]; then
        frac="$user_frac"
    elif read -r est_w est_kv est_oh est_total <<< "$(estimate_worker_vram "$model" "$max_model_len" "$max_seqs" "$engine" 2>/dev/null)" && [[ -n "$est_total" ]]; then
        frac=$(gpu_worker_fraction "$engine" "$est_total" "$est_kv")
        from_estimator=true
    fi

    # --workers-per-gpu divides profiler/user/estimator results only
    if [[ -n "$frac" && "$workers_per_gpu" -gt 1 ]]; then
        frac=$(awk -v f="$frac" -v n="$workers_per_gpu" 'BEGIN { printf "%.2f", f / n }')
    fi

    echo "$frac"
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

    echo "=== get_model_params ==="

    local out
    out=$(get_model_params "Qwen/Qwen3-0.6B")
    _assert "known model returns 5 fields" "0.6 2 28 8 128" "$out"

    out=$(get_model_params "nope/unknown" 2>/dev/null)
    _assert "unknown model returns empty" "" "$out"

    get_model_params "nope/unknown" >/dev/null 2>&1
    _assert "unknown model exits 1" "1" "$?"

    echo ""
    echo "=== estimate_worker_vram ==="

    out=$(estimate_worker_vram "Qwen/Qwen3-0.6B" 4096 2 vllm)
    _assert "returns 4 space-separated fields" "4" "$(echo "$out" | wc -w | tr -d ' ')"

    local w kv oh total
    read -r w kv oh total <<< "$out"
    _assert "weights > 0" "yes" "$(awk -v v="$w" 'BEGIN { print (v > 0) ? "yes" : "no" }')"
    _assert "total > weights" "yes" "$(awk -v t="$total" -v w="$w" 'BEGIN { print (t > w) ? "yes" : "no" }')"

    out=$(estimate_worker_vram "nope/unknown" 2>/dev/null)
    _assert "unknown model returns empty" "" "$out"

    local out_vllm out_sglang
    out_vllm=$(estimate_worker_vram "Qwen/Qwen3-0.6B" 4096 2 vllm)
    out_sglang=$(estimate_worker_vram "Qwen/Qwen3-0.6B" 4096 2 sglang)
    _assert "sglang overhead > vllm overhead" "yes" \
        "$(awk -v v="$out_vllm" -v s="$out_sglang" 'BEGIN {
            split(v, a); split(s, b); print (b[3]+0 > a[3]+0) ? "yes" : "no"
        }')"

    echo ""
    echo "=== build_gpu_mem_args: estimator path (known model) ==="

    local frac
    frac=$(build_gpu_mem_args vllm --model "Qwen/Qwen3-0.6B" --max-model-len 4096 --max-num-seqs 2)
    _assert "FRACTION non-empty" "yes" "$([[ -n "$frac" ]] && echo yes || echo no)"

    echo ""
    echo "=== build_gpu_mem_args: unknown model, no default ==="

    frac=$(build_gpu_mem_args vllm --model "nope/unknown")
    _assert "FRACTION empty" "" "$frac"

    echo ""
    echo "=== build_gpu_mem_args: profiler wins over all ==="

    frac=$(_PROFILE_PYTEST_VRAM_FRAC_OVERRIDE=0.55 \
        build_gpu_mem_args vllm --model "Qwen/Qwen3-0.6B" --gpu-memory-utilization 0.70)
    _assert "FRACTION = profiler (beats user flag)" "0.55" "$frac"

    echo ""
    echo "=== build_gpu_mem_args: user flag wins over estimator ==="

    frac=$(build_gpu_mem_args vllm --model "Qwen/Qwen3-0.6B" --gpu-memory-utilization 0.70)
    _assert "FRACTION = user flag" "0.70" "$frac"

    echo ""
    echo "=== build_gpu_mem_args: empty user flag falls through ==="

    frac=$(build_gpu_mem_args vllm --model "Qwen/Qwen3-0.6B" --max-model-len 4096 --max-num-seqs 2 --gpu-memory-utilization "")
    _assert "FRACTION = estimator" "yes" "$([[ -n "$frac" ]] && echo yes || echo no)"

    echo ""
    echo "=== build_gpu_mem_args: --workers-per-gpu divides estimator ==="

    local undivided
    undivided=$(build_gpu_mem_args vllm --model "Qwen/Qwen3-0.6B" --max-model-len 4096 --max-num-seqs 2)
    frac=$(build_gpu_mem_args vllm --model "Qwen/Qwen3-0.6B" --max-model-len 4096 --max-num-seqs 2 --workers-per-gpu 2)
    local expected_half
    expected_half=$(awk -v f="$undivided" 'BEGIN { printf "%.2f", f / 2 }')
    _assert "FRACTION halved" "$expected_half" "$frac"

    echo ""
    echo "=== build_gpu_mem_args: --workers-per-gpu divides profiler ==="

    frac=$(_PROFILE_PYTEST_VRAM_FRAC_OVERRIDE=0.80 \
        build_gpu_mem_args vllm --model "Qwen/Qwen3-0.6B" --workers-per-gpu 2)
    _assert "FRACTION = 0.80/2 = 0.40" "0.40" "$frac"

    echo ""
    echo "=== build_gpu_mem_args: sglang engine (sglang flag names) ==="

    frac=$(build_gpu_mem_args sglang --model-path "Qwen/Qwen3-0.6B" --context-length 4096 --max-running-requests 2)
    _assert "sglang FRACTION non-empty" "yes" "$([[ -n "$frac" ]] && echo yes || echo no)"

    echo ""
    echo "=== build_gpu_mem_args: trtllm engine (trtllm flag names) ==="

    frac=$(build_gpu_mem_args trtllm --model-path "Qwen/Qwen3-0.6B" --max-seq-len 4096 --max-batch-size 2)
    _assert "trtllm FRACTION non-empty" "yes" "$([[ -n "$frac" ]] && echo yes || echo no)"

    echo ""
    echo "=== build_gpu_mem_args: --mem-fraction-static user flag (sglang) ==="

    frac=$(build_gpu_mem_args sglang --model-path "Qwen/Qwen3-0.6B" --mem-fraction-static 0.60)
    _assert "FRACTION = user flag" "0.60" "$frac"

    echo ""
    echo "=== build_gpu_mem_args: missing --model ==="

    build_gpu_mem_args vllm 2>/dev/null
    _assert "missing --model exits 1" "1" "$?"

    echo ""
    echo "=== gpu_worker_fraction: explicit args ==="

    local frac
    frac=$(gpu_worker_fraction vllm 4.0 0.9)
    _assert "vllm returns non-empty" "yes" "$([[ -n "$frac" ]] && echo yes || echo no)"

    frac=$(gpu_worker_fraction trtllm 4.0 0.9)
    _assert "trtllm returns non-empty" "yes" "$([[ -n "$frac" ]] && echo yes || echo no)"

    gpu_worker_fraction badengine 4.0 0.9 >/dev/null 2>&1
    _assert "bad engine exits 1" "1" "$?"

    echo ""
    echo "=========================================="
    echo "Results: $pass passed, $fail failed"
    echo "=========================================="
    [[ "$fail" -eq 0 ]]
}

# CLI mode: only when executed directly (not sourced by another script)
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    if [[ "${1:-}" == "--self-test" ]]; then
        _gpu_utils_self_test
        exit $?
    fi
    if [[ $# -gt 0 ]]; then
        build_gpu_mem_args "$@"
        exit $?
    fi

    cat <<'HELP'
gpu_utils.sh — GPU memory fraction estimator

Usage:
  ./gpu_utils.sh <engine> --model <name> [options...]
  ./gpu_utils.sh --self-test

Engines: vllm, sglang, trtllm

Examples:
  ./gpu_utils.sh vllm --model Qwen/Qwen3-0.6B
  ./gpu_utils.sh vllm --model Qwen/Qwen3-0.6B --max-model-len 4096 --max-num-seqs 2
  ./gpu_utils.sh vllm --model Qwen/Qwen3-0.6B --workers-per-gpu 2
  ./gpu_utils.sh sglang --model Qwen/Qwen3-0.6B --context-length 8192
  ./gpu_utils.sh trtllm --model meta-llama/Meta-Llama-3.1-8B-Instruct --max-seq-len 4096

Options:
  --model NAME               Model name (required)
    aliases: --model-path
  --max-model-len N          Max sequence length (default: 4096)
    aliases: --context-length, --max-seq-len
  --max-num-seqs N           Concurrent sequences (default: 2)
    aliases: --max-running-requests, --max-batch-size
  --gpu-memory-utilization F Override fraction (vllm flag)
    aliases: --mem-fraction-static
  --workers-per-gpu N        Divide fraction by N (shared-GPU disagg)
  --self-test                Run built-in test suite

Output: prints the fraction to stdout (empty if model is unknown).
HELP
    exit 0
fi
