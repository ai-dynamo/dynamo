#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal vllm serve wrapper for benchmark sweeps.
# Launched by the sweep orchestrator via: bash vllm_serve.sh --model <model> [extra_args...]
#
# When $DYN_NSYS_CAPTURE=cudaProfilerApi is set, the wrapper prefixes nsys
# around `vllm serve` and passes `--profiler-config.profiler cuda` so the
# orchestrator's /start_profile + /stop_profile HTTP triggers drive the
# capture window. See plans/2026-04-21/nsys-sweep-capture-range/plan.md.

set -e

MODEL=""
CAPACITY_GB=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"; shift 2 ;;
        --multimodal-embedding-cache-capacity-gb)
            CAPACITY_GB="$2"; shift 2 ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "ERROR: --model is required" >&2
    exit 1
fi

EC_ARGS=()
if [[ "$CAPACITY_GB" != "0" ]]; then
    EC_ARGS=(--ec-transfer-config "{
        \"ec_role\": \"ec_both\",
        \"ec_connector\": \"DynamoMultimodalEmbeddingCacheConnector\",
        \"ec_connector_module_path\": \"dynamo.vllm.multimodal_utils.multimodal_embedding_cache_connector\",
        \"ec_connector_extra_config\": {\"multimodal_embedding_cache_capacity_gb\": $CAPACITY_GB}
    }")
fi

GPU_MEM_UTIL=".9"
KV_BYTES="${_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES:-}"
if [[ -n "$KV_BYTES" ]]; then
    GPU_MEM_ARGS="--kv-cache-memory-bytes $KV_BYTES --gpu-memory-utilization 0.01"
else
    GPU_MEM_ARGS="--gpu-memory-utilization $GPU_MEM_UTIL"
fi

NSYS_PREFIX=()
VLLM_PROFILER_ARGS=()
if [[ "${DYN_NSYS_CAPTURE:-}" == "cudaProfilerApi" ]]; then
    : "${DYN_NSYS_OUT:?DYN_NSYS_OUT is required when DYN_NSYS_CAPTURE=cudaProfilerApi}"
    NSYS_BIN="${DYN_NSYS_BIN:-$(command -v nsys)}"
    if [[ -z "$NSYS_BIN" ]] || [[ ! -x "$NSYS_BIN" ]]; then
        echo "FATAL: nsys not found on PATH (set DYN_NSYS_BIN to override)." >&2
        exit 1
    fi

    # Preflight: assert this nsys build supports the shutdown model the
    # orchestrator-triggered capture depends on. Cheap (~1s, /bin/true target).
    PREFLIGHT_OUT="$(mktemp -u /tmp/nsys_preflight_XXXXXX.nsys-rep)"
    if ! "$NSYS_BIN" profile \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop-shutdown \
        --kill=sigterm \
        -o "$PREFLIGHT_OUT" \
        --force-overwrite=true \
        /bin/true >/dev/null 2>&1; then
        echo "FATAL: $NSYS_BIN lacks --capture-range-end=stop-shutdown + --kill=sigterm support." >&2
        echo "       Upgrade nsys or drop nsys_capture from the sweep config." >&2
        rm -f "$PREFLIGHT_OUT"* 2>/dev/null
        exit 1
    fi
    rm -f "$PREFLIGHT_OUT"* 2>/dev/null

    mkdir -p "$(dirname "$DYN_NSYS_OUT")"
    # Note: `--process-scope=process-tree` exists on newer nsys builds and
    # would bound the trace to the target's descendants. It's not accepted by
    # nsys 2025.5.2, so we rely on `--sample=none --cpuctxsw=none` alone to
    # keep host-thread noise down. Upgrade once available.
    NSYS_PREFIX=(
        "$NSYS_BIN" profile
        --trace=nvtx,cuda
        --sample=none
        --cpuctxsw=none
        --capture-range=cudaProfilerApi
        --capture-range-end=stop-shutdown
        --kill=sigterm
        -o "$DYN_NSYS_OUT"
        --force-overwrite=true
    )
    VLLM_PROFILER_ARGS=(--profiler-config.profiler cuda)
    echo "[vllm_serve.sh] nsys capture armed: $DYN_NSYS_OUT" >&2
fi

# exec so nsys becomes the tracked process directly (no lingering bash);
# ServerManager._process then tracks nsys's PID with clean pgroup semantics.
exec "${NSYS_PREFIX[@]}" vllm serve "$MODEL" \
    --enable-log-requests \
    --max-model-len 16384 \
    $GPU_MEM_ARGS \
    "${EC_ARGS[@]}" \
    "${VLLM_PROFILER_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
