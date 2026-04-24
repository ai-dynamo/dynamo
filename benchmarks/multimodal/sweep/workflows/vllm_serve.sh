#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# vllm serve wrapper for benchmark sweeps. Wraps the server in a single nsys
# instance (matching dynamo_serve.sh's defaults) so the vllm-serve config
# produces a comparable .nsys-rep alongside dynamo-fd / dynamo-fd-ec.
# Launched by the sweep orchestrator via: bash vllm_serve.sh --model <model> [extra_args...]

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

# ---------------------------------------------------------------------------
# nsys wrapping (mirrors dynamo_serve.sh conventions).
# ---------------------------------------------------------------------------
NSYS_BIN="${DYN_NSYS_BIN:-/opt/nvidia/nsight-systems-cli/2026.2.1/bin/nsys}"
# Default groups reps by day under /dynamo-tmp/MM-DD/nsys/ to match the sweep
# yaml's `output_dir: /dynamo-tmp/MM-DD/<slug>` pattern. Override via
# DYN_NSYS_DIR (e.g. yaml env) to co-locate with aiperf outputs.
NSYS_DIR="${DYN_NSYS_DIR:-/dynamo-tmp/$(date +%m-%d)/nsys}"
# Full process lifetime capture by default. delay=0 starts tracing immediately;
# duration=86400 (24 h) means --duration never fires. nsys only stops when the
# orchestrator sends SIGINT (see --kill=sigterm below) and finalizes cleanly.
# Avoids the windowed-capture footgun where --duration expires mid-load and
# kills the engine. Override via DYN_NSYS_DELAY_S / DYN_NSYS_DURATION_S.
NSYS_DELAY_S="${DYN_NSYS_DELAY_S:-0}"
NSYS_DURATION_S="${DYN_NSYS_DURATION_S:-86400}"

# Stage .qdstrm on scratch, not container /tmp (Pyxis overlay evaporates at
# job end, orphaning any .qdstrm from a SIGKILL'd finalize).
export TMPDIR="${DYN_NSYS_TMPDIR:-/dynamo-tmp/nsys-staging}"
mkdir -p "$TMPDIR"
mkdir -p "$NSYS_DIR"
TS=$(date +%Y%m%d_%H%M%S)
NSYS_OUT="$NSYS_DIR/vllm_$TS.nsys-rep"

DISABLE_NSYS="${DYN_DISABLE_NSYS:-0}"
if [[ "$DISABLE_NSYS" == "1" ]]; then
    echo "[nsys] DYN_DISABLE_NSYS=1 — running vllm serve without nsys wrap" >&2
fi

# Auto-install nsys from the mounted .deb if missing. Fail loudly — silent
# fall-through produces an unwrapped server and no trace artifact.
if [[ "$DISABLE_NSYS" != "1" && ! -x "$NSYS_BIN" ]]; then
    NSYS_DEB=$(ls /nsys/NsightSystems-linux-cli-public-*.deb 2>/dev/null | head -1)
    if [[ -n "$NSYS_DEB" ]]; then
        echo "[nsys] $NSYS_BIN missing; installing from $NSYS_DEB" >&2
        DEBIAN_FRONTEND=noninteractive apt-get update >&2 || true
        DEBIAN_FRONTEND=noninteractive apt-get install -y "$NSYS_DEB" >&2 || {
            echo "[nsys] apt install $NSYS_DEB failed; falling back to dpkg -i + apt --fix-broken" >&2
            dpkg -i "$NSYS_DEB" >&2 || true
            DEBIAN_FRONTEND=noninteractive apt-get -y --fix-broken install >&2 || {
                echo "[nsys] FATAL: all install paths failed; refusing to run without profiling" >&2
                exit 1
            }
        }
    else
        echo "[nsys] FATAL: $NSYS_BIN not executable and no .deb at /nsys/ — refusing to run without profiling" >&2
        exit 1
    fi
fi

if [[ "$DISABLE_NSYS" == "1" ]]; then
    NSYS_PREFIX=()
else
    NSYS_PREFIX=(
        "$NSYS_BIN" profile
        --trace=nvtx,cuda
        --sample=none
        --cpuctxsw=none
        --delay="$NSYS_DELAY_S"
        --duration="$NSYS_DURATION_S"
        # --kill=sigterm: on SIGINT, nsys STOPS tracing, SIGTERMs the child,
        # waits for exit, then finalizes the .nsys-rep.
        --kill=sigterm
        --force-overwrite=true
        -o "$NSYS_OUT"
    )
    echo "[nsys] vllm-serve -> $NSYS_OUT" >&2
fi

NSYS_PID=0

# Install the trap BEFORE launching so a fast orchestrator SIGINT during the
# startup window is still forwarded to nsys. Per project/code.md, profiler
# wrappers must forward SIGINT/SIGTERM — never `trap '' INT TERM`.
cleanup() {
    [[ "$NSYS_PID" -gt 0 ]] && kill -INT "$NSYS_PID" 2>/dev/null || true
    for _ in $(seq 1 150); do
        [[ "$NSYS_PID" -gt 0 ]] && kill -0 "$NSYS_PID" 2>/dev/null || break
        sleep 1
    done
    exit 0
}
trap cleanup INT TERM

"${NSYS_PREFIX[@]}" vllm serve "$MODEL" \
    --enable-log-requests \
    --max-model-len 16384 \
    $GPU_MEM_ARGS \
    "${EC_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" &
NSYS_PID=$!
wait "$NSYS_PID"
