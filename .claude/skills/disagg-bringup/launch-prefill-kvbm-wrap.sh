#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Variant of `launch-prefill.sh` that wraps vLLM via the new
# `python -m kvbm.vllm.prefill` entrypoint. Same kvbmctl render + env exports
# as the baseline launcher; only the final exec swaps
# `vllm.entrypoints.openai.api_server` for `kvbm.vllm.prefill`, which
# auto-attaches a `PrefillRouterHandler` after engine creation and registers
# this worker with the hub's prefill router as a Velo backend.
#
# Used by `disagg-smoke/prefill-router-smoke.sh` as variant A.
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${KVBM_REPO:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
. "$SCRIPT_DIR/hardware-profiles.sh"
kvbm_apply_disagg_bringup_profile
. "$REPO/.claude/skills/kvbm-hub-bringup/hub-lib.sh"

KVBM_VENV=${KVBM_VENV:-$REPO/.sandbox}
KVBM_CONNECTOR_MODULE_PATH=${KVBM_CONNECTOR_MODULE_PATH:-kvbm.v2.vllm.connector}
KVBMCTL=${KVBM_KVBMCTL_BIN:-$REPO/target/debug/kvbmctl}
HUB_URL=${KVBM_HUB_URL:-http://127.0.0.1:1337}

KV_RENDERED=$(kvbm_hub_render_vllm "$KVBMCTL" "$HUB_URL" disagg \
    --role prefill \
    --kv-connector-module-path "$KVBM_CONNECTOR_MODULE_PATH") \
  || { echo "kvbmctl render failed (is the hub up at $HUB_URL with --features disagg?)" >&2; exit 1; }
eval "KV_ARGS=( $KV_RENDERED )"

export CUDA_VISIBLE_DEVICES="$KVBM_PREFILL_CUDA_VISIBLE_DEVICES"
export DYN_KVBM_CPU_CACHE_GB="$KVBM_CPU_CACHE_GB"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export KVBM_SKIP_VLLM_VERSION_CHECK=${KVBM_SKIP_VLLM_VERSION_CHECK:-1}

NIXL_LIBS=${KVBM_NIXL_LIBS:-}
if [ -z "$NIXL_LIBS" ]; then
    for cand in "$KVBM_VENV"/lib/python*/site-packages/.nixl_cu12.mesonpy.libs \
                "$KVBM_VENV"/lib/python*/site-packages/.nixl_cu13.mesonpy.libs; do
        [ -d "$cand" ] && NIXL_LIBS="$cand" && break
    done
fi
if [ -n "$NIXL_LIBS" ]; then
    export LD_LIBRARY_PATH="$NIXL_LIBS:$NIXL_LIBS/plugins:${LD_LIBRARY_PATH:-}"
    export NIXL_PLUGIN_DIR="$NIXL_LIBS/plugins"
fi

# Same arg shape as launch-prefill.sh — model, served-model-name, port 8000,
# kv-transfer-config from kvbmctl. The kvbm.vllm.prefill entrypoint reuses
# vLLM's FlexibleArgumentParser and run_server verbatim; it differs only in
# the monkey-patched build_and_serve that runs kvbm.hub.try_wrap_engine.
exec "$KVBM_VENV/bin/python3" -m kvbm.vllm.prefill \
  --model "$KVBM_MODEL" \
  --served-model-name "$KVBM_MODEL" \
  --max-num-seqs "$KVBM_MAX_NUM_SEQS" \
  --gpu-memory-utilization "$KVBM_PREFILL_GPU_MEMORY_UTILIZATION" \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --port 8000 \
  "${KV_ARGS[@]}"
