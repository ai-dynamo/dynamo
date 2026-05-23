#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# Launch ONE plain aggregated vLLM instance with the KVBM v2 connector for the
# remote-search smoke. No P/D specialization — `kv_role=kv_both`.
#
# Differs from kvindex-smoke/launch-instance.sh only in the rendered feature
# set: `indexer,p2p` instead of `indexer`. The connector therefore (a) wires its
# KV-index publisher AND (b) registers Feature::P2P (hub-discoverable transfer
# peer with a session factory) — both required for remote search. The
# `leader.remote_search.enabled=true` toggle is seeded into the hub base_config
# (start-hub.sh KVBM_HUB_KVBM) and rendered into every connector's config.
#
# Usage:
#   KVBM_INSTANCE_PORT=8000 bash launch-instance.sh   # FOREGROUND; background from caller
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${KVBM_REPO:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
. "$REPO/.claude/skills/disagg-bringup/hardware-profiles.sh"
kvbm_apply_disagg_bringup_profile
. "$REPO/.claude/skills/kvbm-hub-bringup/hub-lib.sh"

KVBM_VENV=${KVBM_VENV:-$REPO/.sandbox}
KVBM_CONNECTOR_MODULE_PATH=${KVBM_CONNECTOR_MODULE_PATH:-kvbm.v2.vllm.connector}
KVBMCTL=${KVBM_KVBMCTL_BIN:-$REPO/target/debug/kvbmctl}
PORT=${KVBM_INSTANCE_PORT:-8000}
HUB_URL=${KVBM_HUB_URL:-http://127.0.0.1:1337}
GMU=${KVBM_GPU_MEMORY_UTILIZATION:-0.15}

# Render from the live hub with BOTH features so the connector requests p2p +
# indexer (handshake intersects requested∩offered; p2p must be requested for
# wire_p2p to fire). remote_search.enabled rides in via the hub base_config.
KV_RENDERED=$(kvbm_hub_render_vllm "$KVBMCTL" "$HUB_URL" indexer,p2p \
    --kv-connector-module-path "$KVBM_CONNECTOR_MODULE_PATH") \
  || { echo "kvbmctl render failed (is the hub up at $HUB_URL?)" >&2; exit 1; }
eval "KV_ARGS=( $KV_RENDERED )"

export CUDA_VISIBLE_DEVICES="${KVBM_SINGLE_CUDA_VISIBLE_DEVICES:-0}"
export DYN_KVBM_CPU_CACHE_GB="$KVBM_CPU_CACHE_GB"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export KVBM_SKIP_VLLM_VERSION_CHECK=${KVBM_SKIP_VLLM_VERSION_CHECK:-1}

# nixl libs on LD_LIBRARY_PATH (no baked rpath); inherited by EngineCore subproc.
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

exec "$KVBM_VENV/bin/python3" -m vllm.entrypoints.openai.api_server \
  --model "$KVBM_MODEL" \
  --served-model-name "$KVBM_MODEL" \
  --max-num-seqs "$KVBM_MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GMU" \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --port "$PORT" \
  "${KV_ARGS[@]}"
