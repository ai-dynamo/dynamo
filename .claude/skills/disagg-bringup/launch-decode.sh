#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# Launch the Decode vLLM instance for the disagg-bringup smoke.
# Same as launch-prefill.sh but `--role decode` and port 8001.
#
# The `--kv-transfer-config` blob is RENDERED by kvbmctl from the live hub
# (`--features disagg --role decode`). All deployment-wide config (tokio
# workers, nixl backends, control dev+metrics, onboard mode, block_layout) is
# hub-authoritative (seeded at start-hub); only the per-instance disagg role
# (--role decode) is supplied here.
#
# Env vars:
#   KVBM_VENV / KVBM_HARDWARE_PROFILE / KVBM_HUB_URL / KVBM_KVBMCTL_BIN
#   KVBM_CONNECTOR_MODULE_PATH (default: kvbm.v2.vllm.connector)
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
    --role decode \
    --kv-connector-module-path "$KVBM_CONNECTOR_MODULE_PATH") \
  || { echo "kvbmctl render failed (is the hub up at $HUB_URL with --features disagg?)" >&2; exit 1; }
eval "KV_ARGS=( $KV_RENDERED )"

export CUDA_VISIBLE_DEVICES="$KVBM_DECODE_CUDA_VISIBLE_DEVICES"
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

exec "$KVBM_VENV/bin/python3" -m vllm.entrypoints.openai.api_server \
  --model "$KVBM_MODEL" \
  --served-model-name "$KVBM_MODEL" \
  --max-num-seqs "$KVBM_MAX_NUM_SEQS" \
  --gpu-memory-utilization "$KVBM_DECODE_GPU_MEMORY_UTILIZATION" \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --port 8001 \
  "${KV_ARGS[@]}"
