#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# Launch the Prefill vLLM instance for the disagg-bringup smoke.
#
# The `--kv-transfer-config` blob is RENDERED by kvbmctl from the live hub
# (block_size / max_model_len / block_layout / leader.hub all come from the hub
# aggregate; `--features disagg --role prefill` resolves the disagg feature +
# its p2p dependency and pins the CD role). Only free fields (onboard mode,
# tokio workers, control metrics, nixl backends) are passed as `--kvbm`
# overrides. Replaces the old hand-written `leader.disagg.hub_url` schema (the
# `hub_url` knob was removed — `leader.hub.url` is the only hub path now).
#
# Env vars:
#   KVBM_VENV          (default: <repo>/.sandbox)
#   KVBM_HARDWARE_PROFILE (default: spark-gb10; also h100-a100, custom)
#   KVBM_HUB_URL       (default: http://127.0.0.1:1337) — discovery base
#   KVBM_KVBMCTL_BIN   (default: <repo>/target/debug/kvbmctl — built by start-hub.sh)
#   KVBM_ONBOARD_MODE  (default: inter) — inter | intra (free field)
#   KVBM_CONNECTOR_MODULE_PATH (default: kvbm.v2.vllm.connector)
#   NOTE: block_layout is hub-authoritative now (set via the hub's --layout,
#   i.e. KVBM_HUB_LAYOUT at start-hub); it is no longer a launcher knob.
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
KVBM_ONBOARD_MODE=${KVBM_ONBOARD_MODE:-inter}
case "$KVBM_ONBOARD_MODE" in
  inter|intra) ;;
  *) echo "KVBM_ONBOARD_MODE must be 'inter' or 'intra', got: '$KVBM_ONBOARD_MODE'" >&2; exit 1 ;;
esac

KV_RENDERED=$(kvbm_hub_render_vllm "$KVBMCTL" "$HUB_URL" disagg \
    --role prefill \
    --kv-connector-module-path "$KVBM_CONNECTOR_MODULE_PATH" \
    --kvbm leader.onboard.mode="$KVBM_ONBOARD_MODE" \
    --kvbm leader.tokio.worker_threads=2 \
    --kvbm worker.tokio.worker_threads=2 \
    --kvbm leader.control.dev=true \
    --kvbm leader.control.metrics=true \
    --kvbm 'worker.nixl.backends.UCX={}' \
    --kvbm 'worker.nixl.backends.POSIX={}') \
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

# --block-size / --max-model-len / --kv-transfer-config come from kvbmctl.
exec "$KVBM_VENV/bin/python3" -m vllm.entrypoints.openai.api_server \
  --model "$KVBM_MODEL" \
  --served-model-name "$KVBM_MODEL" \
  --max-num-seqs "$KVBM_MAX_NUM_SEQS" \
  --gpu-memory-utilization "$KVBM_PREFILL_GPU_MEMORY_UTILIZATION" \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --port 8000 \
  "${KV_ARGS[@]}"
