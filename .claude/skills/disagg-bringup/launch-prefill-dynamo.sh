#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Variant of `launch-prefill.sh` that runs the dynamo.vllm worker in
# prefill mode. The hub-side wiring is identical to the kvbm.vllm.prefill
# variant: the rendered kv-transfer-config carries the leader.disagg.role +
# leader.hub.url signal, and worker_factory.py:627-675 auto-attaches a
# PrefillRouterHandler after engine creation -> Velo backend registration
# with the hub's prefill router.
#
# The dynamo.vllm worker also registers Dynamo runtime endpoints over NATS
# + etcd, but nothing in this smoke consumes them — requests flow:
#     client -> decode (vllm serve :8001) -> hub PrefillRouter (velo) -> prefill (dynamo.vllm)
#
# Used by `disagg-smoke/prefill-router-smoke.sh` as variant B.
# Requires NATS (localhost:4222) and etcd (localhost:2379) running.
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

# Dynamo runtime needs NATS + etcd; default to localhost where they
# already run on this box.
export NATS_SERVER=${NATS_SERVER:-nats://localhost:4222}
export ETCD_ENDPOINTS=${ETCD_ENDPOINTS:-http://localhost:2379}
export DYN_NAMESPACE=${DYN_NAMESPACE:-dynamo}

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

# dynamo.vllm uses its own arg parser (Dynamo wrapper around vLLM's). The
# prefill worker doesn't serve HTTP; the auto-attach hook in
# worker_factory.py:627-675 registers this process with the hub's prefill
# router via Velo, and dispatch happens over velo (not HTTP). We still need
# --kv-transfer-config from kvbmctl so the engine carries the hub URL +
# disagg role that the auto-attach hook reads.
exec "$KVBM_VENV/bin/python3" -m dynamo.vllm \
  --disaggregation-mode prefill \
  --model "$KVBM_MODEL" \
  --served-model-name "$KVBM_MODEL" \
  --max-num-seqs "$KVBM_MAX_NUM_SEQS" \
  --gpu-memory-utilization "$KVBM_PREFILL_GPU_MEMORY_UTILIZATION" \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  "${KV_ARGS[@]}"
