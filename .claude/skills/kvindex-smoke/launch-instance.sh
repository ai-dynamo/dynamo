#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# Launch ONE plain aggregated vLLM instance with the KVBM v2 connector for the
# kvindex smoke. No P/D specialization — `kv_role=kv_both`, no `disagg` block.
#
# The instance publishes G2 block events to the hub's KV index: the hub URL is
# injected under `kv_connector_extra_config.leader.events.kv_index_hub_url` so
# it survives vLLM's EngineCore subprocess spawn (an env var would not — see
# disagg-bringup/SKILL.md's universal-mode note). The connector probes that hub
# for the KV-indexer feature and, when present and block-size-compatible, wires
# a ZMQ PUB publisher onto the block-registry EventsManager.
#
# Usage:
#   KVBM_INSTANCE_PORT=8000 bash launch-instance.sh   # FOREGROUND; background from caller
#
# Env vars (with defaults):
#   KVBM_VENV               (default: <repo>/.sandbox)
#   KVBM_INSTANCE_PORT      (default: 8000)
#   KVBM_HUB_URL            (default: http://127.0.0.1:1337) — discovery base
#   KVBM_CONNECTOR_MODULE_PATH (default: kvbm.v2.vllm.connector)
#   KVBM_HARDWARE_PROFILE   (default: spark-gb10) — sizing via hardware-profiles.sh
#   KVBM_GPU_MEMORY_UTILIZATION (default: 0.15 — two instances share one GB10)
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${KVBM_REPO:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
# Reuse the disagg-bringup hardware profile for model + sizing.
. "$REPO/.claude/skills/disagg-bringup/hardware-profiles.sh"
kvbm_apply_disagg_bringup_profile

KVBM_VENV=${KVBM_VENV:-$REPO/.sandbox}
KVBM_CONNECTOR_MODULE_PATH=${KVBM_CONNECTOR_MODULE_PATH:-kvbm.v2.vllm.connector}
PORT=${KVBM_INSTANCE_PORT:-8000}
HUB_URL=${KVBM_HUB_URL:-http://127.0.0.1:1337}
# Two instances coexist on one GB10 — keep utilization low.
GMU=${KVBM_GPU_MEMORY_UTILIZATION:-0.15}

export CUDA_VISIBLE_DEVICES="${KVBM_SINGLE_CUDA_VISIBLE_DEVICES:-0}"
export DYN_KVBM_CPU_CACHE_GB="$KVBM_CPU_CACHE_GB"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# This worktree's .sandbox may carry a vLLM nightly newer than kvbm's tested
# range (the exact 0.19.x cu130 nightly wheels get GC'd from the index). The
# v2 connector's KVConnector hooks are stable across these; bypass the guard.
export KVBM_SKIP_VLLM_VERSION_CHECK=${KVBM_SKIP_VLLM_VERSION_CHECK:-1}

# kvbm's _core.so links nixl 0.10.x and is built without a baked rpath (the
# maturin patchelf step is best-effort). Put the venv's nixl libs on
# LD_LIBRARY_PATH so `import kvbm` resolves `libnixl.so` — this env var IS
# inherited by vLLM's EngineCore subprocess (unlike figment KVBM_* vars).
# Honors $KVBM_NIXL_LIBS override; otherwise auto-detects the cu12/cu13 wheel.
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
  --max-model-len "$KVBM_MAX_MODEL_LEN" \
  --max-num-seqs "$KVBM_MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GMU" \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --port "$PORT" \
  --kv-transfer-config '{
    "kv_connector": "DynamoConnector",
    "kv_role": "kv_both",
    "kv_load_failure_policy": "recompute",
    "kv_connector_module_path": "'"$KVBM_CONNECTOR_MODULE_PATH"'",
    "kv_connector_extra_config": {
      "default": { "block_layout": "operational" },
      "leader": {
        "events":  { "kv_index_hub_url": "'"$HUB_URL"'" },
        "cache":   { "host": { "cache_size_gb": '"$KVBM_CPU_CACHE_GB"' } },
        "tokio":   { "worker_threads": 2 },
        "control": { "metrics": true }
      },
      "worker": {
        "nixl":  { "backends": { "UCX": {}, "POSIX": {} } },
        "tokio": { "worker_threads": 2 }
      }
    }
  }'
