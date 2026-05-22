#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# Launch ONE plain aggregated vLLM instance with the KVBM v2 connector for the
# kvindex smoke. No P/D specialization — `kv_role=kv_both`, no `disagg` block.
#
# The instance publishes G2 block events to the hub's KV index. The connector
# reaches the hub via `kv_connector_extra_config.leader.hub.{url,features}`
# (injected into the `--kv-transfer-config` JSON so it survives vLLM's EngineCore
# subprocess spawn — an env var would not; see disagg-bringup/SKILL.md's
# universal-mode note). At startup the connector pulls `GET {hub}/v1/config`,
# resolves the `indexer` feature, registers `Feature::Indexer`, and wires a
# ZMQ PUB publisher onto the block-registry EventsManager from the aggregate's
# advertised endpoint.
#
# The `--kv-transfer-config` blob is RENDERED by `kvbmctl` from the live hub
# (block_size / max_model_len / block_layout / leader.hub all come from the hub
# aggregate). Common free fields (tokio workers, nixl backends, control.metrics)
# are seeded into the hub's base_config via KVBM_HUB_KVBM in start-hub.sh;
# no per-launcher --kvbm overrides are needed here.
#
# Usage:
#   KVBM_INSTANCE_PORT=8000 bash launch-instance.sh   # FOREGROUND; background from caller
#
# Env vars (with defaults):
#   KVBM_VENV               (default: <repo>/.sandbox)
#   KVBM_INSTANCE_PORT      (default: 8000)
#   KVBM_HUB_URL            (default: http://127.0.0.1:1337) — discovery base
#   KVBM_KVBMCTL_BIN        (default: <repo>/target/debug/kvbmctl — built by start-hub.sh)
#   KVBM_CONNECTOR_MODULE_PATH (default: kvbm.v2.vllm.connector)
#   KVBM_HARDWARE_PROFILE   (default: spark-gb10) — sizing via hardware-profiles.sh
#   KVBM_GPU_MEMORY_UTILIZATION (default: 0.15 — two instances share one GB10)
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${KVBM_REPO:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
# Reuse the disagg-bringup hardware profile for model + sizing.
. "$REPO/.claude/skills/disagg-bringup/hardware-profiles.sh"
kvbm_apply_disagg_bringup_profile
# Reusable hub helpers (kvbm_hub_render_vllm).
. "$REPO/.claude/skills/kvbm-hub-bringup/hub-lib.sh"

KVBM_VENV=${KVBM_VENV:-$REPO/.sandbox}
KVBM_CONNECTOR_MODULE_PATH=${KVBM_CONNECTOR_MODULE_PATH:-kvbm.v2.vllm.connector}
KVBMCTL=${KVBM_KVBMCTL_BIN:-$REPO/target/debug/kvbmctl}
PORT=${KVBM_INSTANCE_PORT:-8000}
HUB_URL=${KVBM_HUB_URL:-http://127.0.0.1:1337}
# Two instances coexist on one GB10 — keep utilization low.
GMU=${KVBM_GPU_MEMORY_UTILIZATION:-0.15}

# Render the vLLM connector args from the live hub via the shared helper
# (kvbm_hub_render_vllm checks the binary exists). kvbmctl emits
# `--block-size <N> --max-model-len <M> --kv-transfer-config '{…}'`; the hub is
# the source of truth for those. Common free fields (tokio workers, nixl
# backends, control.metrics) are seeded into the hub's base_config via
# KVBM_HUB_KVBM in start-hub.sh, so no per-launcher --kvbm overrides are needed.
KV_RENDERED=$(kvbm_hub_render_vllm "$KVBMCTL" "$HUB_URL" indexer \
    --kv-connector-module-path "$KVBM_CONNECTOR_MODULE_PATH") \
  || { echo "kvbmctl render failed (is the hub up at $HUB_URL?)" >&2; exit 1; }

# kvbmctl shell-quotes the (space-free) JSON; eval re-parses the quotes so the
# blob stays a single argv element. Output is trusted (we built the renderer).
eval "KV_ARGS=( $KV_RENDERED )"

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

# --block-size, --max-model-len, and --kv-transfer-config come from kvbmctl
# (${KV_ARGS[@]}); do not also pass --max-model-len here — vLLM rejects dupes.
# The connector module path is overridable via KVBM_CONNECTOR_MODULE_PATH; pass
# it through kvbmctl's flag so the rendered blob carries it.
exec "$KVBM_VENV/bin/python3" -m vllm.entrypoints.openai.api_server \
  --model "$KVBM_MODEL" \
  --served-model-name "$KVBM_MODEL" \
  --max-num-seqs "$KVBM_MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GMU" \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --port "$PORT" \
  "${KV_ARGS[@]}"
