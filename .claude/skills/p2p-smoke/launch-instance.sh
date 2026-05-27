#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# Launch ONE vLLM instance wired to kvbm-hub as a standalone P2P peer for the
# p2p smoke. No conditional-disagg role — the instance registers `Feature::P2P`
# (the P2.5 standalone-p2p connector path), gets its session factory + transfer
# control surface, and can serve / pull G2 block copies.
#
# The connector reaches the hub via `kv_connector_extra_config.leader.hub.
# {url, features:["p2p"]}` — RENDERED by `kvbmctl` from the live hub (so
# block_size / max_model_len / block_layout / leader.hub all come from the hub
# aggregate). Only free fields (tokio workers, nixl backends, control.metrics)
# are passed as `--kvbm` overrides. Mirrors kvindex-smoke/launch-instance.sh;
# the only difference is the rendered feature (`p2p` vs `indexer`).
#
# Usage:
#   P2P_PORT=8000 bash launch-instance.sh   # FOREGROUND; background from caller
#
# Env vars (with defaults):
#   PYTHON_BIN         (optional explicit Python; otherwise resolved from
#                       KVBM_VENV, runtime image, or PATH)
#   KVBM_VENV          (optional Python environment for runtime + NIXL lib search)
#   P2P_PORT           — vLLM listen port (e.g. 8000, 8002); required
#   KVBM_HUB_URL       (default: http://127.0.0.1:1337) — discovery base
#   KVBM_KVBMCTL_BIN   (default: <repo>/target/debug/kvbmctl — built by start-hub.sh)
#   P2P_HARDWARE_PROFILE (default: h100-a100; also spark-gb10, custom) — sizing
#   P2P_GMU / P2P_CACHE_GB / P2P_MAX_MODEL_LEN / P2P_MODEL — from the profile
#   P2P_CUDA_VISIBLE_DEVICES (default: 0)
#   P2P_ENFORCE_EAGER  (default: 1) — keep the functional smoke off the vLLM
#                       compile path; set 0 only when debugging compiled startup
#   KVBM_CONNECTOR_MODULE_PATH (default: kvbm.v2.vllm.connector)
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${KVBM_REPO:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
. "$REPO/.claude/skills/disagg-bringup/hardware-profiles.sh"
kvbm_apply_p2p_profile
# Reusable hub helpers (kvbm_hub_render_vllm).
. "$REPO/.claude/skills/kvbm-hub-bringup/hub-lib.sh"

resolve_python() {
  if [ -n "${PYTHON_BIN:-}" ]; then
    [ -x "$PYTHON_BIN" ] || { echo "PYTHON_BIN is set but not executable: $PYTHON_BIN" >&2; return 2; }
  elif [ -n "${KVBM_VENV:-}" ] && [ -x "$KVBM_VENV/bin/python3" ]; then
    PYTHON_BIN="$KVBM_VENV/bin/python3"
  elif [ -n "${KVBM_VENV:-}" ] && [ -x "$KVBM_VENV/bin/python" ]; then
    PYTHON_BIN="$KVBM_VENV/bin/python"
  elif [ -x /opt/dynamo/venv/bin/python3 ]; then
    KVBM_VENV=/opt/dynamo/venv
    PYTHON_BIN=/opt/dynamo/venv/bin/python3
  elif [ -x /opt/dynamo/venv/bin/python ]; then
    KVBM_VENV=/opt/dynamo/venv
    PYTHON_BIN=/opt/dynamo/venv/bin/python
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v python3)
  else
    echo "No usable Python found; set PYTHON_BIN or KVBM_VENV" >&2
    return 2
  fi
  export PYTHON_BIN
  [ -z "${KVBM_VENV:-}" ] || export KVBM_VENV
}
resolve_python

KVBM_CONNECTOR_MODULE_PATH=${KVBM_CONNECTOR_MODULE_PATH:-kvbm.v2.vllm.connector}
KVBMCTL=${KVBM_KVBMCTL_BIN:-$REPO/target/debug/kvbmctl}
PORT=${P2P_PORT:?"P2P_PORT required"}
HUB_URL=${KVBM_HUB_URL:-http://127.0.0.1:1337}
GMU=${P2P_GMU:-0.70}
CACHE_GB=${P2P_CACHE_GB:-2}
MAX_MODEL_LEN=${P2P_MAX_MODEL_LEN:-2048}
P2P_ENFORCE_EAGER=${P2P_ENFORCE_EAGER:-1}
case "$P2P_ENFORCE_EAGER" in
  1|true|yes|on) ENFORCE_EAGER_ARGS=(--enforce-eager) ;;
  0|false|no|off) ENFORCE_EAGER_ARGS=() ;;
  *) echo "P2P_ENFORCE_EAGER must be boolean, got $P2P_ENFORCE_EAGER" >&2; exit 2 ;;
esac

# Render the vLLM connector args from the live hub. The hub is the source of
# truth for block_size / max_model_len / block_layout / leader.hub; free fields
# (tokio workers, control metrics, nixl backends) are overrides.
#
# Features: `p2p,indexer`. p2p is the feature under test (remote-controllable
# block-copy peer, no disagg role). indexer is enabled purely as a *hash
# discovery* mechanism for the harness: vLLM's EngineCore subprocess does not
# surface the Rust `kvbm_audit` tracing, so the smoke cannot scrape offloaded
# block hashes from the instance log. Instead each instance publishes its G2
# block events to the hub index, and the orchestrator reads the holder's block
# hashes (as decimal `hash_u128`) back via the indexer HTTP API to feed
# `kvbmctl p2p pin`.
KV_RENDERED=$(kvbm_hub_render_vllm "$KVBMCTL" "$HUB_URL" p2p,indexer \
    --kv-connector-module-path "$KVBM_CONNECTOR_MODULE_PATH" \
    --kvbm leader.tokio.worker_threads=2 \
    --kvbm worker.tokio.worker_threads=2 \
    --kvbm leader.control.metrics=true \
    --kvbm 'worker.nixl.backends.UCX={}' \
    --kvbm 'worker.nixl.backends.POSIX={}') \
  || { echo "kvbmctl render failed (is the hub up at $HUB_URL with --features p2p?)" >&2; exit 1; }

# kvbmctl shell-quotes the JSON so the rendered fragment is pasteable. Parse it
# with shlex instead of eval, then require the exact vLLM flags this smoke needs.
KV_ARGS=()
while IFS= read -r -d '' arg; do
    KV_ARGS+=("$arg")
done < <("$PYTHON_BIN" - "$KV_RENDERED" <<'PY'
import json
import shlex
import sys

rendered = sys.argv[1]
try:
    args = shlex.split(rendered)
except ValueError as exc:
    raise SystemExit(f"invalid kvbmctl render output: {exc}")

expected_flags = ["--block-size", "--max-model-len", "--kv-transfer-config"]
if len(args) != len(expected_flags) * 2:
    raise SystemExit(f"unexpected kvbmctl render arg count: {len(args)}")

for idx, flag in enumerate(expected_flags):
    actual = args[idx * 2]
    if actual != flag:
        raise SystemExit(
            f"unexpected kvbmctl render flag at position {idx * 2}: {actual}"
        )

for idx in (1, 3):
    try:
        value = int(args[idx])
    except ValueError as exc:
        raise SystemExit(f"{args[idx - 1]} must be an integer") from exc
    if value <= 0:
        raise SystemExit(f"{args[idx - 1]} must be positive")

try:
    json.loads(args[5])
except json.JSONDecodeError as exc:
    raise SystemExit(f"--kv-transfer-config must be valid JSON: {exc}") from exc

for arg in args:
    sys.stdout.buffer.write(arg.encode("utf-8"))
    sys.stdout.buffer.write(b"\0")
PY
)

export CUDA_VISIBLE_DEVICES="${P2P_CUDA_VISIBLE_DEVICES:-0}"
export DYN_KVBM_CPU_CACHE_GB="$CACHE_GB"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export KVBM_SKIP_VLLM_VERSION_CHECK=${KVBM_SKIP_VLLM_VERSION_CHECK:-1}

# kvbm's _core.so links nixl 0.10.x without a baked rpath; put the venv's nixl
# libs on LD_LIBRARY_PATH so `import kvbm` resolves `libnixl.so` (inherited by
# vLLM's EngineCore subprocess, unlike figment KVBM_* vars).
NIXL_LIBS=${KVBM_NIXL_LIBS:-}
if [ -z "$NIXL_LIBS" ] && [ -n "${KVBM_VENV:-}" ] && [ -d "$KVBM_VENV" ]; then
    for cand in "$KVBM_VENV"/lib/python*/site-packages/.nixl_cu12.mesonpy.libs \
                "$KVBM_VENV"/lib/python*/site-packages/.nixl_cu13.mesonpy.libs; do
        [ -d "$cand" ] && NIXL_LIBS="$cand" && break
    done
fi
if [ -n "$NIXL_LIBS" ]; then
    export LD_LIBRARY_PATH="$NIXL_LIBS:$NIXL_LIBS/plugins:${LD_LIBRARY_PATH:-}"
    export NIXL_PLUGIN_DIR="$NIXL_LIBS/plugins"
fi

# --block-size / --max-model-len / --kv-transfer-config come from kvbmctl
# (${KV_ARGS[@]}); do not also pass --max-model-len here — vLLM rejects dupes.
echo "[p2p-smoke] P2P_ENFORCE_EAGER=$P2P_ENFORCE_EAGER" >&2
exec "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$P2P_MODEL" \
  --served-model-name "$P2P_MODEL" \
  --max-num-seqs 8 \
  --gpu-memory-utilization "$GMU" \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --port "$PORT" \
  "${ENFORCE_EAGER_ARGS[@]}" \
  "${KV_ARGS[@]}"
