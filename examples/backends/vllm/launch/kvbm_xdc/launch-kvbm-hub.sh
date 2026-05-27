#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$SCRIPT_DIR/hardware-profiles.sh"
if [ -n "${KVBM_HUB_MODEL:-}" ] && [ -z "${MODEL:-}" ]; then
  MODEL=$KVBM_HUB_MODEL
  export MODEL
fi
kvbm_xdc_apply_hardware_profile model-only

LOG=${1:?"usage: $0 <log_path>"}
WORKTREE=${WORKTREE:-${KVBM_REPO:-/workspace}}

if [ -n "${KVBM_HUB_MODEL:-}" ]; then
  HUB_MODEL=$KVBM_HUB_MODEL
else
  : "${MODEL:?MODEL must be set by KVBM_HARDWARE_PROFILE or env override}"
  HUB_MODEL=${SERVED_MODEL_NAME:-$MODEL}
fi

find_hub_bin() {
  local candidate
  for candidate in \
    "${KVBM_HUB_BIN:-}" \
    /usr/local/bin/kvbm_hub \
    "$WORKTREE/target/release/kvbm_hub" \
    "$WORKTREE/target/debug/kvbm_hub"; do
    [ -n "$candidate" ] || continue
    if [ -x "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

HUB_BIN=$(find_hub_bin || true)
KVBM_HUB_BUILD=${KVBM_HUB_BUILD:-auto}
if [ -z "$HUB_BIN" ] && { [ "$KVBM_HUB_BUILD" = "1" ] || [ "$KVBM_HUB_BUILD" = "auto" ]; }; then
  if command -v cargo >/dev/null 2>&1; then
    echo "[kvbm-xdc] building kvbm_hub in $WORKTREE" > "$LOG"
    (
      cd "$WORKTREE"
      cargo build --bin kvbm_hub
    ) >> "$LOG" 2>&1
    HUB_BIN=$(find_hub_bin || true)
  fi
fi

if [ -z "$HUB_BIN" ]; then
  echo "kvbm_hub binary not found. Set KVBM_HUB_BIN or build with KVBM_HUB_BUILD=1." >&2
  exit 2
fi

KVBM_HUB_FEATURES=${KVBM_HUB_FEATURES:-disagg}
KVBM_HUB_DISCOVERY_PORT=${KVBM_HUB_DISCOVERY_PORT:-1337}
KVBM_HUB_CONTROL_PORT=${KVBM_HUB_CONTROL_PORT:-8337}
KVBM_HUB_VELO_PORT=${KVBM_HUB_VELO_PORT:-1338}
KVBM_HUB_BLOCK_SIZE=${KVBM_HUB_BLOCK_SIZE:-16}
KVBM_HUB_MAX_SEQ_LEN=${KVBM_HUB_MAX_SEQ_LEN:-${MAX_MODEL_LEN:-1024}}
KVBM_HUB_LAYOUT=${KVBM_HUB_LAYOUT:-${KVBM_BLOCK_LAYOUT:-operational}}
KVBM_HUB_G2_MEMORY_GIB=${KVBM_HUB_G2_MEMORY_GIB:-${CPU_CACHE_GB:-2}}
KVBM_HUB_HEARTBEAT_SECS=${KVBM_HUB_HEARTBEAT_SECS:-10}
KVBM_KV_INDEX_ADVERTISE_HOST=${KVBM_KV_INDEX_ADVERTISE_HOST:-127.0.0.1}
KVBM_HUB_PREFILL_VLLM_URL=${KVBM_HUB_PREFILL_VLLM_URL:-${KVBM_HUB_PREFILL_URL:-}}
KVBM_HUB_PREFILL_VLLM_MODEL=${KVBM_HUB_PREFILL_VLLM_MODEL:-$HUB_MODEL}
KVBM_ONBOARD_MODE=${KVBM_ONBOARD_MODE:-inter}
RUST_LOG=${RUST_LOG:-info,kvbm_hub=debug,kvbm_connector=debug}

case "$KVBM_HUB_LAYOUT" in
  operational|universal) ;;
  *) echo "KVBM_HUB_LAYOUT must be operational or universal, got $KVBM_HUB_LAYOUT" >&2; exit 2 ;;
esac
case "$KVBM_ONBOARD_MODE" in
  inter|intra) ;;
  *) echo "KVBM_ONBOARD_MODE must be inter or intra, got $KVBM_ONBOARD_MODE" >&2; exit 2 ;;
esac

args=(
  --discovery-port "$KVBM_HUB_DISCOVERY_PORT"
  --control-port "$KVBM_HUB_CONTROL_PORT"
  --velo-port "$KVBM_HUB_VELO_PORT"
  --heartbeat-interval-secs "$KVBM_HUB_HEARTBEAT_SECS"
  --block-size "$KVBM_HUB_BLOCK_SIZE"
  --max-seq-len "$KVBM_HUB_MAX_SEQ_LEN"
  --layout "$KVBM_HUB_LAYOUT"
  --kv-index-advertise-host "$KVBM_KV_INDEX_ADVERTISE_HOST"
  --features "$KVBM_HUB_FEATURES"
)

if [ -n "${KVBM_HUB_G2_BLOCKS:-}" ]; then
  args+=(--g2-block "$KVBM_HUB_G2_BLOCKS")
else
  args+=(--g2-memory "$KVBM_HUB_G2_MEMORY_GIB")
fi

if [ -n "$KVBM_HUB_PREFILL_VLLM_URL" ]; then
  args+=(
    --prefill-vllm-url "$KVBM_HUB_PREFILL_VLLM_URL"
    --prefill-vllm-model "$KVBM_HUB_PREFILL_VLLM_MODEL"
  )
fi

KVBM_HUB_KVBM=${KVBM_HUB_KVBM:-"leader.tokio.worker_threads=2
worker.tokio.worker_threads=2
leader.control.metrics=true
leader.control.dev=true
leader.onboard.mode=$KVBM_ONBOARD_MODE
worker.nixl.backends.UCX={}
worker.nixl.backends.POSIX={}"}

if [ -n "${KVBM_HUB_KVBM_CONFIG:-}" ]; then
  args+=(--kvbm-config "$KVBM_HUB_KVBM_CONFIG")
fi
if [ -n "$KVBM_HUB_KVBM" ]; then
  while IFS= read -r kv; do
    [ -n "$kv" ] && args+=(--kvbm "$kv")
  done <<< "$KVBM_HUB_KVBM"
fi

export RUST_LOG
exec "$HUB_BIN" "${args[@]}" >> "$LOG" 2>&1
