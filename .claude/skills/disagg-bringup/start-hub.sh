#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# Start kvbm_hub with the conditional-disagg dispatcher enabled.
#
# Always passes `--prefill-vllm-url` and `--prefill-vllm-model` so the
# hub's CD prefill dispatcher worker is active — without those flags
# the hub queues remote-prefill requests but never pushes them to the
# prefill instance, and decode hangs at the lifecycle watchdog.
#
# Usage:
#   bash start-hub.sh <log_path>
#
# Env vars honored (with defaults):
#   KVBM_REPO        (default: worktree root inferred from this script's path)
#   KVBM_HUB_BIN     (default: $KVBM_REPO/target/debug/kvbm_hub)
#   KVBM_HUB_MODEL   (default: selected by KVBM_HARDWARE_PROFILE)
#   KVBM_HARDWARE_PROFILE (default: spark-gb10; also supports h100-a100, custom)
#   KVBM_HUB_PREFILL_URL (default: http://127.0.0.1:8000)
#   KVBM_HUB_DISCOVERY_PORT (default: 1337)
#   KVBM_HUB_CONTROL_PORT   (default: 8337)
#   KVBM_HUB_VELO_PORT      (default: 1338)
#   RUST_LOG                (default: info,kvbm_hub=debug,kvbm_connector=debug,kvbm_audit=info)
#
# Repo auto-detection: the script lives at
#   <repo>/.claude/skills/disagg-bringup/start-hub.sh
# so the default $KVBM_REPO is the directory four levels above this file.
# This makes the script work correctly when run from a git worktree without
# requiring the caller to set KVBM_REPO=/path/to/worktree explicitly.
#
# The script runs the hub in the FOREGROUND.  Background it from the
# caller (e.g. `bash start-hub.sh hub.log &`).

set -eu

LOG=${1:?"usage: $0 <log_path>"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DEFAULT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO=${KVBM_REPO:-$REPO_DEFAULT}
HUB=${KVBM_HUB_BIN:-$REPO/target/debug/kvbm_hub}
if [ -n "${KVBM_HUB_MODEL:-}" ]; then
    MODEL=$KVBM_HUB_MODEL
else
    . "$SCRIPT_DIR/hardware-profiles.sh"
    kvbm_apply_disagg_bringup_profile
    MODEL=$KVBM_MODEL
fi
PREFILL_URL=${KVBM_HUB_PREFILL_URL:-http://127.0.0.1:8000}
DISC_PORT=${KVBM_HUB_DISCOVERY_PORT:-1337}
CTRL_PORT=${KVBM_HUB_CONTROL_PORT:-8337}
VELO_PORT=${KVBM_HUB_VELO_PORT:-1338}
RUST_LOG=${RUST_LOG:-info,kvbm_hub=debug,kvbm_connector=debug,kvbm_audit=info}

# Never run a stale hub. 2026-05-19 incident: a May-15 debug binary ran
# against May-19 kvbm-hub source, so the CD registration handshake the fresh
# connector spoke didn't match the old hub → `conditional-disagg hub
# registration failed` → EngineCore died → smoke hung on its readiness loop.
# A `test -x $HUB` existence guard does NOT catch a stale binary; cargo does.
# cargo build is incremental: a near-noop when fresh, a rebuild when stale.
# Skip only when the caller pinned an explicit binary (KVBM_HUB_BIN) or opts
# out (KVBM_HUB_SKIP_BUILD=1). Build output prepends to the hub log.
if [ -z "${KVBM_HUB_BIN:-}" ] && [ "${KVBM_HUB_SKIP_BUILD:-0}" != "1" ]; then
    if [ -d /usr/local/cuda/bin ] && [[ ":$PATH:" != *":/usr/local/cuda/bin:"* ]]; then
        export PATH="/usr/local/cuda/bin:$PATH"
    fi
    export CUDA_PATH=${CUDA_PATH:-/usr/local/cuda} CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
    echo "[start-hub] cargo build --bin kvbm_hub (incremental; rebuilds iff stale)…" > "$LOG"
    if ! ( cd "$REPO" && cargo build --bin kvbm_hub ) >> "$LOG" 2>&1; then
        echo "[start-hub] kvbm_hub build FAILED — see $LOG" >&2
        exit 1
    fi
fi

if [ ! -x "$HUB" ]; then
    echo "kvbm_hub binary missing at $HUB" >&2
    echo "build it via: cargo build --bin kvbm_hub" >&2
    exit 1
fi

export RUST_LOG
exec "$HUB" \
    --discovery-port "$DISC_PORT" \
    --control-port "$CTRL_PORT" \
    --velo-port "$VELO_PORT" \
    --heartbeat-interval-secs 10 \
    --prefill-vllm-url "$PREFILL_URL" \
    --prefill-vllm-model "$MODEL" \
    >> "$LOG" 2>&1
