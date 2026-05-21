#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# Start kvbm_hub with the KV indexer feature enabled — and nothing else.
#
# Unlike disagg-bringup/start-hub.sh, this launcher does NOT enable the
# conditional-disagg prefill dispatcher: the kvindex smoke uses plain
# aggregated instances with no P/D specialization. The hub's only job here is
# to bind the KV-index ZMQ ingest socket and serve /v1/features/kv-index.
#
# Usage:
#   bash start-hub.sh <log_path>   # runs in FOREGROUND; background from caller
#
# Env vars (with defaults):
#   KVBM_REPO                 (default: worktree root inferred from this path)
#   KVBM_HUB_BIN              (default: $KVBM_REPO/target/debug/kvbm_hub)
#   KVBM_HUB_SKIP_BUILD       (default: 0 — rebuild incrementally, never stale)
#   KVBM_HUB_DISCOVERY_PORT   (default: 1337)
#   KVBM_HUB_CONTROL_PORT     (default: 8337)
#   KVBM_HUB_VELO_PORT        (default: 1338)
#   KVBM_KV_INDEX_BLOCK_SIZE  (default: 16  — must match the workers' page size)
#   KVBM_KV_INDEX_MAX_SEQ_LEN (default: 1024 — multiple of block size)
#   KVBM_KV_INDEX_ADVERTISE_HOST (default: 127.0.0.1)
#   RUST_LOG                  (default: info,kvbm_hub=debug,kvbm_connector=debug)
set -eu

LOG=${1:?"usage: $0 <log_path>"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=${KVBM_REPO:-"$(cd "$SCRIPT_DIR/../../.." && pwd)"}
HUB=${KVBM_HUB_BIN:-$REPO/target/debug/kvbm_hub}
DISC_PORT=${KVBM_HUB_DISCOVERY_PORT:-1337}
CTRL_PORT=${KVBM_HUB_CONTROL_PORT:-8337}
VELO_PORT=${KVBM_HUB_VELO_PORT:-1338}
BLOCK_SIZE=${KVBM_KV_INDEX_BLOCK_SIZE:-16}
MAX_SEQ_LEN=${KVBM_KV_INDEX_MAX_SEQ_LEN:-1024}
ADVERTISE_HOST=${KVBM_KV_INDEX_ADVERTISE_HOST:-127.0.0.1}
RUST_LOG=${RUST_LOG:-info,kvbm_hub=debug,kvbm_connector=debug}

# Never run a stale hub (see disagg-bringup/start-hub.sh for the incident).
if [ -z "${KVBM_HUB_BIN:-}" ] && [ "${KVBM_HUB_SKIP_BUILD:-0}" != "1" ]; then
    if [ -d /usr/local/cuda/bin ] && [[ ":$PATH:" != *":/usr/local/cuda/bin:"* ]]; then
        export PATH="/usr/local/cuda/bin:$PATH"
    fi
    export CUDA_PATH=${CUDA_PATH:-/usr/local/cuda} CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
    echo "[start-hub] cargo build --bin kvbm_hub (incremental)…" > "$LOG"
    if ! ( cd "$REPO" && cargo build --bin kvbm_hub ) >> "$LOG" 2>&1; then
        echo "[start-hub] kvbm_hub build FAILED — see $LOG" >&2
        exit 1
    fi
fi

if [ ! -x "$HUB" ]; then
    echo "kvbm_hub binary missing at $HUB (build via: cargo build --bin kvbm_hub)" >&2
    exit 1
fi

export RUST_LOG
exec "$HUB" \
    --discovery-port "$DISC_PORT" \
    --control-port "$CTRL_PORT" \
    --velo-port "$VELO_PORT" \
    --heartbeat-interval-secs 10 \
    --kv-index-block-size "$BLOCK_SIZE" \
    --kv-index-max-seq-len "$MAX_SEQ_LEN" \
    --kv-index-advertise-host "$ADVERTISE_HOST" \
    >> "$LOG" 2>&1
