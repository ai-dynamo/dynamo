#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Reusable kvbm_hub launcher. The hub is the runtime source of truth: it serves
# the feature set named by `--features`, publishes `GET /v1/config` (the
# aggregate the connector handshake + `kvbmctl` consume), and validates every
# registrant against its `primary` config (block_size / max_seq_len / layout).
#
# Sizing-agnostic by design: a caller (smoke) derives block_size / max_seq_len /
# g2 sizing from its own hardware profile and passes them via env. With nothing
# set it boots a sensible all-features demo hub.
#
# Builds BOTH bins (`kvbm_hub` + `kvbmctl`) so connector launchers can render
# their `--kv-transfer-config` from the live hub.
#
# Usage:
#   bash start-hub.sh <log_path>     # runs in FOREGROUND; background from caller
#
# Env vars (with defaults):
#   KVBM_REPO                 (default: worktree root inferred from this path)
#   KVBM_HUB_BIN              (default: $KVBM_REPO/target/debug/kvbm_hub)
#   KVBM_HUB_SKIP_BUILD       (default: 0 — rebuild incrementally, never stale)
#   KVBM_HUB_DISCOVERY_PORT   (default: 1337)
#   KVBM_HUB_CONTROL_PORT     (default: 8337)
#   KVBM_HUB_VELO_PORT        (default: 1338)
#   KVBM_HUB_FEATURES         (default: "" = all supported; csv subset of
#                              p2p,disagg,indexer — deps auto-added)
#   KVBM_HUB_BLOCK_SIZE       (default: 16; power of two in 16..=512)
#   KVBM_HUB_MAX_SEQ_LEN      (default: 1024; non-zero multiple of block size)
#   KVBM_HUB_G2_MEMORY_GIB    (default: 1 unless KVBM_HUB_G2_BLOCKS set; advisory)
#   KVBM_HUB_G2_BLOCKS        (default: unset; advisory; wins over memory if set)
#   KVBM_HUB_LAYOUT           (default: operational; or universal)
#   KVBM_HUB_HEARTBEAT_SECS   (default: 10)
#   KVBM_KV_INDEX_ADVERTISE_HOST (default: 127.0.0.1)
#   KVBM_HUB_PREFILL_VLLM_URL / KVBM_HUB_PREFILL_VLLM_MODEL
#                              (optional; enables the CD prefill dispatcher — both
#                               required together)
#   RUST_LOG                  (default: info,kvbm_hub=debug,kvbm_connector=debug)
set -eu

LOG=${1:?"usage: $0 <log_path>"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=${KVBM_REPO:-"$(cd "$SCRIPT_DIR/../../.." && pwd)"}
. "$SCRIPT_DIR/hub-lib.sh"

HUB=${KVBM_HUB_BIN:-$REPO/target/debug/kvbm_hub}
DISC_PORT=${KVBM_HUB_DISCOVERY_PORT:-1337}
CTRL_PORT=${KVBM_HUB_CONTROL_PORT:-8337}
VELO_PORT=${KVBM_HUB_VELO_PORT:-1338}
FEATURES=${KVBM_HUB_FEATURES:-}
BLOCK_SIZE=${KVBM_HUB_BLOCK_SIZE:-16}
MAX_SEQ_LEN=${KVBM_HUB_MAX_SEQ_LEN:-1024}
LAYOUT=${KVBM_HUB_LAYOUT:-operational}
HEARTBEAT=${KVBM_HUB_HEARTBEAT_SECS:-10}
ADVERTISE_HOST=${KVBM_KV_INDEX_ADVERTISE_HOST:-127.0.0.1}
RUST_LOG=${RUST_LOG:-info,kvbm_hub=debug,kvbm_connector=debug}

# Never run a stale hub (see disagg-bringup/start-hub.sh for the incident).
if [ -z "${KVBM_HUB_BIN:-}" ] && [ "${KVBM_HUB_SKIP_BUILD:-0}" != "1" ]; then
    if ! kvbm_hub_build "$REPO" "$LOG"; then
        echo "[kvbm-hub-bringup] kvbm_hub/kvbmctl build FAILED — see $LOG" >&2
        exit 1
    fi
fi

if [ ! -x "$HUB" ]; then
    echo "kvbm_hub binary missing at $HUB (build via: cargo build --bin kvbm_hub)" >&2
    exit 1
fi

# Assemble flags. `--features` is omitted when unset (hub default = all).
args=( --discovery-port "$DISC_PORT"
       --control-port "$CTRL_PORT"
       --velo-port "$VELO_PORT"
       --heartbeat-interval-secs "$HEARTBEAT"
       --block-size "$BLOCK_SIZE"
       --max-seq-len "$MAX_SEQ_LEN"
       --layout "$LAYOUT"
       --kv-index-advertise-host "$ADVERTISE_HOST" )
[ -n "$FEATURES" ] && args+=( --features "$FEATURES" )

# Advisory G2 sizing — at least one is required by the hub. Blocks win over GiB.
if [ -n "${KVBM_HUB_G2_BLOCKS:-}" ]; then
    args+=( --g2-block "$KVBM_HUB_G2_BLOCKS" )
else
    args+=( --g2-memory "${KVBM_HUB_G2_MEMORY_GIB:-1}" )
fi

# Optional CD prefill dispatcher (both env vars required together).
if [ -n "${KVBM_HUB_PREFILL_VLLM_URL:-}" ]; then
    args+=( --prefill-vllm-url "$KVBM_HUB_PREFILL_VLLM_URL"
            --prefill-vllm-model "${KVBM_HUB_PREFILL_VLLM_MODEL:?KVBM_HUB_PREFILL_VLLM_MODEL required with _URL}" )
fi

export RUST_LOG
exec "$HUB" "${args[@]}" >> "$LOG" 2>&1
