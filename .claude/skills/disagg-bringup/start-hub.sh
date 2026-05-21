#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# Start kvbm_hub with the disagg (conditional-disagg) prefill dispatcher enabled.
#
# Thin wrapper over the reusable `kvbm-hub-bringup/start-hub.sh`: the hub is the
# runtime source of truth (serves `GET /v1/config`, validates registrants
# against its primary block_size/max_seq_len/layout), and the disagg launchers
# render their `--kv-transfer-config` from it via kvbmctl. This wrapper pins the
# disagg feature set + the CD prefill dispatcher (`--prefill-vllm-url/-model`,
# without which the hub queues remote-prefill requests but never pushes them and
# decode hangs at the lifecycle watchdog) and derives sizing from the hardware
# profile.
#
# Usage:
#   bash start-hub.sh <log_path>            # FOREGROUND; background from caller
#
# Env vars honored (with defaults):
#   KVBM_REPO        (default: worktree root inferred from this script's path)
#   KVBM_HUB_MODEL   (default: selected by KVBM_HARDWARE_PROFILE)
#   KVBM_HARDWARE_PROFILE (default: spark-gb10; also h100-a100, custom)
#   KVBM_HUB_FEATURES (default: disagg — dep-expands to disagg+p2p; set to
#                      "disagg,indexer" etc. to co-enable more)
#   KVBM_HUB_PREFILL_URL (default: http://127.0.0.1:8000)
#   KVBM_HUB_BLOCK_SIZE / KVBM_HUB_MAX_SEQ_LEN / KVBM_HUB_G2_MEMORY_GIB (sizing)
#   KVBM_HUB_DISCOVERY_PORT / KVBM_HUB_CONTROL_PORT / KVBM_HUB_VELO_PORT
set -eu

LOG=${1:?"usage: $0 <log_path>"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=${KVBM_REPO:-"$(cd "$SCRIPT_DIR/../../.." && pwd)"}
HUB_BRINGUP="$REPO/.claude/skills/kvbm-hub-bringup"

if [ -n "${KVBM_HUB_MODEL:-}" ]; then
    MODEL=$KVBM_HUB_MODEL
else
    . "$SCRIPT_DIR/hardware-profiles.sh"
    kvbm_apply_disagg_bringup_profile
    MODEL=$KVBM_MODEL
fi

# Delegate to the reusable launcher. It builds kvbm_hub + kvbmctl, passes the
# now-required --block-size/--max-seq-len/--g2 flags, dep-expands disagg→p2p,
# and wires the CD prefill dispatcher from KVBM_HUB_PREFILL_VLLM_{URL,MODEL}.
export KVBM_REPO="$REPO"
export KVBM_HUB_FEATURES=${KVBM_HUB_FEATURES:-disagg}
export KVBM_HUB_BLOCK_SIZE=${KVBM_HUB_BLOCK_SIZE:-16}
export KVBM_HUB_MAX_SEQ_LEN=${KVBM_HUB_MAX_SEQ_LEN:-1024}
# block_layout is hub-authoritative now (the launchers render it from the hub,
# not from an injected env var). Route the smoke's KVBM_BLOCK_LAYOUT knob to the
# hub's --layout so e.g. KVBM_BLOCK_LAYOUT=universal actually takes effect (the
# connectors then render + register universal, and verify-block-layout matches).
export KVBM_HUB_LAYOUT=${KVBM_HUB_LAYOUT:-${KVBM_BLOCK_LAYOUT:-operational}}
export KVBM_HUB_G2_MEMORY_GIB=${KVBM_HUB_G2_MEMORY_GIB:-2}
export KVBM_HUB_PREFILL_VLLM_URL=${KVBM_HUB_PREFILL_VLLM_URL:-${KVBM_HUB_PREFILL_URL:-http://127.0.0.1:8000}}
export KVBM_HUB_PREFILL_VLLM_MODEL=${KVBM_HUB_PREFILL_VLLM_MODEL:-$MODEL}

exec bash "$HUB_BRINGUP/start-hub.sh" "$LOG"
