#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# remote-search-smoke hub launcher. Like kvindex-smoke's, but the hub serves
# BOTH the KV indexer (discovery: who holds which block) AND p2p (so instances
# register as hub-discoverable, remote-controllable transfer peers). Remote
# search needs both: the indexer to locate a holder, p2p to pull from it.
#
# Usage:
#   bash start-hub.sh <log_path>   # runs in FOREGROUND; background from caller
set -eu

LOG=${1:?"usage: $0 <log_path>"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=${KVBM_REPO:-"$(cd "$SCRIPT_DIR/../../.." && pwd)"}

# Sizing from the shared profile (Qwen3-0.6B / max_model_len / cpu cache).
. "$REPO/.claude/skills/disagg-bringup/hardware-profiles.sh"
kvbm_apply_disagg_bringup_profile

# Indexer (discovery) + p2p (transfer plane). Align hub primary + advisory G2.
export KVBM_HUB_FEATURES=indexer,p2p
export KVBM_HUB_BLOCK_SIZE=${KVBM_HUB_BLOCK_SIZE:-16}
export KVBM_HUB_MAX_SEQ_LEN=${KVBM_HUB_MAX_SEQ_LEN:-$KVBM_MAX_MODEL_LEN}
export KVBM_HUB_G2_MEMORY_GIB=${KVBM_HUB_G2_MEMORY_GIB:-$KVBM_CPU_CACHE_GB}
# Common KvbmConfig overrides seeded into the hub's base_config so all connectors
# inherit them. `leader.remote_search.enabled=true` turns on the hub-indexer
# remote-search path on every connector that registers.
export KVBM_HUB_KVBM=${KVBM_HUB_KVBM:-"$(cat <<'EOF'
leader.tokio.worker_threads=2
worker.tokio.worker_threads=2
leader.control.metrics=true
leader.remote_search.enabled=true
worker.nixl.backends.UCX={}
worker.nixl.backends.POSIX={}
EOF
)"}

exec bash "$REPO/.claude/skills/kvbm-hub-bringup/start-hub.sh" "$LOG"
