#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# kvindex-smoke hub launcher — a thin wrapper over the reusable
# `kvbm-hub-bringup/start-hub.sh`. It pins the hub to the KV indexer feature
# (no p2p, no CD prefill dispatcher: this smoke uses plain aggregated instances)
# and derives sizing from the shared hardware profile so the hub's primary
# (block_size / max_seq_len) and advisory G2 line up with what the instances
# launch with — kvbmctl then renders matching vLLM flags + connector config.
#
# Usage:
#   bash start-hub.sh <log_path>   # runs in FOREGROUND; background from caller
#
# Sizing/port env vars are honored by the reusable launcher; see
# .claude/skills/kvbm-hub-bringup/start-hub.sh for the full list.
set -eu

LOG=${1:?"usage: $0 <log_path>"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=${KVBM_REPO:-"$(cd "$SCRIPT_DIR/../../.." && pwd)"}

# Sizing from the shared profile (Qwen3-0.6B / max_model_len / cpu cache).
. "$REPO/.claude/skills/disagg-bringup/hardware-profiles.sh"
kvbm_apply_disagg_bringup_profile

# KV-index only; align the hub primary + advisory G2 with the model.
export KVBM_HUB_FEATURES=indexer
export KVBM_HUB_BLOCK_SIZE=${KVBM_HUB_BLOCK_SIZE:-16}
export KVBM_HUB_MAX_SEQ_LEN=${KVBM_HUB_MAX_SEQ_LEN:-$KVBM_MAX_MODEL_LEN}
export KVBM_HUB_G2_MEMORY_GIB=${KVBM_HUB_G2_MEMORY_GIB:-$KVBM_CPU_CACHE_GB}
# Common KvbmConfig overrides seeded into the hub's base_config so all connectors
# inherit them without each launcher repeating the same flags.
export KVBM_HUB_KVBM=${KVBM_HUB_KVBM:-"$(cat <<'EOF'
leader.tokio.worker_threads=2
worker.tokio.worker_threads=2
leader.control.metrics=true
worker.nixl.backends.UCX={}
worker.nixl.backends.POSIX={}
EOF
)"}

exec bash "$REPO/.claude/skills/kvbm-hub-bringup/start-hub.sh" "$LOG"
