#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Hub launcher for the remote-search uniform-TP=2 scenario.
# Thin wrapper over kvbm-hub-bringup/start-hub.sh:
#   - features = indexer,p2p   (indexer discovers holders; p2p does the pull)
#   - leader.remote_search.enabled=true seeded into hub base_config so every
#     connector that registers gets remote-search wired without per-launcher
#     repetition (same pattern as .claude/skills/remote-search-smoke/start-hub.sh).
#
# Usage:
#   bash start-hub-remote-search.sh <log_path>   # FOREGROUND; background from caller
set -eu

LOG=${1:?"usage: $0 <log_path>"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/env.sh"

export KVBM_HUB_FEATURES=indexer,p2p
export KVBM_HUB_KVBM=${KVBM_HUB_KVBM:-"leader.tokio.worker_threads=2
worker.tokio.worker_threads=2
leader.control.metrics=true
leader.remote_search.enabled=true
worker.nixl.backends.UCX={}
worker.nixl.backends.POSIX={}"}

exec bash "$KVBM_REPO/.claude/skills/kvbm-hub-bringup/start-hub.sh" "$LOG"
