#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Hub launcher for conditional-disagg scenarios (uniform TP=2 + asymmetric).
#
# Mirrors disagg-bringup/start-hub.sh but:
#   - Sourceable defaults come from this bundle's env.sh (32B / bs=64 / 128k).
#   - Hardware-profile and the spark-gb10 "shared GPU" assumption are NOT
#     applied; sizing is owned by env.sh.
#   - Prefill URL/model are arguments (orchestrator picks the right port).
#
# Usage:
#   PREFILL_PORT=8000 bash start-hub-disagg.sh <log_path>
set -eu

LOG=${1:?"usage: $0 <log_path>"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/env.sh"

PREFILL_PORT=${PREFILL_PORT:-8000}
export KVBM_HUB_FEATURES=${KVBM_HUB_FEATURES:-disagg}
export KVBM_HUB_PREFILL_VLLM_URL=${KVBM_HUB_PREFILL_VLLM_URL:-http://127.0.0.1:$PREFILL_PORT}
export KVBM_HUB_PREFILL_VLLM_MODEL=${KVBM_HUB_PREFILL_VLLM_MODEL:-$KVBM_MODEL}

# leader.onboard.mode is a smoke knob; default inter (the same default
# disagg-bringup/start-hub.sh uses).
KVBM_ONBOARD_MODE=${KVBM_ONBOARD_MODE:-inter}
case "$KVBM_ONBOARD_MODE" in
  inter|intra) ;;
  *) echo "KVBM_ONBOARD_MODE must be 'inter' or 'intra', got: '$KVBM_ONBOARD_MODE'" >&2; exit 1 ;;
esac

export KVBM_HUB_KVBM=${KVBM_HUB_KVBM:-"leader.tokio.worker_threads=2
worker.tokio.worker_threads=2
leader.control.metrics=true
leader.control.dev=true
leader.onboard.mode=$KVBM_ONBOARD_MODE
worker.nixl.backends.UCX={}
worker.nixl.backends.POSIX={}"}

exec bash "$KVBM_REPO/.claude/skills/kvbm-hub-bringup/start-hub.sh" "$LOG"
