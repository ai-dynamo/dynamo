#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# Stream all KVBM disagg smoke logs concurrently, surface errors immediately.
#
# Usage: monitor-smoke.sh <experiment_dir> [smoke_log]
#
# Designed for Claude Code's Monitor tool — each matching line becomes a
# notification. The match alternation covers happy-path transitions AND failure
# signatures, per the Monitor tool's "silence is not success" guidance.
#
# Why this exists: an earlier R14 run silently crashed both vLLMs with
# ModuleNotFoundError on `import kvbm`, but the outer smoke log only contained
# "waiting for both vLLMs..." and never emitted a terminal-state line. A Monitor
# watching only the outer log timed out at 10 min with no signal. The actual
# failure was in prefill.log + decode.log — per-component logs the outer wrapper
# never reflects. This script tails ALL component logs and emits matching lines
# tagged with the source log name.
#
# Usage with Monitor:
#   Monitor({
#     command: 'ssh <host> bash skills/kvbm-disagg-smoke/scripts/monitor-smoke.sh <exp_dir>',
#     description: "smoke progress + failures",
#     timeout_ms: 900000
#   })

set -euo pipefail
EXP_DIR="${1:?usage: monitor-smoke.sh <experiment_dir> [smoke_log]}"
SMOKE_LOG="${2:-}"

LOGS=(
  "$EXP_DIR/prefill.log"
  "$EXP_DIR/decode.log"
  "$EXP_DIR/hub.log"
)
[ -n "$SMOKE_LOG" ] && [ -f "$SMOKE_LOG" ] && LOGS+=("$SMOKE_LOG")

# Wait up to 60s for at least one log to appear (vLLM creates them within ~5s)
deadline=$(($(date +%s) + 60))
while [ "$(date +%s)" -lt "$deadline" ]; do
  for f in "${LOGS[@]}"; do
    [ -f "$f" ] && break 2
  done
  sleep 1
done

# Broad alternation: happy-path transitions + failure signatures + audit-event
# discriminators. Add new modes as they are observed; a wider filter is better
# than narrower because Monitor needs to wake on EVERY terminal state, not just
# the success path.
PATTERN='Traceback|ModuleNotFoundError|ImportError|panicked|^FATAL|^ERROR|cudaError|cudaErrorNoKernelImageForDevice|core dumped|killed by signal|begin_remote_prefill called twice|Invalid transition|gnmt_passthrough_non_cd|ensure_started_async_onboard|ensure_started_zero_passthrough|policy_decision|prefill_cd_payload_(installed|drop)|worker_pull_chunk_start|worker_g2_to_g1_done|prefill_cleanup_pending_usaa|prefill_finalize_observer_watchdog|Validation|two-request-smoke (complete|failed)|smoke (PASS|FAIL)|R[0-9]+ (passed|failed)|out of memory|OOM'

# Concurrently tail all present logs, prefix lines with [logname], grep terminal/progress events.
{
  for log in "${LOGS[@]}"; do
    [ -f "$log" ] || continue
    name="$(basename "$log" .log)"
    tail -F -n +1 "$log" 2>/dev/null | sed -u "s|^|[$name] |" &
  done
  wait
} | grep -E --line-buffered "$PATTERN"
