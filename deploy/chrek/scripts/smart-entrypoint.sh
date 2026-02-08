#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Smart entrypoint wrapper for CRIU checkpoint/restore
# Automatically detects checkpoints and falls back to cold start if not found
#
# Behavior:
# 1. If DYN_CHECKPOINT_LOCATION (or DYN_CHECKPOINT_HASH) is set and checkpoint exists -> restore
# 2. Otherwise -> execute provided command (cold start)

set -e

# Enable debug output if DEBUG=1
if [ "${DEBUG:-0}" = "1" ]; then
  set -x
fi

# Configuration from environment
CHECKPOINT_HASH="${DYN_CHECKPOINT_HASH:-}"

# Prefer DYN_CHECKPOINT_LOCATION (full path). Fall back to constructing from HASH for backward compat.
if [ -n "${DYN_CHECKPOINT_LOCATION:-}" ]; then
  CHECKPOINT_LOCATION="$DYN_CHECKPOINT_LOCATION"
elif [ -n "$CHECKPOINT_HASH" ]; then
  CHECKPOINT_LOCATION="/checkpoints/$CHECKPOINT_HASH"
else
  CHECKPOINT_LOCATION=""
fi

# Log function for consistent output
log() {
  echo "[smart-entrypoint] $*" >&2
}

# Check if a checkpoint exists and is ready for immediate restore
has_ready_checkpoint() {
  # If checkpoint location is not set, no immediate checkpoint
  if [ -z "$CHECKPOINT_LOCATION" ]; then
    log "No checkpoint location resolved (DYN_CHECKPOINT_LOCATION and DYN_CHECKPOINT_HASH both unset)"
    return 1
  fi

  # Check if checkpoint directory exists
  if [ ! -d "$CHECKPOINT_LOCATION" ]; then
    log "Checkpoint directory not found: $CHECKPOINT_LOCATION"
    return 1
  fi

  # Check for checkpoint.done marker which is written LAST in the checkpoint process
  # This is more reliable than inventory.img (created by CRIU) or rootfs-diff.tar (may be mid-write)
  # Order: metadata.json -> CRIU dump (*.img) -> rootfs-diff.tar -> checkpoint.done
  DONE_MARKER="$CHECKPOINT_LOCATION/checkpoint.done"
  if [ ! -f "$DONE_MARKER" ]; then
    log "Checkpoint incomplete - checkpoint.done not found in: $CHECKPOINT_LOCATION"
    log "Checkpoint may still be in progress..."
    return 1
  fi

  log "Checkpoint found at $CHECKPOINT_LOCATION (checkpoint.done marker present)"
  return 0
}

# Main logic
if has_ready_checkpoint; then
  log "=========================================="
  log "CHECKPOINT RESTORE MODE"
  log "=========================================="
  log "Checkpoint: ${CHECKPOINT_HASH:-unknown}"
  log "Location: $CHECKPOINT_LOCATION"
  log "Invoking restore-entrypoint..."
  log "=========================================="

  # Execute restore-entrypoint
  # Any args passed to this script are forwarded (though restore-entrypoint ignores them)
  exec /restore-entrypoint "$@"
else
  log "=========================================="
  log "COLD START MODE"
  log "=========================================="

  # No checkpoint found - fall back to cold start
  if [ $# -eq 0 ]; then
    # No args provided - this is likely an error
    log "ERROR: No checkpoint to restore and no command provided"
    log "Set DYN_CHECKPOINT_HASH to restore a checkpoint, or provide a command to run"
    exit 1
  fi

  log "No checkpoint to restore"
  log "Executing command: $*"
  log "=========================================="

  # Execute the provided command
  exec "$@"
fi

