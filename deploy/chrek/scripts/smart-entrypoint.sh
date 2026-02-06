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
# 1. If DYN_CHECKPOINT_HASH is set and checkpoint exists -> restore
# 2. If WAIT_FOR_CHECKPOINT=1 -> wait for checkpoint (restore-entrypoint handles this)
# 3. Otherwise -> execute provided command (cold start)

set -e

# Enable debug output if DEBUG=1
if [ "${DEBUG:-0}" = "1" ]; then
  set -x
fi

# Configuration from environment
CHECKPOINT_PATH="${DYN_CHECKPOINT_PATH:-/checkpoints}"
CHECKPOINT_HASH="${DYN_CHECKPOINT_HASH:-}"
WAIT_FOR_CHECKPOINT="${WAIT_FOR_CHECKPOINT:-0}"

# Log function for consistent output
log() {
  echo "[smart-entrypoint] $*" >&2
}

# Check if a checkpoint exists and should be restored
should_restore_checkpoint() {
  # If WAIT_FOR_CHECKPOINT is set, always use restore-entrypoint
  # (it will wait for the checkpoint to appear)
  if [ "$WAIT_FOR_CHECKPOINT" = "1" ]; then
    log "WAIT_FOR_CHECKPOINT=1, will wait for checkpoint via restore-entrypoint"
    return 0
  fi

  # If checkpoint hash is not set, no restore
  if [ -z "$CHECKPOINT_HASH" ]; then
    log "DYN_CHECKPOINT_HASH not set, no checkpoint to restore"
    return 1
  fi

  # Check if checkpoint directory exists
  CHECKPOINT_DIR="$CHECKPOINT_PATH/$CHECKPOINT_HASH"
  if [ ! -d "$CHECKPOINT_DIR" ]; then
    log "Checkpoint directory not found: $CHECKPOINT_DIR"
    return 1
  fi

  # Check for checkpoint.done marker which is written LAST in the checkpoint process
  # This is more reliable than inventory.img (created by CRIU) or rootfs-diff.tar (may be mid-write)
  # Order: metadata.json -> CRIU dump (*.img) -> rootfs-diff.tar -> checkpoint.done
  DONE_MARKER="$CHECKPOINT_DIR/checkpoint.done"
  if [ ! -f "$DONE_MARKER" ]; then
    log "Checkpoint incomplete - checkpoint.done not found in: $CHECKPOINT_DIR"
    log "Checkpoint may still be in progress..."
    return 1
  fi

  log "Checkpoint found: $CHECKPOINT_HASH (checkpoint.done marker present)"
  return 0
}

# Main logic
if should_restore_checkpoint; then
  log "=========================================="
  log "CHECKPOINT RESTORE MODE"
  log "=========================================="
  log "Checkpoint: $CHECKPOINT_HASH"
  log "Location: $CHECKPOINT_PATH/$CHECKPOINT_HASH"
  log "Invoking restore-entrypoint..."
  log "=========================================="

  # Execute restore-entrypoint
  # Any args passed to this script are forwarded (though restore-entrypoint ignores them)
  exec /restore-entrypoint "$@"
else
  log "=========================================="
  log "COLD START MODE"
  log "=========================================="

  # No checkpoint found or not requested - fall back to cold start
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

