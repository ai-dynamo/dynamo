# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Generic retry with exponential backoff.
# Usage: retry <command> [args...]
#
# Safe under `set -e`: the `if` conditional context prevents a failed
# command from triggering an immediate exit.
retry() {
  local max_attempts=3 wait_seconds=10 attempt=1

  while true; do
    if "$@"; then
      return 0
    fi
    echo "Command failed (attempt ${attempt}/${max_attempts}): $*" >&2

    if (( attempt >= max_attempts )); then
      echo "Command failed after ${max_attempts} attempts: $*" >&2
      return 1
    fi

    echo "Retrying in ${wait_seconds}s..."
    sleep "$wait_seconds"
    attempt=$((attempt + 1))
    wait_seconds=$(( wait_seconds * 2 > 120 ? 120 : wait_seconds * 2 ))
  done
}
