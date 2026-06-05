#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Build kvbm_hub + kvbmctl in debug. Idempotent — cargo handles incremental.
# Honors PATH/CUDA_PATH the same way kvbm-hub-bringup/hub-lib.sh does, since
# kvbm-config pulls CUDA.
#
# Usage: bash build-deps.sh
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/env.sh"
. "$KVBM_REPO/.claude/skills/kvbm-hub-bringup/hub-lib.sh"

LOG=${1:-/tmp/qwen3-tp2-build.log}
echo "[build-deps] cargo build --bin kvbm_hub --bin kvbmctl (log: $LOG)"
if ! kvbm_hub_build "$KVBM_REPO" "$LOG"; then
  echo "[build-deps] BUILD FAILED — tail of $LOG:" >&2
  tail -n 50 "$LOG" >&2
  exit 1
fi

ls -lh "$KVBM_HUB_BIN" "$KVBM_KVBMCTL_BIN"
echo "[build-deps] OK"
