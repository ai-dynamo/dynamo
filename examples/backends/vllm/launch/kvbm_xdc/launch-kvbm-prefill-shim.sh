#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$SCRIPT_DIR/hardware-profiles.sh"
. "$SCRIPT_DIR/common.sh"
kvbm_xdc_apply_hardware_profile model-only

WORKTREE=${WORKTREE:-/workspace}
VENV=${VENV:-}
SOURCE_KVBM_RUNTIME_ENV=${SOURCE_KVBM_RUNTIME_ENV:-0}
KVBM_HUB_PREFILL_ENDPOINT=${KVBM_HUB_PREFILL_ENDPOINT:?KVBM_HUB_PREFILL_ENDPOINT is required}
KVBM_HUB_PREFILL_SHIM_HOST=${KVBM_HUB_PREFILL_SHIM_HOST:-127.0.0.1}
KVBM_HUB_PREFILL_SHIM_PORT=${KVBM_HUB_PREFILL_SHIM_PORT:-8001}
PYTHON_BIN=${PYTHON_BIN:-}

: "${MODEL:?MODEL must be set by KVBM_HARDWARE_PROFILE or env override}"

kvbm_xdc_prepare_runtime

echo "[prefill-shim] model=$MODEL endpoint=$KVBM_HUB_PREFILL_ENDPOINT port=$KVBM_HUB_PREFILL_SHIM_PORT"
exec "$PYTHON_BIN" "$SCRIPT_DIR/kvbm-prefill-completions-shim.py"
