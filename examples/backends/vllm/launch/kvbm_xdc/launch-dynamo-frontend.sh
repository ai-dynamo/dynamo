#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$SCRIPT_DIR/hardware-profiles.sh"
. "$SCRIPT_DIR/common.sh"
kvbm_xdc_apply_hardware_profile model-only

WORKTREE=${WORKTREE:-/workspace}
NAMESPACE=${NAMESPACE:-dynamo}
HTTP_PORT=${HTTP_PORT:-8000}
ROUTER_MODE=${ROUTER_MODE:-kv}
ROUTER_RESET_STATES=${ROUTER_RESET_STATES:-1}
ENFORCE_DISAGG=${ENFORCE_DISAGG:-1}
ROUTER_MIN_INITIAL_WORKERS=${ROUTER_MIN_INITIAL_WORKERS:-0}
VENV=${VENV:-}
SOURCE_KVBM_RUNTIME_ENV=${SOURCE_KVBM_RUNTIME_ENV:-0}
PYTHON_BIN=${PYTHON_BIN:-}

: "${MODEL:?MODEL must be set by KVBM_HARDWARE_PROFILE or env override}"

kvbm_xdc_prepare_runtime

args=(
  --namespace "$NAMESPACE"
  --http-port "$HTTP_PORT"
  --router-mode "$ROUTER_MODE"
  --router-min-initial-workers "$ROUTER_MIN_INITIAL_WORKERS"
)

if [[ "$ROUTER_RESET_STATES" == "1" ]]; then
  args+=(--router-reset-states)
fi
if [[ "$ENFORCE_DISAGG" == "1" ]]; then
  args+=(--enforce-disagg)
fi

echo "[frontend] model=$MODEL namespace=$NAMESPACE http_port=$HTTP_PORT router=$ROUTER_MODE enforce_disagg=$ENFORCE_DISAGG"
exec "$PYTHON_BIN" -m dynamo.frontend "${args[@]}"
