#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$SCRIPT_DIR/hardware-profiles.sh"
kvbm_xdc_apply_hardware_profile model-only

WORKTREE=${WORKTREE:-/workspace}
NAMESPACE=${NAMESPACE:-dynamo}
HTTP_PORT=${HTTP_PORT:-8000}
ROUTER_MODE=${ROUTER_MODE:-kv}
ROUTER_RESET_STATES=${ROUTER_RESET_STATES:-1}
ENFORCE_DISAGG=${ENFORCE_DISAGG:-1}
ROUTER_MIN_INITIAL_WORKERS=${ROUTER_MIN_INITIAL_WORKERS:-0}
VENV=${VENV:-/opt/dynamo/venv}

: "${MODEL:?MODEL must be set by KVBM_HARDWARE_PROFILE or env override}"

export PATH="$VENV/bin:/usr/local/cargo/bin:/usr/local/cuda/bin:$PATH"
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
for lib_dir in "$WORKTREE/.image-target/debug/deps" "$WORKTREE/.image-target-kvbm/debug/deps" /usr/local/lib/python*/site-packages/nixl_cu*.libs; do
  if [[ -d "$lib_dir" ]]; then
    export LD_LIBRARY_PATH="$lib_dir:${LD_LIBRARY_PATH:-}"
  fi
done
if [[ -d "$WORKTREE/lib/bindings/python/src" ]]; then
  export PYTHONPATH="$WORKTREE/lib/bindings/python/src:${PYTHONPATH:-}"
fi
if [[ -d "$WORKTREE/lib/bindings/kvbm/python" ]]; then
  export PYTHONPATH="$WORKTREE/lib/bindings/kvbm/python:${PYTHONPATH:-}"
fi
if [[ -d "$WORKTREE/components/src" ]]; then
  export PYTHONPATH="$WORKTREE/components/src:${PYTHONPATH:-}"
fi

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
exec "$VENV/bin/python" -m dynamo.frontend "${args[@]}"
