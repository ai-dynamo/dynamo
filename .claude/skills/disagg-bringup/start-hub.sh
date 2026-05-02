#!/bin/bash
# Start kvbm_hub with the conditional-disagg dispatcher enabled.
#
# Always passes `--prefill-vllm-url` and `--prefill-vllm-model` so the
# hub's CD prefill dispatcher worker is active — without those flags
# the hub queues remote-prefill requests but never pushes them to the
# prefill instance, and decode hangs at the lifecycle watchdog.
#
# Usage:
#   bash start-hub.sh <log_path>
#
# Env vars honored (with defaults):
#   KVBM_HUB_BIN     (default: $REPO/target/debug/kvbm_hub)
#   KVBM_HUB_MODEL   (default: Qwen/Qwen3-0.6B)
#   KVBM_HUB_PREFILL_URL (default: http://127.0.0.1:8000)
#   KVBM_HUB_DISCOVERY_PORT (default: 1337)
#   KVBM_HUB_CONTROL_PORT   (default: 8337)
#   KVBM_HUB_VELO_PORT      (default: 1338)
#   RUST_LOG                (default: info,kvbm_hub=debug,kvbm_connector=debug,kvbm_audit=info)
#
# The script runs the hub in the FOREGROUND.  Background it from the
# caller (e.g. `bash start-hub.sh hub.log &`).

set -eu

LOG=${1:?"usage: $0 <log_path>"}
REPO=${KVBM_REPO:-/home/ryan/repos/dynamo}
HUB=${KVBM_HUB_BIN:-$REPO/target/debug/kvbm_hub}
MODEL=${KVBM_HUB_MODEL:-Qwen/Qwen3-0.6B}
PREFILL_URL=${KVBM_HUB_PREFILL_URL:-http://127.0.0.1:8000}
DISC_PORT=${KVBM_HUB_DISCOVERY_PORT:-1337}
CTRL_PORT=${KVBM_HUB_CONTROL_PORT:-8337}
VELO_PORT=${KVBM_HUB_VELO_PORT:-1338}
RUST_LOG=${RUST_LOG:-info,kvbm_hub=debug,kvbm_connector=debug,kvbm_audit=info}

if [ ! -x "$HUB" ]; then
    echo "kvbm_hub binary missing at $HUB" >&2
    echo "build it via: cargo build --bin kvbm_hub" >&2
    exit 1
fi

export RUST_LOG
exec "$HUB" \
    --discovery-port "$DISC_PORT" \
    --control-port "$CTRL_PORT" \
    --velo-port "$VELO_PORT" \
    --heartbeat-interval-secs 10 \
    --prefill-vllm-url "$PREFILL_URL" \
    --prefill-vllm-model "$MODEL" \
    > "$LOG" 2>&1
