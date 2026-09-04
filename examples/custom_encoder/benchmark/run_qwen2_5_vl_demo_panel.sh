#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Start one demo server, wait for readiness, then open its labeled client shell.

set -euo pipefail

SIDE="${1:-}"
case "$SIDE" in
    control|dynamo-vllm) ;;
    *)
        echo >&2 "Usage: $0 {control|dynamo-vllm}"
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_SCRIPT="$SCRIPT_DIR/run_qwen2_5_vl_demo_server.sh"
SHELL_RC="$SCRIPT_DIR/demo_shell_rc.sh"
HTTP_PORT="${DYN_DEMO_HTTP_PORT:-8000}"
export DYN_DEMO_SIDE="$SIDE"
export DYN_DEMO_GPU_INDEX="${DYN_DEMO_GPU_INDEX:-0}"
export DYN_DEMO_HTTP_PORT="$HTTP_PORT"
export DYN_DEMO_URL="${DYN_DEMO_URL:-http://127.0.0.1:$HTTP_PORT}"
export DYN_DEMO_COORD_SESSION="${DYN_DEMO_COORD_SESSION:-matched-h100-demo}"

PANEL_ROOT="${DYN_DEMO_PANEL_ROOT:-/dynamo-tmp/logs/$(date +%m-%d)/qwen25-custom-encoder-live-demo/panels}"
SERVER_LOG="${DYN_DEMO_SERVER_LOG:-$PANEL_ROOT/$SIDE-$(hostname)-server.log}"
SERVER_PID=""

cleanup() {
    if [[ -n "$SERVER_PID" ]]; then
        kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT HUP TERM

for command in curl jq nvidia-smi setsid; do
    command -v "$command" >/dev/null
done
for path in "$SERVER_SCRIPT" "$SHELL_RC"; do
    test -f "$path"
done

mkdir -p "$PANEL_ROOT"
setsid bash "$SERVER_SCRIPT" "$SIDE" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

printf 'Starting %s server on %s; log: %s\n' "$SIDE" "$(hostname)" "$SERVER_LOG"
deadline=$((SECONDS + ${DYN_DEMO_READY_TIMEOUT_SECONDS:-300}))
served_model=""
while [[ -z "$served_model" ]]; do
    models_json="$(curl -fsS "$DYN_DEMO_URL/v1/models" 2>/dev/null || true)"
    if [[ -n "$models_json" ]]; then
        served_model="$(
            jq -er '.data[0].id | select(type == "string" and length > 0)' \
                <<<"$models_json" 2>/dev/null || true
        )"
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo >&2 "Demo server exited before becoming ready"
        tail -n 100 "$SERVER_LOG" >&2
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        echo >&2 "Timed out waiting for $DYN_DEMO_URL/v1/models"
        tail -n 100 "$SERVER_LOG" >&2
        exit 1
    fi
    sleep 2
done

printf 'Server ready: %s\n' "$served_model"

# Keep this parent alive so its EXIT trap tears down the detached server if the
# user exits the panel. The interactive child receives the terminal directly.
bash --noprofile --rcfile "$SHELL_RC" -i
