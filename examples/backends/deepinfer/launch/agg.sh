#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Native DeepInfer serving; add --mock for CPU-only synthetic inference.

set -euo pipefail
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MOCK=false
for arg in "$@"; do
    if [[ "$arg" == "--mock" ]]; then
        MOCK=true
    fi
done
if "$MOCK"; then
    MODEL_PATH="${MODEL_PATH:-$(python3 "$SCRIPT_DIR/../prepare_wrapper_model.py")}"
    SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepinfer-mock}"
else
    : "${MODEL_PATH:?Set MODEL_PATH to the local DeepInfer model checkpoint, or add --mock}"
    SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepinfer}"
fi
export DYN_NAMESPACE="${DYN_NAMESPACE:-deepinfer}"
export DYN_DISCOVERY_BACKEND=file
export DYN_REQUEST_PLANE=tcp
export DYN_RESPONSE_PLANE=tcp
export DYN_HTTP_PORT="${DYN_HTTP_PORT:-8000}"
export DYN_HTTP_HOST="${DYN_HTTP_HOST:-127.0.0.1}"
DISCOVERY_DIR="$(mktemp -d)"
export DYN_FILE_KV="$DISCOVERY_DIR"

cleanup() {
    local pids
    pids="$(jobs -pr)"
    if [[ -n "$pids" ]]; then
        kill $pids 2>/dev/null || true
        wait $pids 2>/dev/null || true
    fi
    rm -rf -- "$DISCOVERY_DIR"
}
trap cleanup EXIT

print_launch_banner "DeepInfer aggregated serving (mock=$MOCK)" "$SERVED_MODEL_NAME" "$DYN_HTTP_PORT"
python3 -m dynamo.frontend --router-mode round-robin &
DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT:-0}" python3 -m dynamo.deepinfer \
    --model "$MODEL_PATH" --served-model-name "$SERVED_MODEL_NAME" "$@" &
wait_any_exit
