#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# CPU-only DeepInfer mock with the real Dynamo token frontend.

set -euo pipefail
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MODEL_PATH="${MODEL_PATH:-$(python3 "$SCRIPT_DIR/../prepare_model.py")}"
export DYN_NAMESPACE="${DYN_NAMESPACE:-deepinfer-mock}"
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

print_launch_banner "DeepInfer mock (CPU only)" "deepinfer-mock" "$DYN_HTTP_PORT"
python3 -m dynamo.frontend --router-mode round-robin &
DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT:-0}" python3 -m dynamo.deepinfer \
    --model "$MODEL_PATH" --served-model-name deepinfer-mock "$@" &
wait_any_exit
