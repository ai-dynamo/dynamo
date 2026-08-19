#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"
trap dynamo_exit_trap EXIT

MODEL="${DYN_MODEL:-Qwen/Qwen3-0.6B}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
WORKER_SYSTEM_PORT="${DYN_WORKER_SYSTEM_PORT:-8081}"

export DYN_DISCOVERY_BACKEND="${DYN_DISCOVERY_BACKEND:-file}"
export DYN_EVENT_PLANE="${DYN_EVENT_PLANE:-zmq}"
export DYN_REQUEST_PLANE="${DYN_REQUEST_PLANE:-tcp}"
export DYN_MODEL="$MODEL"

print_launch_banner --max-tokens 1 \
    "Launching Dynamo with bespoke worker orchestration" \
    "hello-world" \
    "$HTTP_PORT" \
    "Backend: dyn://my.endpoint.generate" \
    "Hello + World: concurrent, inline" \
    "Merge: inline"

python3 -m dynamo.frontend \
    --http-port "$HTTP_PORT" \
    --model-path "$MODEL" \
    --model-name hello-world &

DYN_SYSTEM_PORT="$WORKER_SYSTEM_PORT" \
python3 -m examples.custom_backend.workflow_hello_world.dynamo_bespoke.worker &

wait_any_exit
