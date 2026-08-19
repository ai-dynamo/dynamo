#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"
trap dynamo_exit_trap EXIT

MODEL="${DYN_MODEL:-Qwen/Qwen3-0.6B}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
HELLO_SYSTEM_PORT="${DYN_HELLO_SYSTEM_PORT:-8081}"
WORLD_SYSTEM_PORT="${DYN_WORLD_SYSTEM_PORT:-8082}"
MERGE_SYSTEM_PORT="${DYN_MERGE_SYSTEM_PORT:-8083}"
ORCHESTRATOR_SYSTEM_PORT="${DYN_ORCHESTRATOR_SYSTEM_PORT:-8084}"

export DYN_DISCOVERY_BACKEND="${DYN_DISCOVERY_BACKEND:-file}"
export DYN_EVENT_PLANE="${DYN_EVENT_PLANE:-zmq}"
export DYN_REQUEST_PLANE="${DYN_REQUEST_PLANE:-tcp}"

print_launch_banner --max-tokens 1 \
    "Launching Dynamo with a dedicated orchestrator worker" \
    "hello-world" \
    "$HTTP_PORT" \
    "Orchestrator: dyn://workflow-hello-world.orchestrator.generate" \
    "Hello + World: concurrent, remote" \
    "Merge: remote"

python3 -m dynamo.frontend \
    --http-port "$HTTP_PORT" &

DYN_SYSTEM_PORT="$HELLO_SYSTEM_PORT" \
python3 -m examples.custom_backend.workflow_hello_world.dynamo.worker hello &

DYN_SYSTEM_PORT="$WORLD_SYSTEM_PORT" \
python3 -m examples.custom_backend.workflow_hello_world.dynamo.worker world &

DYN_SYSTEM_PORT="$MERGE_SYSTEM_PORT" \
python3 -m examples.custom_backend.workflow_hello_world.dynamo.worker merge &

DYN_MODEL="$MODEL" \
DYN_SYSTEM_PORT="$ORCHESTRATOR_SYSTEM_PORT" \
python3 -m examples.custom_backend.workflow_hello_world.dynamo_orchestrator.worker &

wait_any_exit
