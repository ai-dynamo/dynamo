#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"
trap dynamo_exit_trap EXIT

MODEL="${DYN_MODEL:-Qwen/Qwen3-0.6B}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
HELLO_SYSTEM_PORT="${DYN_HELLO_SYSTEM_PORT:-8081}"
WORLD_SYSTEM_PORT="${DYN_WORLD_SYSTEM_PORT:-8082}"
MERGE_SYSTEM_PORT="${DYN_MERGE_SYSTEM_PORT:-8083}"

export DYN_DISCOVERY_BACKEND="${DYN_DISCOVERY_BACKEND:-file}"
export DYN_EVENT_PLANE="${DYN_EVENT_PLANE:-zmq}"
export DYN_REQUEST_PLANE="${DYN_REQUEST_PLANE:-tcp}"

print_launch_banner --max-tokens 1 \
    "Launching declarative Dynamo workflow" \
    "hello-world" \
    "$HTTP_PORT" \
    "Hello stage: remote" \
    "World stage: remote" \
    "Merge stage: remote"

python3 -m dynamo.frontend \
    --http-port "$HTTP_PORT" \
    --model-path "$MODEL" \
    --model-name hello-world \
    --workflow-provider \
    examples.custom_backend.workflow_hello_world.dynamo.provider:provide_workflow &

DYN_SYSTEM_PORT="$HELLO_SYSTEM_PORT" \
python3 -m examples.custom_backend.workflow_hello_world.dynamo.worker hello &

DYN_SYSTEM_PORT="$WORLD_SYSTEM_PORT" \
python3 -m examples.custom_backend.workflow_hello_world.dynamo.worker world &

DYN_SYSTEM_PORT="$MERGE_SYSTEM_PORT" \
python3 -m examples.custom_backend.workflow_hello_world.dynamo.worker merge &

wait_any_exit
