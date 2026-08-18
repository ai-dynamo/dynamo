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
source "$SCRIPT_DIR/../../../common/launch_utils.sh"
trap dynamo_exit_trap EXIT

HTTP_PORT="${BESPOKE_HTTP_PORT:-8000}"
HELLO_PORT="${BESPOKE_HELLO_PORT:-8101}"
WORLD_PORT="${BESPOKE_WORLD_PORT:-8102}"

export BESPOKE_HELLO_URL="http://127.0.0.1:${HELLO_PORT}/run"
export BESPOKE_WORLD_URL="http://127.0.0.1:${WORLD_PORT}/run"

print_launch_banner --max-tokens 1 \
    "Launching bespoke workflow orchestration" \
    "hello-world" \
    "$HTTP_PORT" \
    "Hello stage: http://127.0.0.1:${HELLO_PORT}/run" \
    "World stage: http://127.0.0.1:${WORLD_PORT}/run" \
    "Merge stage: inline"

python3 -m examples.custom_backend.workflow_hello_world.bespoke.app \
    hello --port "$HELLO_PORT" &
python3 -m examples.custom_backend.workflow_hello_world.bespoke.app \
    world --port "$WORLD_PORT" &
python3 -m examples.custom_backend.workflow_hello_world.bespoke.app \
    gateway --port "$HTTP_PORT" &

wait_any_exit
