#!/usr/bin/env bash
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

# Start Slime against a fixed set of SGLang engines managed by Dynamo.

set -euo pipefail

: "${SLIME_HOME:?Set SLIME_HOME to the Slime checkout or installed source tree}"
: "${DYNAMO_ENGINE_ADDRS:=slime-sglang-engine-0:30000 slime-sglang-engine-1:30000}"

read -r -a ENGINE_ADDRS <<<"${DYNAMO_ENGINE_ADDRS}"

exec python3 "${SLIME_HOME}/train.py" \
    --rollout-external-engine-addrs "${ENGINE_ADDRS[@]}" \
    --rollout-function-path slime.rollout.sglang_rollout.generate_rollout \
    --custom-generate-function-path \
    slime.rollout.sglang_streaming_rollout.generate_streaming \
    --sglang-incremental-streaming-output \
    "$@"
