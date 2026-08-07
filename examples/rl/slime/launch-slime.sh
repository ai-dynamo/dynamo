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

# Start Slime against Dynamo's shared SGLang rollout frontend.

set -euo pipefail

: "${SLIME_HOME:?Set SLIME_HOME to the Slime checkout or installed source tree}"
: "${DYNAMO_ROLLOUT_URL:?Set DYNAMO_ROLLOUT_URL to the Dynamo frontend origin}"

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
export PYTHONPATH="${SCRIPT_DIR}:${SLIME_HOME}${PYTHONPATH:+:${PYTHONPATH}}"

exec python3 "${SLIME_HOME}/train.py" \
    --rollout-external \
    --rollout-external-rollout-url "${DYNAMO_ROLLOUT_URL}" \
    --rollout-external-dynamic-discovery-path \
    dynamo_discovery.discover_engine_control_urls \
    "$@"
