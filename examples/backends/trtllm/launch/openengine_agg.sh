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
source "$SCRIPT_DIR/openengine_common.sh"
trap dynamo_exit_trap EXIT

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-0.6B}
HTTP_PORT=${DYN_HTTP_PORT:-${FRONTEND_PORT:-8000}}
export DYN_HTTP_PORT="$HTTP_PORT"
TRTLLM_HTTP_PORT=${TRTLLM_HTTP_PORT:-8001}
OPENENGINE_PORT=${OPENENGINE_PORT:-50051}
SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081}
export DYNAMO_HOME=${DYNAMO_HOME:-"$(readlink -f "$SCRIPT_DIR/../../../..")"}
TRTLLM_CONFIG=${TRTLLM_CONFIG:-$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3/agg.yaml}
OPENENGINE_LAUNCH_MODE=${OPENENGINE_LAUNCH_MODE:-text}
export DYN_LORA_ENABLED=${DYN_LORA_ENABLED:-false}
export DYN_LORA_PATH=${DYN_LORA_PATH:-/tmp/dynamo_loras}
readonly OPENENGINE_SCHEMA_RELEASE=cea19cb06acf03c911b84d5c147e519b60dd92a6
export OPENENGINE_SCHEMA_RELEASE

warn_trtllm_serve_profile_overrides "$TRTLLM_CONFIG"
print_openengine_aggregate_banner "$OPENENGINE_LAUNCH_MODE" "$MODEL_PATH" \
  "$HTTP_PORT" "$TRTLLM_HTTP_PORT" "$OPENENGINE_PORT" "$SYSTEM_PORT"

# The sidecar advertises no frontend media decoder. This intentionally keeps
# http(s) URLs and data URIs intact until TRT-LLM fetches/decodes them.
python3 -m dynamo.frontend --router-mode round-robin &
trtllm-serve "$MODEL_PATH" \
  --backend pytorch \
  --config "$TRTLLM_CONFIG" \
  --host 127.0.0.1 \
  --port "$TRTLLM_HTTP_PORT" \
  --openengine-host 127.0.0.1 \
  --openengine-port "$OPENENGINE_PORT" &
DYN_SYSTEM_PORT="$SYSTEM_PORT" dynamo-openengine-sidecar \
  --openengine-endpoint "http://127.0.0.1:$OPENENGINE_PORT" \
  --expected-engine tensorrt_llm &

wait_any_exit
