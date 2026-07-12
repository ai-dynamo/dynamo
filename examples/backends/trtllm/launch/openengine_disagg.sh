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
PREFILL_HTTP_PORT=${PREFILL_HTTP_PORT:-8001}
DECODE_HTTP_PORT=${DECODE_HTTP_PORT:-8002}
PREFILL_PORT=${PREFILL_PORT:-50051}
DECODE_PORT=${DECODE_PORT:-50052}
PREFILL_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081}
DECODE_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082}
export DYNAMO_HOME=${DYNAMO_HOME:-"$(readlink -f "$SCRIPT_DIR/../../../..")"}
PREFILL_CONFIG=${PREFILL_CONFIG:-$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3/prefill-lora.yaml}
DECODE_CONFIG=${DECODE_CONFIG:-$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3/decode-lora.yaml}
OPENENGINE_LAUNCH_MODE=${OPENENGINE_LAUNCH_MODE:-text}
export DYN_LORA_ENABLED=${DYN_LORA_ENABLED:-true}
export DYN_LORA_PATH=${DYN_LORA_PATH:-/tmp/dynamo_loras}
readonly OPENENGINE_SCHEMA_RELEASE=cea19cb06acf03c911b84d5c147e519b60dd92a6
export OPENENGINE_SCHEMA_RELEASE

warn_trtllm_serve_profile_overrides "$PREFILL_CONFIG and $DECODE_CONFIG"
print_openengine_disagg_banner "$OPENENGINE_LAUNCH_MODE" "$MODEL_PATH" \
  "$HTTP_PORT" "$PREFILL_PORT" "$DECODE_PORT"

# The sidecars advertise no frontend media decoder. URL/data media is therefore
# preserved for both context and generation workers, which is required for
# Qwen3-VL context-first P/D mRoPE reconstruction.
python3 -m dynamo.frontend --router-mode kv &
CUDA_VISIBLE_DEVICES=${PREFILL_CUDA_VISIBLE_DEVICES:-0} \
  trtllm-serve "$MODEL_PATH" --server_role CONTEXT \
  --backend pytorch \
  --config "$PREFILL_CONFIG" \
  --host 127.0.0.1 \
  --port "$PREFILL_HTTP_PORT" \
  --openengine-host 127.0.0.1 \
  --openengine-port "$PREFILL_PORT" &
CUDA_VISIBLE_DEVICES=${DECODE_CUDA_VISIBLE_DEVICES:-1} \
  trtllm-serve "$MODEL_PATH" --server_role GENERATION \
  --backend pytorch \
  --config "$DECODE_CONFIG" \
  --host 127.0.0.1 \
  --port "$DECODE_HTTP_PORT" \
  --openengine-host 127.0.0.1 \
  --openengine-port "$DECODE_PORT" &

DYN_SYSTEM_PORT="$PREFILL_SYSTEM_PORT" dynamo-openengine-sidecar \
  --openengine-endpoint "http://127.0.0.1:$PREFILL_PORT" \
  --expected-engine tensorrt_llm &
DYN_SYSTEM_PORT="$DECODE_SYSTEM_PORT" dynamo-openengine-sidecar \
  --openengine-endpoint "http://127.0.0.1:$DECODE_PORT" \
  --expected-engine tensorrt_llm &

wait_any_exit
