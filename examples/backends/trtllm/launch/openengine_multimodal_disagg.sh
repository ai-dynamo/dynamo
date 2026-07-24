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

# Qwen3-VL supports ordered image/video URLs and data URIs in both aggregate
# and context-first P/D modes. Decode intentionally receives the original media
# again so TRT-LLM can reconstruct its local multimodal position metadata.
set -euo pipefail
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"
export DYNAMO_HOME=${DYNAMO_HOME:-"$(readlink -f "$SCRIPT_DIR/../../../..")"}

export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-VL-2B-Instruct}
export PREFILL_CONFIG=${PREFILL_CONFIG:-$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3-vl-2b-instruct/prefill.yaml}
export DECODE_CONFIG=${DECODE_CONFIG:-$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3-vl-2b-instruct/decode.yaml}
export DYN_LORA_ENABLED=${DYN_LORA_ENABLED:-false}
export OPENENGINE_LAUNCH_MODE=multimodal
exec "$SCRIPT_DIR/openengine_disagg.sh" "$@"
