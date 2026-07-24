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

# Phi-4 multimodal audio requires TRT-LLM's built-in speech-lora. The engine
# config enables the exact LoRA module mapping and cache bounds for that model.
# Audio P/D is deliberately not advertised by the server in this milestone.
set -euo pipefail
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"
export DYNAMO_HOME=${DYNAMO_HOME:-"$(readlink -f "$SCRIPT_DIR/../../../..")"}

if [[ -z ${MODEL_PATH:-} ]]; then
    echo "MODEL_PATH must point to a raw git clone of microsoft/Phi-4-multimodal-instruct." >&2
    echo "The Hugging Face snapshot layout is not supported by TRT-LLM's Phi-4-MM loader." >&2
    exit 2
fi
if [[ ! -d "$MODEL_PATH/speech-lora" ]]; then
    echo "MODEL_PATH must contain the bundled speech-lora/ directory: $MODEL_PATH" >&2
    exit 2
fi
export MODEL_PATH
export TRTLLM_CONFIG=${TRTLLM_CONFIG:-$DYNAMO_HOME/examples/backends/trtllm/engine_configs/phi-4-multimodal-instruct/audio.yaml}
export DYN_LORA_ENABLED=${DYN_LORA_ENABLED:-false}
export OPENENGINE_LAUNCH_MODE=audio
exec "$SCRIPT_DIR/openengine_agg.sh" "$@"
