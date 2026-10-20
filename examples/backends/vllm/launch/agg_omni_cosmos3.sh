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

# Cosmos uses diffusion memory, not the text-generation KV-cache budget from
# gpu_utils.sh. Pass diffusion parallelism/offload options directly to the worker.
MODEL="nvidia/Cosmos3-Nano"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
                echo "--model requires a model identifier" >&2
                exit 2
            fi
            MODEL="$2"
            shift 2
            ;;
        --model=*)
            MODEL="${1#*=}"
            if [[ -z "$MODEL" ]]; then
                echo "--model requires a model identifier" >&2
                exit 2
            fi
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done
MODEL_PATH="${DYN_COSMOS_MODEL_PATH:-$MODEL}"
MODALITY="${DYN_COSMOS_MODALITY:-image}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"

case "$MODALITY" in
    image|video) ;;
    *)
        echo "DYN_COSMOS_MODALITY must be image or video" >&2
        exit 1
        ;;
esac

print_launch_banner --no-curl "Launching Cosmos3 ($MODALITY)" "$MODEL" "$HTTP_PORT"
echo "Model path: $MODEL_PATH"
echo "Guardrails follow the model default; pass --no-guardrails to disable them explicitly."

trap 'echo Cleaning up...; kill 0' EXIT
python -m dynamo.frontend &

DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    python -m dynamo.vllm.omni \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL" \
    --output-modalities "$MODALITY" \
    --default-video-fps 24 \
    --enforce-eager \
    --media-output-fs-url "${DYN_COSMOS_MEDIA_OUTPUT_FS_URL:-file:///tmp/dynamo_media}" \
    "${EXTRA_ARGS[@]}" &

wait_any_exit
