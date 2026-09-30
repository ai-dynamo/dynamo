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
# shellcheck source=../../../common/gpu_utils.sh
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"
# shellcheck source=../../../common/launch_utils.sh
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path)
            if [[ $# -lt 2 || -z "$2" ]]; then
                echo "--model-path requires a model name or local path" >&2
                exit 2
            fi
            MODEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--model-path MODEL] [additional SGLang/Dynamo worker flags]"
            echo "Environment: MODEL, DYN_HTTP_PORT (8000), DYN_SYSTEM_PORT (8081), CONTEXT_LENGTH (4096)"
            exit 0
            ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

command -v setsid >/dev/null || { echo "setsid is required for scoped process cleanup" >&2; exit 1; }
GPU_MEM_ARGS=$(build_sglang_gpu_mem_args)
GPU_ARGS=()
if [[ -n "$GPU_MEM_ARGS" ]]; then
    read -r -a GPU_ARGS <<< "$GPU_MEM_ARGS"
fi
export DYN_HTTP_PORT="${DYN_HTTP_PORT:-8000}"
WORKER_PORT=$(dyn_port DYN_SYSTEM_PORT 1 "${DYN_SYSTEM_PORT:-8081}")
CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"
CHILD_GROUPS=()

# Each child starts its own session; override shared cleanup to signal only those sessions.
dynamo_reap_and_exit() {
    local result="${1:-0}"
    trap - EXIT
    trap '' TERM INT
    local child
    for child in "${CHILD_GROUPS[@]}"; do
        kill -TERM -- "-$child" 2>/dev/null || true
    done
    local attempt running
    for ((attempt=0; attempt<50; attempt++)); do
        running=false
        for child in "${CHILD_GROUPS[@]}"; do
            if kill -0 -- "-$child" 2>/dev/null; then running=true; fi
        done
        if [[ "$running" == false ]]; then break; fi
        sleep 0.1
    done
    for child in "${CHILD_GROUPS[@]}"; do
        kill -KILL -- "-$child" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    exit "$result"
}
trap dynamo_exit_trap EXIT
trap 'dynamo_reap_and_exit 0' TERM INT

print_launch_banner "Launching Experimental System One (Aggregated SGLang)" "$MODEL" "$DYN_HTTP_PORT" \
    "System One: POST /v1/systemone (zero-output candidate scoring)" \
    "Smoke test: MODEL='$MODEL' BASE_URL=http://localhost:$DYN_HTTP_PORT bash examples/systemone/smoke.sh"

setsid python3 -m dynamo.frontend --enable-systemone-api --migration-limit 0 &
CHILD_GROUPS+=("$!")

DYN_SYSTEM_PORT="$WORKER_PORT" setsid python3 -m dynamo.sglang \
    --model-path "$MODEL" \
    --served-model-name "$MODEL" \
    --context-length "$CONTEXT_LENGTH" \
    --page-size 16 \
    --tp 1 \
    --enable-metrics \
    --disable-piecewise-cuda-graph \
    "${GPU_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" &
CHILD_GROUPS+=("$!")

wait_any_exit
