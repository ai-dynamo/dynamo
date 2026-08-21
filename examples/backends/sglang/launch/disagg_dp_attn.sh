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
#
# Disaggregated DP-attention serving on two shared GPUs. Each prefill/decode
# worker owns two DP ranks over the same GPU pair, which keeps this smoke test
# representative without requiring a separate two-GPU pair for each role.
# GPUs: 2

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MODEL="silence09/DeepSeek-R1-Small-2layers"
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            if [[ $# -lt 2 || "$2" == -* ]]; then
                echo "Missing value for --model"
                exit 1
            fi
            MODEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--model <name>]"
            echo "  --model <name>  MoE model to serve (default: $MODEL)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Parse arguments before installing the process-group cleanup trap so --help or
# an invalid option cannot tear down the caller via kill 0.
trap 'echo Cleaning up...; kill 0' EXIT

GPU_MEM_ARGS=$(build_sglang_gpu_mem_args)
DISAGG_BOOTSTRAP_PORT="${DYN_DISAGG_BOOTSTRAP_PORT:-12345}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"

# SGLang derives a contiguous port block from --dist-init-addr: the supplied
# port plus the next six. Keep the prefill and decode bases at least seven
# apart so the two blocks stay disjoint. The serve test overrides both with
# separately allocated blocks; these defaults keep the script runnable by hand.
PREFILL_DIST_INIT_ADDR="${SGLANG_PREFILL_DIST_INIT_ADDR:-127.0.0.1:29500}"
DECODE_DIST_INIT_ADDR="${SGLANG_DECODE_DIST_INIT_ADDR:-127.0.0.1:29510}"

print_launch_banner "Launching Disaggregated DP Attention (2 shared GPUs)" "$MODEL" "$HTTP_PORT"

python3 -m dynamo.frontend &

# SGLang DP attention uses tp_size == dp_size: the attention replicas occupy
# the visible GPU pair while MoE weights are tensor-parallel across that pair.
# Both P/D workers intentionally share the pair; the tiny two-layer model keeps
# the combined footprint suitable for CI.
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
python3 -m dynamo.sglang \
  --model-path "$MODEL" \
  --served-model-name "$MODEL" \
  --page-size 64 \
  --tp 2 \
  --dp-size 2 \
  --enable-dp-attention \
  --dist-init-addr "$PREFILL_DIST_INIT_ADDR" \
  --trust-remote-code \
  --disaggregation-mode prefill \
  --disaggregation-bootstrap-port "$DISAGG_BOOTSTRAP_PORT" \
  --disaggregation-transfer-backend nixl \
  --load-balance-method round_robin \
  --nccl-port "${DYN_SYSTEM_PORT3:-8083}" \
  --context-length 1024 \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --enable-metrics \
  $GPU_MEM_ARGS &

# Serialize model loading so the two shared-GPU workers do not race for peak
# initialization memory. Decode must not start if prefill never becomes ready.
wait_for_ready "http://localhost:${DYN_SYSTEM_PORT1:-8081}/health" 120

DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
python3 -m dynamo.sglang \
  --model-path "$MODEL" \
  --served-model-name "$MODEL" \
  --page-size 64 \
  --tp 2 \
  --dp-size 2 \
  --enable-dp-attention \
  --dist-init-addr "$DECODE_DIST_INIT_ADDR" \
  --trust-remote-code \
  --disaggregation-mode decode \
  --disaggregation-bootstrap-port "$DISAGG_BOOTSTRAP_PORT" \
  --disaggregation-transfer-backend nixl \
  --prefill-round-robin-balance \
  --nccl-port "${DYN_SYSTEM_PORT4:-8084}" \
  --context-length 1024 \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --enable-metrics \
  $GPU_MEM_ARGS &

wait_any_exit
