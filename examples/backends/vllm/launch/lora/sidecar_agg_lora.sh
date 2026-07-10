#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../../common/launch_utils.sh"

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"
MAX_LORAS="${MAX_LORAS:-4}"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"
OPENENGINE_HOST="${OPENENGINE_HOST:-127.0.0.1}"
OPENENGINE_PORT="${OPENENGINE_PORT:-50051}"
VLLM_RS_HTTP_PORT="${VLLM_RS_HTTP_PORT:-8100}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
SYSTEM_PORT="${DYN_SYSTEM_PORT:-8081}"

export DYN_LORA_ENABLED="${DYN_LORA_ENABLED:-true}"
export DYN_LORA_PATH="${DYN_LORA_PATH:-/tmp/dynamo_loras}"
mkdir -p "$DYN_LORA_PATH"

print_launch_banner --no-curl "Launching Sidecar Aggregated Serving + LoRA (1 GPU)" "$MODEL" "$HTTP_PORT"

python -m dynamo.frontend &

vllm-rs serve "$MODEL" \
    --port "$VLLM_RS_HTTP_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --engine-rpc-host "$OPENENGINE_HOST" \
    --engine-rpc-port "$OPENENGINE_PORT" \
    --enforce-eager \
    --max-num-seqs "$MAX_CONCURRENT_SEQS" \
    --enable-lora \
    --max-loras "$MAX_LORAS" \
    --max-lora-rank "$MAX_LORA_RANK" &

DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT="$SYSTEM_PORT" \
    dynamo-vllm-sidecar \
    --openengine-endpoint "${OPENENGINE_HOST}:${OPENENGINE_PORT}" &

wait_any_exit
