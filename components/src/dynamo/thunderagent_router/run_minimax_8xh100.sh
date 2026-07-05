#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Bring up the MiniMax-M2 A/B stack on one 8xH100 node. Usage:
#   ./run_minimax_8xh100.sh ta
#   ./run_minimax_8xh100.sh kv
set -euo pipefail

POLICY="${1:-}"
if [[ "$POLICY" != "ta" && "$POLICY" != "kv" ]]; then
    echo "usage: $0 ta|kv" >&2
    exit 2
fi

cleanup() {
    jobs -pr | xargs -r kill
}
trap cleanup EXIT

MODEL_PATH="${MODEL_PATH:-MiniMaxAI/MiniMax-M2.7}"
MODEL_NAME_ROUTER="${MODEL_NAME_ROUTER:-MiniMaxAI/MiniMax-M2}"
WORKER_MODEL="$MODEL_NAME_ROUTER"
BLOCK_SIZE=16
HTTP_PORT=8100

if [[ "$POLICY" == "ta" ]]; then
    WORKER_MODEL="dyn-internal-minimax-m2"
fi

export PYTHONHASHSEED=0
export DYN_DISCOVERY_BACKEND=file
export DYN_FILE_KV="${DYN_FILE_KV:-/tmp/dynamo-minimax-${POLICY}-$$}"
export DYN_REQUEST_PLANE=tcp
export DYN_EVENT_PLANE=zmq
mkdir -p "$DYN_FILE_KV"

DYN_SYSTEM_PORT=8181 DYN_FORWARDPASS_METRIC_PORT=20081 \
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m dynamo.vllm \
    --model "$MODEL_PATH" --served-model-name "$WORKER_MODEL" \
    --tensor-parallel-size 4 --block-size "$BLOCK_SIZE" \
    --kv-cache-dtype fp8 --enable-prefix-caching \
    --dyn-tool-call-parser minimax_m2 \
    --dyn-reasoning-parser minimax_append_think \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}' &

DYN_SYSTEM_PORT=8182 DYN_FORWARDPASS_METRIC_PORT=20082 \
VLLM_NIXL_SIDE_CHANNEL_PORT=20098 CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -m dynamo.vllm \
    --model "$MODEL_PATH" --served-model-name "$WORKER_MODEL" \
    --tensor-parallel-size 4 --block-size "$BLOCK_SIZE" \
    --kv-cache-dtype fp8 --enable-prefix-caching \
    --dyn-tool-call-parser minimax_m2 \
    --dyn-reasoning-parser minimax_append_think \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20090","enable_kv_cache_events":true}' &

if [[ "$POLICY" == "ta" ]]; then
    DYN_SYSTEM_PORT=8183 python -m dynamo.thunderagent_router \
        --endpoint dynamo.backend.generate \
        --model-name "$MODEL_NAME_ROUTER" \
        --model-path "$MODEL_PATH" \
        --dyn-tool-call-parser minimax_m2 \
        --dyn-reasoning-parser minimax_append_think \
        --router-block-size "$BLOCK_SIZE" \
        --router-reset-states \
        --shared-cache-type none &
    ROUTER_MODE=round-robin
else
    ROUTER_MODE=kv
fi

DYN_SYSTEM_PORT=8184 python -m dynamo.frontend \
    --http-host 0.0.0.0 \
    --http-port "$HTTP_PORT" \
    --router-mode "$ROUTER_MODE" \
    --router-reset-states \
    --shared-cache-type none \
    --admission-control none &

until curl -fsS "http://127.0.0.1:${HTTP_PORT}/v1/models/${WORKER_MODEL}/ready" 2>/dev/null \
    | jq -e '([.namespaces[].worker_types.aggregated.workers // 0] | add) == 2' >/dev/null; do
    sleep 5
done
until curl -fsS "http://127.0.0.1:${HTTP_PORT}/v1/models" 2>/dev/null | grep -Fq "$MODEL_NAME_ROUTER"; do
    sleep 5
done
echo "$POLICY stack ready at http://127.0.0.1:${HTTP_PORT}/v1"

wait
