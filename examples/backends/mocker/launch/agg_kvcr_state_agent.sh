#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# GPU-free KVCR router path: KV router frontend + external KV state agent +
# mocker workers. Each mocker DP rank attaches its raw ZMQ stream to the state
# agent; host-pinned (G2) events are KVCR-owned, so the router indexes them in
# its HostPinned lower tier under one CacheOwner per worker DP rank. With
# ROUTER_HINTS=1 workers advertise router hints, so the router also runs its
# hinted lower-tier lookup. Mockers ignore the hints: this measures router cost,
# not KV transfer benefit.
set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DP_SIZE="${DP_SIZE:-1}"
NUM_HOST_BLOCKS="${NUM_HOST_BLOCKS:-4096}"
ZMQ_BASE_PORT="${ZMQ_BASE_PORT:-20080}"
ROUTER_HINTS="${ROUTER_HINTS:-1}"
# Like a vLLM worker's dynamo_state_agent settings: the connectable address the
# state agent dials for raw KV events, and this process's first CacheOwner slot.
RAW_HOST="${RAW_HOST:-$(hostname -I | awk '{print $1}')}"
CACHE_SLOT="${CACHE_SLOT:-0}"
STATE_AGENT_SYSTEM_PORT="${STATE_AGENT_SYSTEM_PORT:-9091}"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching KVCR state agent + KV router + mocker" "$MODEL" "$HTTP_PORT"

# 1. One state agent per namespace, sized for every worker DP rank. Workers
#    wait up to DYN_KV_STATE_AGENT_HOST_DISCOVERY_TIMEOUT_SECS (30 s) for it.
DYN_COMPONENT=kv_state_agent \
DYN_HEALTH_CHECK_ENABLED=false \
DYN_SYSTEM_PORT="$STATE_AGENT_SYSTEM_PORT" \
python3 -m dynamo.kv_state_agent --max-slots $((NUM_WORKERS * DP_SIZE)) &

wait_for_ready "http://127.0.0.1:${STATE_AGENT_SYSTEM_PORT}/live" 60

# 2. Mocker workers. Each worker's DP ranks publish on base_port + dp_rank.
ZMQ_PORTS=$(seq -s, "$ZMQ_BASE_PORT" "$DP_SIZE" $((ZMQ_BASE_PORT + (NUM_WORKERS - 1) * DP_SIZE)))
HINT_ARGS=()
[ "$ROUTER_HINTS" = "1" ] && HINT_ARGS=(--kv-state-agent-router-hints)
python3 -m dynamo.mocker \
    --model-path "$MODEL" \
    --block-size "$BLOCK_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --data-parallel-size "$DP_SIZE" \
    --num-host-blocks "$NUM_HOST_BLOCKS" \
    --zmq-kv-events-ports "$ZMQ_PORTS" \
    --kv-state-agent \
    --kv-state-agent-raw-host "$RAW_HOST" \
    --kv-state-agent-cache-slot "$CACHE_SLOT" \
    "${HINT_ARGS[@]}" &

# 3. Frontend with KV routing.
python3 -m dynamo.frontend \
    --router-mode kv \
    --kv-cache-block-size "$BLOCK_SIZE" &

wait_any_exit
