#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Bare-process Global Router demo: two aggregated mocker pools, CPU-only (no GPU).
#
# Topology (all on one host, over a shared etcd + NATS):
#
#   client --> frontend (ns: grdemo) --> global router (ns: grdemo)
#                                          |-- agg pool 0: local router + mocker (ns: agg_pool_0)
#                                          +-- agg pool 1: local router + mocker (ns: agg_pool_1)
#
# The global router picks a pool from the request's SLA target (TTFT here) using the
# 2D grid in agg_config.json, then forwards to that pool's local router.
#
# Prerequisites: a running etcd (:2379) and NATS (:4222) -- e.g. `docker compose -f
# deploy/docker-compose.yml up -d` -- and the `dynamo` Python package installed
# (`pip install ai-dynamo[...]` or an editable source install on PYTHONPATH).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="$HERE/logs"; mkdir -p "$LOG"
export DYN_LOG="${DYN_LOG:-info}"

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"   # real HF id: the global router fetches this model's card
POOL_MODEL="qwen-pool-demo"         # distinct served-model-name so the frontend does NOT discover
                                    # the pool workers directly and merge them into the public model
GR_NS="grdemo"                      # global router + frontend namespace
PORT="${PORT:-8000}"

echo "== launching 2 agg mocker pools, logs in $LOG =="
for i in 0 1; do
  python3 -m dynamo.mocker \
    --endpoint "dyn://agg_pool_${i}.backend.generate" \
    --model-path "$MODEL" --model-name "$POOL_MODEL" \
    --block-size 64 --disaggregation-mode agg --speedup-ratio 10 \
    > "$LOG/pool${i}_worker.log" 2>&1 &
  python3 -m dynamo.router \
    --endpoint "agg_pool_${i}.backend.generate" --router-block-size 64 \
    > "$LOG/pool${i}_router.log" 2>&1 &
done

echo "== waiting 25s for pool routers to register (the global router errors if a pool router isn't up yet) =="
sleep 25

echo "== launching global router (ns: $GR_NS) =="
python3 -m dynamo.global_router \
  --config "$HERE/agg_config.json" --model-name "$MODEL" --namespace "$GR_NS" \
  > "$LOG/global_router.log" 2>&1 &
sleep 8

# Frontend is scoped to the global router's namespace. NOTE: discovery is global across
# namespaces, so the distinct POOL_MODEL above (not --namespace) is what keeps the frontend
# from bypassing the global router. See ../README.md and the global_router README "Common Pitfalls".
echo "== launching frontend on :$PORT (ns: $GR_NS) =="
python3 -m dynamo.frontend --http-port "$PORT" --namespace "$GR_NS" \
  > "$LOG/frontend.log" 2>&1 &

echo "== started. settling 6s, then listing models =="
sleep 6
curl -s "http://localhost:${PORT}/v1/models"; echo
echo "== ready. PIDs: $(jobs -p | tr '\n' ' ') =="
echo "   send a request: $HERE/client.sh $PORT"
echo "   tail logs:      tail -f $LOG/*.log"
echo "   stop all:       pkill -f 'dynamo.(mocker|router|global_router|frontend)'"
wait
