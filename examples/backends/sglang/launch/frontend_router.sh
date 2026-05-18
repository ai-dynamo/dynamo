#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Frontend-only launcher with configurable router mode. Pairs with
# diffusion_llada_multi.sh (one or more workers).
#
# Usage: frontend_router.sh --mode <round-robin|kv-approx|kv-events>
#   round-robin : stateless RR, no KV indexer.
#   kv-approx   : --router-mode kv --no-router-kv-events (router self-records routing decisions).
#   kv-events   : --router-mode kv (default; requires engine to emit BlockStored events).

set -e

MODE=""
HTTP_PORT="${HTTP_PORT:-8001}"
KV_BLOCK_SIZE="${KV_BLOCK_SIZE:-32}"          # must match worker --page-size
KV_OVERLAP_SCORE_WEIGHT="${KV_OVERLAP_SCORE_WEIGHT:-2.0}"
KV_TTL_SECS="${KV_TTL_SECS:-300}"
OSL_LOAD_WEIGHT="${OSL_LOAD_WEIGHT:-0.0}"     # diffusion-LM: weight OSL block cost

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --http-port) HTTP_PORT="$2"; shift 2 ;;
    --router-osl-load-weight) OSL_LOAD_WEIGHT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

case "$MODE" in
  round-robin)
    EXTRA="--router-mode round-robin"
    ;;
  kv-approx)
    EXTRA="--router-mode kv --no-router-kv-events --router-kv-overlap-score-weight $KV_OVERLAP_SCORE_WEIGHT --router-ttl-secs $KV_TTL_SECS --kv-cache-block-size $KV_BLOCK_SIZE"
    ;;
  kv-events)
    EXTRA="--router-mode kv --router-kv-events --router-kv-overlap-score-weight $KV_OVERLAP_SCORE_WEIGHT --kv-cache-block-size $KV_BLOCK_SIZE"
    ;;
  *)
    echo "Usage: $0 --mode <round-robin|kv-approx|kv-events>" >&2
    exit 1
    ;;
esac

# Only inject OSL load weight on kv modes (the flag is on the kv-router config).
if [[ "$MODE" == "kv-approx" || "$MODE" == "kv-events" ]]; then
  EXTRA="$EXTRA --router-osl-load-weight $OSL_LOAD_WEIGHT"
fi

echo "=========================================="
echo "Dynamo Frontend"
echo "  Port:       $HTTP_PORT"
echo "  Mode:       $MODE"
echo "  Block size: $KV_BLOCK_SIZE"
echo "  OSL weight: $OSL_LOAD_WEIGHT"
echo "  Extra:      $EXTRA"
echo "=========================================="

exec python -m dynamo.frontend --http-port "$HTTP_PORT" $EXTRA
