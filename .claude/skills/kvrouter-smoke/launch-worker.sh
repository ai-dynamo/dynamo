#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# Launch a single dynamo.vllm worker with KVBM v2 connector and prefix
# caching enabled. The worker registers as
# dyn://dynamo.backend.generate; multiple instances on this endpoint
# show up to the router as multiple route targets.
#
# Env vars:
#   KVBM_VENV         (default: ryan-velo-messenger/.sandbox)
#   KVR_PORT          — vLLM API port (not used by dynamo.vllm; it
#                       serves over Dynamo runtime instead. Kept so
#                       multiple workers don't collide on the
#                       default vLLM port if it ever opens one).
#   KVR_KV_EVENT_PORT — ZMQ port for vLLM's internal KV events
#                       publisher. Must differ per worker.
#   KVR_WORKER_LABEL  — shown in logs; cosmetic.
#   KVR_GMU           (default: 0.15) — gpu-memory-utilization
#                       matching disagg/p2p smokes for the Spark.
set -eu

KVBM_VENV=${KVBM_VENV:-/home/ryan/repos/dynamo-workspaces/ryan-velo-messenger/.sandbox}
KVR_KV_EVENT_PORT=${KVR_KV_EVENT_PORT:?"KVR_KV_EVENT_PORT required"}
KVR_WORKER_LABEL=${KVR_WORKER_LABEL:-worker}
KVR_GMU=${KVR_GMU:-0.15}
KVR_CACHE_GB=${KVR_CACHE_GB:-2}

# Per-worker KVBM leader ZMQ pub port. The Python consolidator helper
# derives the consolidator's egress port as (this + 1000). Two workers
# on the same host MUST have distinct values, else they collide on the
# egress port and the second worker dies with "failed to start in-process
# consolidator" / "Address already in use" on bind.
#
# Default 56001 matches the Rust constant DEFAULT_LEADER_ZMQ_PUB_PORT.
# launch-worker.sh's caller (kvrouter-smoke.sh) sets a distinct value
# per worker.
DYN_KVBM_LEADER_ZMQ_PUB_PORT=${DYN_KVBM_LEADER_ZMQ_PUB_PORT:-56001}
export DYN_KVBM_LEADER_ZMQ_PUB_PORT

export CUDA_VISIBLE_DEVICES=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# vLLM's batch-invariant flag isn't needed for routing; leave default.

# Stage 2 wired the v2 connector to spawn lib/kvbm-consolidator
# in-process (see lib/kvbm-engine/src/leader/consolidator.rs +
# lib/kvbm-connector/src/connector/leader/init.rs). KvEventPublisher
# subscribes to the consolidator's output port; consolidator merges
# vLLM ZMQ + kvbm-logical EventsManager streams and publishes the
# unified router-wire stream over NATS to the indexer.
#
# Audit visibility: tracing target `kvbm_consolidator_audit` emits
# INFO lines at ingress_zmq / ingress_kvbm / egress, captured by the
# smoke under DYN_LOG=info (Dynamo's tracing env, default info).

# Minimal v2 KVBM connector config — agg mode, no hub/disagg.
# A G2 host cache tier is required: the v2 leader sanity-check rejects
# a config with no tiers (per tests/kvbm_integration/fixtures/server.py:99).
KV_TRANSFER_CONFIG='{
  "kv_connector": "DynamoConnector",
  "kv_role": "kv_both",
  "kv_connector_module_path": "kvbm.v2.vllm.connector",
  "kv_connector_extra_config": {
    "leader": {
      "tokio":   { "worker_threads": 2 },
      "onboard": { "mode": "intra" },
      "cache":   { "host": { "cache_size_gb": '"$KVR_CACHE_GB"'.0 } }
    },
    "worker": {
      "nixl":  { "backends": { "UCX": {}, "POSIX": {} } },
      "tokio": { "worker_threads": 2 }
    }
  }
}'

# vLLM KV events config — needed so dynamo.vllm's KvEventPublisher has
# a ZMQ source to subscribe to and republish over NATS.
# Bind 0.0.0.0 so the in-process subscriber on 127.0.0.1 finds it.
KV_EVENTS_CONFIG='{
  "endpoint": "tcp://*:'"$KVR_KV_EVENT_PORT"'",
  "enable_kv_cache_events": true
}'

echo "[$KVR_WORKER_LABEL] launching dynamo.vllm with KVBM v2, prefix-cache=on, kv-events=$KVR_KV_EVENT_PORT, gmu=$KVR_GMU"

exec "$KVBM_VENV/bin/python" -m dynamo.vllm \
  --namespace dynamo \
  --endpoint dyn://dynamo.backend.generate \
  --model Qwen/Qwen3-0.6B \
  --max-model-len 1024 \
  --max-num-seqs 8 \
  --gpu-memory-utilization "$KVR_GMU" \
  --enable-prefix-caching \
  --kv-transfer-config "$KV_TRANSFER_CONFIG" \
  --kv-events-config "$KV_EVENTS_CONFIG"
