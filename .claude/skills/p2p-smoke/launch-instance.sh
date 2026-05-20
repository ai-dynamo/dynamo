#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# Launch a single Qwen3-0.6B vLLM instance registered with kvbm-hub.
# Identical to disagg-bringup/launch-prefill.sh except port + role + cache_size
# are parameterized so we can spin up N independent instances on one GPU
# for the P2P smoke. Both instances register Feature::P2P (the hub
# enforces P2P alongside ConditionalDisagg, so role=prefill is fine even
# when we never use the CD prefill queue).
#
# Env vars (all required except where noted):
#   KVBM_VENV          (default: /home/ryan/.venvs/dynamo-kvbm)
#   P2P_PORT           — vLLM listen port (e.g. 8000, 8002)
#   P2P_ROLE           — "prefill" or "decode" (cosmetic; transfer endpoints
#                         work on either)
#   P2P_CACHE_GB       (default: 2)  — G2 host cache size
#   P2P_GMU            (default: 0.15) — gpu-memory-utilization (Spark fits ~6 of these)
#   KVBM_BLOCK_LAYOUT  (default: operational)
#   KVBM_ONBOARD_MODE  (default: inter)
set -eu

KVBM_VENV=${KVBM_VENV:-/home/ryan/.venvs/dynamo-kvbm}
P2P_PORT=${P2P_PORT:?"P2P_PORT required"}
P2P_ROLE=${P2P_ROLE:-prefill}
P2P_CACHE_GB=${P2P_CACHE_GB:-2}
P2P_GMU=${P2P_GMU:-0.15}
KVBM_BLOCK_LAYOUT=${KVBM_BLOCK_LAYOUT:-operational}
KVBM_ONBOARD_MODE=${KVBM_ONBOARD_MODE:-inter}

case "$P2P_ROLE" in
  prefill|decode) ;;
  *) echo "P2P_ROLE must be 'prefill' or 'decode', got: '$P2P_ROLE'" >&2; exit 1 ;;
esac
case "$KVBM_BLOCK_LAYOUT" in
  operational|universal) ;;
  *) echo "KVBM_BLOCK_LAYOUT must be 'operational' or 'universal', got: '$KVBM_BLOCK_LAYOUT'" >&2; exit 1 ;;
esac

export CUDA_VISIBLE_DEVICES=0
export DYN_KVBM_CPU_CACHE_GB="$P2P_CACHE_GB"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
exec "$KVBM_VENV/bin/python3" -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-0.6B \
  --max-model-len 1024 \
  --max-num-seqs 8 \
  --gpu-memory-utilization "$P2P_GMU" \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --port "$P2P_PORT" \
  --kv-transfer-config '{
    "kv_connector": "DynamoConnector",
    "kv_role": "kv_both",
    "kv_load_failure_policy": "recompute",
    "kv_connector_module_path": "kvbm.v2.vllm.schedulers.connector",
    "kv_connector_extra_config": {
      "default": { "block_layout": "'"$KVBM_BLOCK_LAYOUT"'" },
      "leader": {
        "disagg":  { "hub_url": "http://127.0.0.1:1337", "role": "'"$P2P_ROLE"'" },
        "cache":   { "host": { "cache_size_gb": '"$P2P_CACHE_GB"'.0 } },
        "tokio":   { "worker_threads": 2 },
        "control": { "metrics": true },
        "onboard": { "mode": "'"$KVBM_ONBOARD_MODE"'" }
      },
      "worker": {
        "nixl":  { "backends": { "UCX": {}, "POSIX": {} } },
        "tokio": { "worker_threads": 2 }
      }
    }
  }'
