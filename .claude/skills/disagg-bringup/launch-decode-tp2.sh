#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# Launch the Decode vLLM instance with TP=2 (asymmetric vs TP=1 prefill).
#
# Sibling of launch-decode.sh but spans TWO GPUs. Pair with
# launch-prefill-gpu2.sh which puts prefill on a separate GPU.
#
# Topology this script targets:
#   decode  : CUDA_VISIBLE_DEVICES=0,1  TP=2  port=8001
#   prefill : CUDA_VISIBLE_DEVICES=2    TP=1  port=8000
#
# Why a sibling script and not a flag: the original launch-decode.sh is
# hardcoded into two-request-smoke.sh and other smokes via path. Forking
# the launcher keeps the legacy TP=1↔TP=1 path stable while we exercise
# the asymmetric path end-to-end through vLLM.
#
# Env vars:
#   KVBM_VENV         (default: <repo>/.sandbox)
#   KVBM_HARDWARE_PROFILE (default: spark-gb10; also supports h100-a100, custom)
#   KVBM_DECODE_GPUS  (default: 0,1)
#   KVBM_DECODE_TP    (default: 2)
#   KVBM_DECODE_MEMUTIL (default: 0.6)  — higher than 0.15 because
#                                          decode no longer shares
#                                          GPU with prefill
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${KVBM_REPO:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
. "$SCRIPT_DIR/hardware-profiles.sh"
kvbm_apply_disagg_bringup_profile

KVBM_VENV=${KVBM_VENV:-$REPO/.sandbox}
KVBM_CONNECTOR_MODULE_PATH=${KVBM_CONNECTOR_MODULE_PATH:-kvbm.v2.vllm.connector}
KVBM_DECODE_GPUS=${KVBM_DECODE_GPUS:-0,1}
KVBM_DECODE_TP=${KVBM_DECODE_TP:-2}
KVBM_DECODE_MEMUTIL=${KVBM_DECODE_MEMUTIL:-$KVBM_DECODE_GPU_MEMORY_UTILIZATION}

export CUDA_VISIBLE_DEVICES=$KVBM_DECODE_GPUS
export DYN_KVBM_CPU_CACHE_GB="$KVBM_CPU_CACHE_GB"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
exec "$KVBM_VENV/bin/python3" -m vllm.entrypoints.openai.api_server \
  --model "$KVBM_MODEL" \
  --served-model-name "$KVBM_MODEL" \
  --max-model-len "$KVBM_MAX_MODEL_LEN" \
  --max-num-seqs "$KVBM_MAX_NUM_SEQS" \
  --tensor-parallel-size "$KVBM_DECODE_TP" \
  --gpu-memory-utilization "$KVBM_DECODE_MEMUTIL" \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --port 8001 \
  --kv-transfer-config '{
    "kv_connector": "DynamoConnector",
    "kv_role": "kv_both",
    "kv_load_failure_policy": "recompute",
    "kv_connector_module_path": "'"$KVBM_CONNECTOR_MODULE_PATH"'",
    "kv_connector_extra_config": {
      "leader": {
        "disagg": { "hub_url": "http://127.0.0.1:1337", "role": "decode" },
        "cache":  { "host": { "cache_size_gb": '"$KVBM_CPU_CACHE_GB"' } },
        "tokio":  { "worker_threads": 2 }
      },
      "worker": {
        "nixl":  { "backends": { "UCX": {}, "POSIX": {} } },
        "tokio": { "worker_threads": 2 }
      }
    }
  }'
