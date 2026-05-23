#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# Launch the Prefill vLLM instance on GPU 2 (TP=1) for the asymmetric
# TP=2-decode + TP=1-prefill smoke.
#
# Sibling of launch-prefill.sh — same vLLM args, just a different GPU
# and higher memutil (no longer co-located with decode).
#
# Env vars:
#   KVBM_VENV          (default: <repo>/.sandbox)
#   KVBM_HARDWARE_PROFILE (default: spark-gb10; also supports h100-a100, custom)
#   KVBM_PREFILL_GPU   (default: 2)
#   KVBM_PREFILL_MEMUTIL (default: 0.6)
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${KVBM_REPO:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
. "$SCRIPT_DIR/hardware-profiles.sh"
kvbm_apply_disagg_bringup_profile

KVBM_VENV=${KVBM_VENV:-$REPO/.sandbox}
KVBM_CONNECTOR_MODULE_PATH=${KVBM_CONNECTOR_MODULE_PATH:-kvbm.v2.vllm.connector}
KVBM_PREFILL_GPU=${KVBM_PREFILL_GPU:-2}
KVBM_PREFILL_MEMUTIL=${KVBM_PREFILL_MEMUTIL:-$KVBM_PREFILL_GPU_MEMORY_UTILIZATION}

export CUDA_VISIBLE_DEVICES=$KVBM_PREFILL_GPU
export DYN_KVBM_CPU_CACHE_GB="$KVBM_CPU_CACHE_GB"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
exec "$KVBM_VENV/bin/python3" -m vllm.entrypoints.openai.api_server \
  --model "$KVBM_MODEL" \
  --served-model-name "$KVBM_MODEL" \
  --max-model-len "$KVBM_MAX_MODEL_LEN" \
  --max-num-seqs "$KVBM_MAX_NUM_SEQS" \
  --gpu-memory-utilization "$KVBM_PREFILL_MEMUTIL" \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --port 8000 \
  --kv-transfer-config '{
    "kv_connector": "DynamoConnector",
    "kv_role": "kv_both",
    "kv_load_failure_policy": "recompute",
    "kv_connector_module_path": "'"$KVBM_CONNECTOR_MODULE_PATH"'",
    "kv_connector_extra_config": {
      "leader": {
        "disagg": { "hub_url": "http://127.0.0.1:1337", "role": "prefill" },
        "cache":  { "host": { "cache_size_gb": '"$KVBM_CPU_CACHE_GB"' } },
        "tokio":  { "worker_threads": 2 }
      },
      "worker": {
        "nixl":  { "backends": { "UCX": {}, "POSIX": {} } },
        "tokio": { "worker_threads": 2 }
      }
    }
  }'
