#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# Launch the Decode vLLM instance for the disagg-bringup smoke.
# Same as launch-prefill.sh but role=decode and port=8001.
#
# Env vars (mirrors start-hub.sh's pattern):
#   KVBM_VENV          (default: <repo>/.sandbox)
#   KVBM_HARDWARE_PROFILE (default: spark-gb10; also supports h100-a100, custom)
#   KVBM_BLOCK_LAYOUT  (default: operational)  — injected into kv_connector_extra_config
#                        so it survives vLLM's EngineCore subprocess spawn.
#                        Valid values: operational | universal
#   KVBM_ONBOARD_MODE  (default: inter)        — see launch-prefill.sh for semantics.
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${KVBM_REPO:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
. "$SCRIPT_DIR/hardware-profiles.sh"
kvbm_apply_disagg_bringup_profile

KVBM_VENV=${KVBM_VENV:-$REPO/.sandbox}
KVBM_CONNECTOR_MODULE_PATH=${KVBM_CONNECTOR_MODULE_PATH:-kvbm.v2.vllm.connector}
KVBM_BLOCK_LAYOUT=${KVBM_BLOCK_LAYOUT:-operational}
case "$KVBM_BLOCK_LAYOUT" in
  operational|universal) ;;
  *) echo "KVBM_BLOCK_LAYOUT must be 'operational' or 'universal', got: '$KVBM_BLOCK_LAYOUT'" >&2; exit 1 ;;
esac
KVBM_ONBOARD_MODE=${KVBM_ONBOARD_MODE:-inter}
case "$KVBM_ONBOARD_MODE" in
  inter|intra) ;;
  *) echo "KVBM_ONBOARD_MODE must be 'inter' or 'intra', got: '$KVBM_ONBOARD_MODE'" >&2; exit 1 ;;
esac

export CUDA_VISIBLE_DEVICES="$KVBM_DECODE_CUDA_VISIBLE_DEVICES"
export DYN_KVBM_CPU_CACHE_GB="$KVBM_CPU_CACHE_GB"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
exec "$KVBM_VENV/bin/python3" -m vllm.entrypoints.openai.api_server \
  --model "$KVBM_MODEL" \
  --served-model-name "$KVBM_MODEL" \
  --max-model-len "$KVBM_MAX_MODEL_LEN" \
  --max-num-seqs "$KVBM_MAX_NUM_SEQS" \
  --gpu-memory-utilization "$KVBM_DECODE_GPU_MEMORY_UTILIZATION" \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --port 8001 \
  --kv-transfer-config '{
    "kv_connector": "DynamoConnector",
    "kv_role": "kv_both",
    "kv_load_failure_policy": "recompute",
    "kv_connector_module_path": "'"$KVBM_CONNECTOR_MODULE_PATH"'",
    "kv_connector_extra_config": {
      "default": { "block_layout": "'"$KVBM_BLOCK_LAYOUT"'" },
      "leader": {
        "disagg":  { "hub_url": "http://127.0.0.1:1337", "role": "decode" },
        "cache":   { "host": { "cache_size_gb": '"$KVBM_CPU_CACHE_GB"' } },
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
