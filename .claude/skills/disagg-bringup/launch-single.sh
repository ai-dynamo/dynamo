#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# Launch a single KVBM-enabled vLLM instance — aggregated (non-disagg) mode.
# No hub, no prefill/decode CD coordination. The connector runs through
# `ConnectorLeader::update_state_after_alloc` which honors `onboard.mode`,
# so `KVBM_ONBOARD_MODE=intra` actually fires `execute_local_layerwise_onboard`
# on warm-prefix requests. This is the bringup the intra-pass-onboard
# smoke uses to exercise the kernel-catalog + `layer_range` path.
#
# Env vars (default-friendly so single-arg invocation works):
#   KVBM_VENV          (default: <repo>/.sandbox)
#   KVBM_HARDWARE_PROFILE (default: spark-gb10; also supports h100-a100, custom)
#   KVBM_BLOCK_LAYOUT  (operational | universal ; default operational)
#   KVBM_ONBOARD_MODE  (inter | intra            ; default inter)
#   KVBM_SINGLE_PORT   (default: 8002)  — separate from 8000/8001 used by
#                       the disagg prefill/decode pair, so single + disagg
#                       smokes can coexist if needed.
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
KVBM_SINGLE_PORT=${KVBM_SINGLE_PORT:-8002}

export CUDA_VISIBLE_DEVICES="$KVBM_SINGLE_CUDA_VISIBLE_DEVICES"
export DYN_KVBM_CPU_CACHE_GB="$KVBM_CPU_CACHE_GB"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
exec "$KVBM_VENV/bin/python3" -m vllm.entrypoints.openai.api_server \
  --model "$KVBM_MODEL" \
  --served-model-name "$KVBM_MODEL" \
  --max-model-len "$KVBM_MAX_MODEL_LEN" \
  --max-num-seqs "$KVBM_MAX_NUM_SEQS" \
  --gpu-memory-utilization "$KVBM_SINGLE_GPU_MEMORY_UTILIZATION" \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --port "$KVBM_SINGLE_PORT" \
  --kv-transfer-config '{
    "kv_connector": "DynamoConnector",
    "kv_role": "kv_both",
    "kv_load_failure_policy": "recompute",
    "kv_connector_module_path": "'"$KVBM_CONNECTOR_MODULE_PATH"'",
    "kv_connector_extra_config": {
      "default": { "block_layout": "'"$KVBM_BLOCK_LAYOUT"'" },
      "leader": {
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
