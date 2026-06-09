#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

MODEL="Qwen/Qwen3-0.6B"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../../../../common/launch_utils.sh"

print_launch_banner "Launching Aggregated Serving + KVBM (1 GPU)" "$MODEL" "$HTTP_PORT"

# Helpful defaults for transfer visibility while validating KVBM behavior.
# Override from environment when needed.
export RUST_LOG="${RUST_LOG:-info,kvbm_connector=debug,kvbm_engine=debug}"
KVBM_CPU_CACHE_GB="${KVBM_CPU_CACHE_GB:-10}"

# Build kv-transfer-config with kv_connector_extra_config so onboarding/offloading
# behavior is explicit and reproducible.
KV_TRANSFER_CONFIG=$(cat <<JSON
{
  "kv_connector": "DynamoConnector",
  "kv_connector_module_path": "kvbm.vllm_integration.connector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "default": {
      "cache": {
        "host": {
          "cache_size_gb": ${KVBM_CPU_CACHE_GB}
        }
      },
      "offload": {
        "g1_to_g2": {
          "policies": ["pass_all"]
        }
      },
      "metrics": {
        "cache_stats_log_interval_secs": 5
      }
    },
    "leader": {
      "onboard": {
        "mode": "intra"
      }
    },
    "worker": {
      "onboard": {
        "mode": "intra"
      }
    }
  }
}
JSON
)

# run ingress
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python -m dynamo.frontend &

# run worker with KVBM enabled
# NOTE: remove --enforce-eager for production use
DYN_KVBM_CPU_CACHE_GB="${KVBM_CPU_CACHE_GB}" \
  python -m dynamo.vllm \
    --model "$MODEL" \
    --block-size 64 \
    --kv-transfer-config "$KV_TRANSFER_CONFIG" \
    --enforce-eager &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
