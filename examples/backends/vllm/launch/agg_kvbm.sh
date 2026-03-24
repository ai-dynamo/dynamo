#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

MODEL="Qwen/Qwen3-0.6B"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

print_launch_banner "Launching Aggregated Serving + KVBM (1 GPU)" "$MODEL" "$HTTP_PORT"


# run ingress
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python -m dynamo.frontend &

# run worker with KVBM enabled
# NOTE: remove --enforce-eager for production use
DYN_KVBM_CPU_CACHE_GB=20 \
  python -m dynamo.vllm --model "$MODEL" --kv-transfer-config '{"kv_connector":"DynamoConnector","kv_connector_module_path":"kvbm.vllm_integration.connector","kv_role":"kv_both"}' --enforce-eager &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
