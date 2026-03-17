#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Explicitly unset PROMETHEUS_MULTIPROC_DIR to let LMCache or Dynamo manage it internally
unset PROMETHEUS_MULTIPROC_DIR

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MODEL="Qwen/Qwen3-0.6B"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Aggregated Serving + LMCache (1 GPU)" "$MODEL" "$HTTP_PORT"

# run ingress
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python -m dynamo.frontend &

# run worker with LMCache enabled (without PROMETHEUS_MULTIPROC_DIR set externally)
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
  python -m dynamo.vllm --model "$MODEL" --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
