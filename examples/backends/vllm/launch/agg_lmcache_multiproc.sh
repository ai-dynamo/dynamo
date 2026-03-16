#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e

# Explicitly set PROMETHEUS_MULTIPROC_DIR (K8s-style deployment)
# Use unique directory per test run to avoid conflicts
export PROMETHEUS_MULTIPROC_DIR=${PROMETHEUS_MULTIPROC_DIR:-/tmp/prometheus_multiproc_$$_$RANDOM}
rm -rf "$PROMETHEUS_MULTIPROC_DIR"
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"

# Cleanup function to remove the directory on exit
cleanup() {
    echo "Cleaning up..."
    rm -rf "$PROMETHEUS_MULTIPROC_DIR"
    kill 0
}
trap cleanup EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MODEL="Qwen/Qwen3-0.6B"
# --block-size 64 is required for XPU; on CUDA vLLM uses its default
if [[ "${DYN_DEVICE:-cuda}" == "xpu" ]]; then
    BLOCK_SIZE_ARG=(--block-size "${DYN_BLOCK_SIZE:-64}")
else
    BLOCK_SIZE_ARG=()
fi
# KV buffer device: set DYN_DEVICE=xpu for Intel XPU hardware (default: cuda)
KV_BUFFER_DEVICE="${DYN_DEVICE:-cuda}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Aggregated + LMCache + Multiproc (1 GPU)" "$MODEL" "$HTTP_PORT"

# run ingress
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python -m dynamo.frontend &

# run worker with LMCache enabled and PROMETHEUS_MULTIPROC_DIR explicitly set
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
  PROMETHEUS_MULTIPROC_DIR="$PROMETHEUS_MULTIPROC_DIR" \
  python -m dynamo.vllm --model "$MODEL" \
    "${BLOCK_SIZE_ARG[@]}" \
    --kv-transfer-config "{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\",\"kv_buffer_device\":\"$KV_BUFFER_DEVICE\"}" &
    
# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
