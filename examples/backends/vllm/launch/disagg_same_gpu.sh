#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Disaggregated prefill/decode on a SINGLE GPU.
# Each worker needs ~4 GiB; the script computes the per-worker fraction
# automatically using gpu_utils.sh, or honours DYN_GPU_MEMORY_FRACTION_OVERRIDE.
set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../common/gpu_utils.sh"

REQUIRED_GB_PER_WORKER=4

# DYN_GPU_MEMORY_FRACTION_OVERRIDE takes precedence (profiler binary search).
# In single-GPU mode, split the override evenly between the two workers.
if [[ -n "${DYN_GPU_MEMORY_FRACTION_OVERRIDE:-}" ]]; then
    GPU_MEM_FRACTION=$(awk -v f="$DYN_GPU_MEMORY_FRACTION_OVERRIDE" 'BEGIN { printf "%.2f", f / 2 }')
else
    GPU_MEM_FRACTION=$(gpu_gb_to_fraction $REQUIRED_GB_PER_WORKER)
fi

echo "Using ${GPU_MEM_FRACTION} memory fraction per worker (${REQUIRED_GB_PER_WORKER} GiB each)"

# run ingress
python3 -m dynamo.frontend &

# run decode worker with metrics on port 8081
# --enforce-eager is added for quick deployment. for production use, need to remove this flag
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
CUDA_VISIBLE_DEVICES=0 \
python3 -m dynamo.vllm \
  --model Qwen/Qwen3-0.6B \
  --enforce-eager \
  --disaggregation-mode decode \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
  --gpu-memory-utilization ${GPU_MEM_FRACTION} \
  --max-model-len 16384 &

# run prefill worker with metrics on port 8082
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
CUDA_VISIBLE_DEVICES=0 \
python3 -m dynamo.vllm \
  --model Qwen/Qwen3-0.6B \
  --enforce-eager \
  --disaggregation-mode prefill \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
  --gpu-memory-utilization ${GPU_MEM_FRACTION} \
  --max-model-len 16384 \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}' &

# Exit immediately if any background process dies (all are long-running servers).
wait -n
EXIT_CODE=$?
echo "A background process exited with code $EXIT_CODE"
exit $EXIT_CODE
