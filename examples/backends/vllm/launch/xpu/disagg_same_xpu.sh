#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Disaggregated prefill/decode on a SINGLE XPU.
#
# Unlike the CUDA version (disagg_same_gpu.sh), we cannot use gpu_utils.sh
# because it depends on nvidia-smi. Instead, set GPU_MEM_FRACTION manually
# based on your XPU's VRAM. For Intel Data Center GPU Max 1550 (128 GiB HBM2e)
# with Qwen3-0.6B, 0.15 per worker (~19 GiB each) is conservative.
#
# Override GPU_MEM_FRACTION via env var to tune for your hardware.

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../../common/launch_utils.sh"

# Common configuration
MODEL="Qwen/Qwen3-0.6B"
BLOCK_SIZE=64
NIXL_BUFFER_DEVICE=xpu
VLLM_NIXL_BACKEND=UCX

# UCX configuration for Intel XPU (Level Zero copy)
export UCX_MEMTYPE_CACHE=0
#export UCX_TLS=shm,ze_copy
# Adjust UCX_NET_DEVICES to match your InfiniBand/RDMA devices, or remove if not using RDMA
# export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1

# ---- Tunable (override via env vars) ----
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"

# GPU memory fraction PER WORKER. Two workers share one XPU.
# nvidia-smi is not available on XPU, so gpu_utils.sh cannot auto-calculate.
# Adjust based on your XPU's total VRAM:
#   Intel Data Center GPU Max 1550 (128 GiB): 0.10-0.15 per worker
#   Intel Data Center GPU Max 1100 (48 GiB):  0.15-0.25 per worker
#   Intel Arc A770 (16 GiB):                  0.35-0.45 per worker
GPU_MEM_FRACTION="${GPU_MEM_FRACTION:-0.40}"

KV_TRANSFER_CONFIG="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_buffer_device\":\"${NIXL_BUFFER_DEVICE}\",\"kv_connector_extra_config\":{\"backends\":[\"${VLLM_NIXL_BACKEND}\"]}}"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Disaggregated on Same XPU (1 XPU)" "$MODEL" "$HTTP_PORT" \
    "Workers:     2 (prefill + decode, fraction ${GPU_MEM_FRACTION} per worker)"

# run ingress
python3 -m dynamo.frontend &

# run decode worker
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
ZE_AFFINITY_MASK=0 python3 -m dynamo.vllm \
  --model "$MODEL" \
  --block-size $BLOCK_SIZE \
  --disaggregation-mode decode \
  --kv-transfer-config "$KV_TRANSFER_CONFIG" \
  --gpu-memory-utilization "${GPU_MEM_FRACTION}" \
  --max-model-len "$MAX_MODEL_LEN" &

# Wait for decode worker to initialize before starting prefill worker
# This prevents both workers from competing for GPU memory simultaneously.
echo "Waiting for decode worker to initialize..."
sleep 20

# run prefill worker
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
ZE_AFFINITY_MASK=0 python3 -m dynamo.vllm \
  --model "$MODEL" \
  --block-size $BLOCK_SIZE \
  --disaggregation-mode prefill \
  --kv-transfer-config "$KV_TRANSFER_CONFIG" \
  --gpu-memory-utilization "${GPU_MEM_FRACTION}" \
  --max-model-len "$MAX_MODEL_LEN" \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}' &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
