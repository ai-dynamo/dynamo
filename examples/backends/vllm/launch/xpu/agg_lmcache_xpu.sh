#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e

# Explicitly unset PROMETHEUS_MULTIPROC_DIR to let LMCache or Dynamo manage it internally
unset PROMETHEUS_MULTIPROC_DIR

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../../common/launch_utils.sh"

export VLLM_TARGET_DEVICE=xpu

# Device affinity: Use auto-selected device via ZE_AFFINITY_MASK if set by test framework,
# otherwise default to device 0
ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0}
export ZE_AFFINITY_MASK

trap 'echo Cleaning up...; kill 0' EXIT

MODEL="Qwen/Qwen3-0.6B"

# ---- Tunable (override via env vars) ----
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"

GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Aggregated Serving + LMCache (1 GPU)" "$MODEL" "$HTTP_PORT"

python -m dynamo.frontend &

DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
  ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK \
  python -m dynamo.vllm --model "$MODEL" --enforce-eager \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_CONCURRENT_SEQS" \
  --block-size "${BLOCK_SIZE:-64}" \
  --gpu-memory-utilization 0.75 \
  $GPU_MEM_ARGS \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_buffer_device":"xpu"}' &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
