#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Explicitly unset PROMETHEUS_MULTIPROC_DIR to let LMCache or Dynamo manage it internally
unset PROMETHEUS_MULTIPROC_DIR

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MODEL="Qwen/Qwen3-0.6B"

if [[ "${DYN_DEVICE:-cuda}" == "xpu" ]]; then
    # --block-size 64 is required for XPU; on CUDA vLLM uses its default
    BLOCK_SIZE_ARG=(--block-size "${BLOCK_SIZE:-64}")
    KV_TRANSFER_CONFIG='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_buffer_device":"xpu"}'
else
    BLOCK_SIZE_ARG=()
    KV_TRANSFER_CONFIG='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
fi

# ---- Tunable (override via env vars) ----
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"

GPU_MEM_FRACTION=$(build_gpu_mem_args vllm --model "$MODEL" --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_CONCURRENT_SEQS")

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Aggregated Serving + LMCache (1 GPU)" "$MODEL" "$HTTP_PORT"

python -m dynamo.frontend &

DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
  python -m dynamo.vllm --model "$MODEL" --enforce-eager \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_CONCURRENT_SEQS" \
  "${BLOCK_SIZE_ARG[@]}" \
  ${GPU_MEM_FRACTION:+--gpu-memory-utilization "$GPU_MEM_FRACTION"} --kv-transfer-config "$KV_TRANSFER_CONFIG" &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
