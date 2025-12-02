#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Simple agg script (not an optimized config)

print_usage() {
    echo "Usage: $0"
    echo ""
    echo "This script runs aggregated mode (single dynamo.sglang instance)"
    exit 1
}

echo "Mode: aggregated"
echo "Command: dynamo"

# Check if required environment variables are set
if [ -z "$HOST_IP_MACHINE" ]; then
    echo "Error: HOST_IP_MACHINE environment variable is not set"
    exit 1
fi

if [ -z "$PORT" ]; then
    echo "Error: PORT environment variable is not set"
    exit 1
fi

if [ -z "$TOTAL_GPUS" ]; then
    echo "Error: TOTAL_GPUS environment variable is not set"
    exit 1
fi

if [ -z "$RANK" ]; then
    echo "Error: RANK environment variable is not set"
    exit 1
fi

if [ -z "$TOTAL_NODES" ]; then
    echo "Error: TOTAL_NODES environment variable is not set"
    exit 1
fi

# Construct command suffix for config dump
command_suffix=""
if [[ -n "${DUMP_CONFIG_PATH}" ]]; then command_suffix="--dump-config-to ${DUMP_CONFIG_PATH}"; fi

set -x
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800
export SGLANG_DG_CACHE_DIR="/configs/dg-10212025"
export FLASHINFER_WORKSPACE_BASE="/configs/flashinfer-cache"

DYN_SKIP_SGLANG_LOG_FORMATTING=1 \
MC_TE_METRIC=true \
SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True \
MC_FORCE_MNNVL=1 \
NCCL_MNNVL_ENABLE=1 \
NCCL_CUMEM_ENABLE=1 \
SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
PYTHONUNBUFFERED=1 \
python3 -m dynamo.sglang \
    --served-model-name deepseek-ai/DeepSeek-R1 \
    --model-path /model/ \
    --skip-tokenizer-init \
    --trust-remote-code \
    --dist-init-addr "$HOST_IP_MACHINE:$PORT" \
    --nnodes "$TOTAL_NODES" \
    --node-rank "$RANK" \
    --tp-size "$TOTAL_GPUS" \
    --dp-size "$TOTAL_GPUS" \
    --enable-dp-attention \
    --host 0.0.0.0 \
    --max-running-requests 30000 \
    --context-length 2200 \
    --disable-radix-cache \
    --moe-a2a-backend deepep \
    --load-balance-method round_robin \
    --deepep-mode normal \
    --ep-dispatch-algorithm dynamic \
    --moe-dense-tp-size 1 \
    --enable-dp-lm-head \
    --disable-shared-experts-fusion \
    --ep-num-redundant-experts 32 \
    --eplb-algorithm deepseek \
    --attention-backend trtllm_mla \
    --kv-cache-dtype fp8_e4m3 \
    --watchdog-timeout 1000000 \
    --disable-cuda-graph \
    --chunked-prefill-size 131072 \
    --max-total-tokens 524288 \
    --deepep-config /configs/deepep_config.json \
    --stream-interval 50 \
    --mem-fraction-static 0.75 ${command_suffix}


