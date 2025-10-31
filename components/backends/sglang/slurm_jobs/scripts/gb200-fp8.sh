#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Function to print usage
print_usage() {
    echo "Usage: $0 <mode>"
    echo "  mode: prefill or decode"
    echo ""
    echo "Examples:"
    echo "  $0 prefill"
    echo "  $0 decode"
    exit 1
}

# Check if correct number of arguments provided
if [ $# -ne 1 ]; then
    echo "Error: Expected 1 argument, got $#"
    print_usage
fi

# Parse arguments
mode=$1

# Validate mode argument
if [ "$mode" != "prefill" ] && [ "$mode" != "decode" ]; then
    echo "Error: mode must be 'prefill' or 'decode', got '$mode'"
    print_usage
fi

echo "Mode: $mode"
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

if [ -z "$USE_INIT_LOCATIONS" ]; then
    echo "Error: USE_INIT_LOCATIONS environment variable is not set"
    exit 1
fi

# Construct command based on mode
if [ "$mode" = "prefill" ]; then
    set -x
    export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800
    export SGL_DG_CACHE_DIR="/configs/dgcache/3p1dcache"

    command_suffix=""
    if [[ "${USE_INIT_LOCATIONS,,}" == "true" ]]; then command_suffix="--init-expert-location /configs/prefill_dsr1-0528_in1000out1000_num40000.json"; fi

    DYN_SKIP_SGLANG_LOG_FORMATTING=1 \
    MC_TE_METRIC=true \
    SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 \
    SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
    SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
    SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True \
    MC_FORCE_MNNVL=1 \
    NCCL_MNNVL_ENABLE=1 \
    NCCL_CUMEM_ENABLE=1 \
    SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
    SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    python3 -m dynamo.sglang \
        --served-model-name deepseek-ai/DeepSeek-R1 \
        --model-path /model/ \
        --skip-tokenizer-init \
        --trust-remote-code \
        --disaggregation-mode prefill \
        --dist-init-addr "$HOST_IP_MACHINE:$PORT" \
        --disaggregation-bootstrap-port 30001 \
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
        --attention-backend cutlass_mla \
        --watchdog-timeout 1000000 \
        --disable-cuda-graph \
        --chunked-prefill-size 131072 \
        --max-total-tokens 524288 \
        --deepep-config /configs/deepep_config.json \
        --stream-interval 50 \
        --mem-fraction-static 0.75 ${command_suffix}

elif [ "$mode" = "decode" ]; then
    set -x
    set -x
    export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800
    export SGL_DG_CACHE_DIR="/configs/dgcache/3p1dcache"

    command_suffix=""
    if [[ "${USE_INIT_LOCATIONS,,}" == "true" ]]; then command_suffix="--init-expert-location /configs/decode_dsr1-0528_loadgen_in1024out1024_num2000_2p12d.json"; fi

    DYN_SKIP_SGLANG_LOG_FORMATTING=1 \
    SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=768  \
    MC_TE_METRIC=true \
    SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 \
    SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
    SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
    SGLANG_DECODE_BOOTSTRAP_TIMEOUT=1000 \
    SGLANG_HACK_SEQ_BOOTSTRAP_ROOM=1 \
    SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True \
    MC_FORCE_MNNVL=1 \
    NCCL_MNNVL_ENABLE=1 \
    NCCL_CUMEM_ENABLE=1 \
    SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
    SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    python3 -m dynamo.sglang \
        --served-model-name deepseek-ai/DeepSeek-R1 \
        --model-path /model/ \
        --skip-tokenizer-init \
        --trust-remote-code \
        --disaggregation-mode decode \
        --dist-init-addr "$HOST_IP_MACHINE:$PORT" \
        --disaggregation-bootstrap-port 30001 \
        --nnodes "$TOTAL_NODES" \
        --node-rank "$RANK" \
        --tp-size "$TOTAL_GPUS" \
        --dp-size "$TOTAL_GPUS" \
        --enable-dp-attention \
        --host 0.0.0.0 \
        --decode-log-interval 1000 \
        --max-running-requests 45000 \
        --context-length 2200 \
        --disable-radix-cache \
        --moe-a2a-backend deepep \
        --prefill-round-robin-balance \
        --deepep-mode low_latency \
        --moe-dense-tp-size 1 \
        --enable-dp-lm-head \
        --cuda-graph-bs 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160 168 176 184 192 200 208 216 224 232 240 248 256 264 272 280 288 296 304 312 320 328 336 344 352 360 368 376 384 416 448 480 512 544 576 608 640 672 704 736 768 \
        --cuda-graph-max-bs 768 \
        --disable-shared-experts-fusion \
        --ep-num-redundant-experts 32 \
        --ep-dispatch-algorithm static \
        --eplb-algorithm deepseek \
        --attention-backend cutlass_mla \
        --watchdog-timeout 1000000 \
        --chunked-prefill-size 36864 \
        --stream-interval 50 \
        --deepep-config /configs/deepep_config.json \
        --mem-fraction-static 0.82 ${command_suffix}
fi
