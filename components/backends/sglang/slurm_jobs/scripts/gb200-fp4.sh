#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Function to print usage
print_usage() {
    echo "Usage: $0 <mode> <cmd>"
    echo "  mode: prefill or decode"
    echo "  cmd:  dynamo or sglang"
    echo ""
    echo "Examples:"
    echo "  $0 prefill dynamo"
    echo "  $0 decode sglang"
    exit 1
}

# Check if correct number of arguments provided
if [ $# -ne 2 ]; then
    echo "Error: Expected 2 arguments, got $#"
    print_usage
fi

# Parse arguments
mode=$1
cmd=$2

# Validate mode argument
if [ "$mode" != "prefill" ] && [ "$mode" != "decode" ]; then
    echo "Error: mode must be 'prefill' or 'decode', got '$mode'"
    print_usage
fi

# Validate cmd argument
if [ "$cmd" != "dynamo" ] && [ "$cmd" != "sglang" ]; then
    echo "Error: cmd must be 'dynamo' or 'sglang', got '$cmd'"
    print_usage
fi

echo "Mode: $mode"
echo "Command: $cmd"


# Check if required environment variables are set
if [ -z "$HOST_IP" ]; then
    echo "Error: HOST_IP environment variable is not set"
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

# TODO: since the args for sglang and dynamo are the same, we can be a bit cleaner here

# Construct command based on mode and cmd
if [ "$mode" = "prefill" ]; then
    if [ "$cmd" = "dynamo" ]; then
        echo "Error: gb200-fp4.sh prefill dynamo is not implemented"
        exit 1
    elif [ "$cmd" = "sglang" ]; then
        # GB200 sglang prefill command
        MC_TE_METRIC=true \
        MC_FORCE_MNNVL=1 \
        NCCL_MNNVL_ENABLE=1 \
        NCCL_CUMEM_ENABLE=1 \
        SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
        SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
        PYTHONUNBUFFERED=1 \
        python3 -m sglang.launch_server \
            --disaggregation-transfer-backend nixl \
            --disaggregation-mode prefill \
            --dist-init-addr "$HOST_IP:$PORT" \
            --disaggregation-bootstrap-port 30001 \
            --nnodes "$TOTAL_NODES" \
            --node-rank "$RANK" \
            --tp-size "$TOTAL_GPUS" \
            --dp-size "$TOTAL_GPUS" \
            --host 0.0.0.0 \
            --decode-log-interval 1 \
            --max-running-requests 1536 \
            --context-length 4224 \
            --disable-radix-cache \
            --disable-shared-experts-fusion \
            --attention-backend cutlass_mla \
            --watchdog-timeout 1000000 \
            --model-path /model/ \
            --served-model-name nvidia/DeepSeek-R1-0528-FP4 \
            --trust-remote-code \
            --enable-dp-attention \
            --disable-cuda-graph \
            --chunked-prefill-size 32768 \
            --max-total-tokens 131072 \
            --port 30000 \
            --max-prefill-tokens 32768 \
            --quantization modelopt_fp4 \
            --enable-flashinfer-trtllm-moe \
            --enable-ep-moe
    fi
elif [ "$mode" = "decode" ]; then
    if [ "$cmd" = "dynamo" ]; then
        echo "Error: gb200-fp4.sh decode dynamo is not implemented"
        exit 1
    elif [ "$cmd" = "sglang" ]; then
        # GB200 sglang decode command
        MC_TE_METRIC=1 \
        NCCL_MNNVL_ENABLE=1 \
        MC_FORCE_MNNVL=1 \
        NCCL_CUMEM_ENABLE=1 \
        SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
        PYTHONUNBUFFERED=1 \
        python3 -m sglang.launch_server \
            --disaggregation-transfer-backend nixl \
            --disaggregation-mode decode \
            --host 0.0.0.0 \
            --decode-log-interval 1 \
            --max-running-requests 1536 \
            --context-length 4224 \
            --max-total-tokens=2048 \
            --disable-radix-cache \
            --disable-shared-experts-fusion \
            --attention-backend cutlass_mla \
            --watchdog-timeout 1000000 \
            --model-path /model/ \
            --served-model-name nvidia/DeepSeek-R1-0528-FP4 \
            --trust-remote-code \
            --dist-init-addr "$HOST_IP:$PORT" \
            --disaggregation-bootstrap-port 30001 \
            --tp-size "$TOTAL_GPUS" \
            --dp-size "$TOTAL_GPUS" \
            --nnodes "$TOTAL_NODES" \
            --node-rank "$RANK" \
            --enable-dp-attention \
            --cuda-graph-bs 64 \
            --port 30000 \
            --quantization modelopt_fp4 \
            --enable-flashinfer-trtllm-moe \
            --enable-ep-moe
    fi
fi
