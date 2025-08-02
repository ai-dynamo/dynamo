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

# Construct command based on mode and cmd
if [ "$mode" = "prefill" ]; then
    if [ "$cmd" = "dynamo" ]; then
        # H100 dynamo prefill command
        python3 -m dynamo.sglang.worker \
            --model-path /model/ \
            --served-model-name deepseek-ai/DeepSeek-R1 \
            --skip-tokenizer-init \
            --disaggregation-mode prefill \
            --disaggregation-transfer-backend nixl \
            --disaggregation-bootstrap-port 30001 \
            --dist-init-addr "$HOST_IP:$PORT" \
            --nnodes "$TOTAL_NODES" \
            --node-rank "$RANK" \
            --tp-size "$TOTAL_GPUS" \
            --dp-size "$TOTAL_GPUS" \
            --enable-dp-attention \
            --decode-log-interval 1 \
            --enable-deepep-moe \
            --page-size 1 \
            --trust-remote-code \
            --moe-dense-tp-size 1 \
            --enable-dp-lm-head \
            --disable-radix-cache \
            --watchdog-timeout 1000000 \
            --enable-two-batch-overlap \
            --deepep-mode normal \
            --mem-fraction-static 0.85 \
            --deepep-config /configs/deepep.json \
            --ep-num-redundant-experts 32 \
            --ep-dispatch-algorithm dynamic \
            --eplb-algorithm deepseek
    elif [ "$cmd" = "sglang" ]; then
        # H100 sglang prefill command
        python3 -m sglang.launch_server \
            --model-path /model/ \
            --served-model-name deepseek-ai/DeepSeek-R1 \
            --disaggregation-transfer-backend nixl \
            --disaggregation-mode prefill \
            --dist-init-addr "$HOST_IP:$PORT" \
            --nnodes "$TOTAL_NODES" \
            --node-rank "$RANK" \
            --tp-size "$TOTAL_GPUS" \
            --dp-size "$TOTAL_GPUS" \
            --enable-dp-attention \
            --decode-log-interval 1 \
            --enable-deepep-moe \
            --page-size 1 \
            --host 0.0.0.0 \
            --trust-remote-code \
            --moe-dense-tp-size 1 \
            --enable-dp-lm-head \
            --disable-radix-cache \
            --watchdog-timeout 1000000 \
            --enable-two-batch-overlap \
            --deepep-mode normal \
            --mem-fraction-static 0.85 \
            --ep-num-redundant-experts 32 \
            --ep-dispatch-algorithm dynamic \
            --eplb-algorithm deepseek \
            --deepep-config /configs/deepep.json
    fi
elif [ "$mode" = "decode" ]; then
    if [ "$cmd" = "dynamo" ]; then
        # H100 dynamo decode command
        python3 -m dynamo.sglang.decode_worker \
            --model-path /model/ \
            --served-model-name deepseek-ai/DeepSeek-R1 \
            --skip-tokenizer-init \
            --disaggregation-mode decode \
            --disaggregation-transfer-backend nixl \
            --disaggregation-bootstrap-port 30001 \
            --dist-init-addr "$HOST_IP:$PORT" \
            --nnodes "$TOTAL_NODES" \
            --node-rank "$RANK" \
            --tp-size "$TOTAL_GPUS" \
            --dp-size "$TOTAL_GPUS" \
            --enable-dp-attention \
            --decode-log-interval 1 \
            --enable-deepep-moe \
            --page-size 1 \
            --trust-remote-code \
            --moe-dense-tp-size 1 \
            --enable-dp-lm-head \
            --disable-radix-cache \
            --watchdog-timeout 1000000 \
            --enable-two-batch-overlap \
            --deepep-mode low_latency \
            --mem-fraction-static 0.835 \
            --ep-num-redundant-experts 32 \
            --cuda-graph-bs 256
    elif [ "$cmd" = "sglang" ]; then
        # H100 sglang decode command
        python3 -m sglang.launch_server \
            --model-path /model/ \
            --disaggregation-transfer-backend nixl \
            --disaggregation-mode decode \
            --dist-init-addr "$HOST_IP:$PORT" \
            --nnodes "$TOTAL_NODES" \
            --node-rank "$RANK" \
            --tp-size "$TOTAL_GPUS" \
            --dp-size "$TOTAL_GPUS" \
            --enable-dp-attention \
            --decode-log-interval 1 \
            --enable-deepep-moe \
            --page-size 1 \
            --host 0.0.0.0 \
            --trust-remote-code \
            --moe-dense-tp-size 1 \
            --enable-dp-lm-head \
            --disable-radix-cache \
            --watchdog-timeout 1000000 \
            --enable-two-batch-overlap \
            --deepep-mode low_latency \
            --mem-fraction-static 0.835 \
            --ep-num-redundant-experts 32 \
            --cuda-graph-bs 256
    fi
fi


