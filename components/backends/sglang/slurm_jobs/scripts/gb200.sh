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
    # We are not using a init-expert-location file for e2e benchmarking
        # We also don't currently have a --deepep-config file for GB200
        # Need to increase --context-length to 10k for 8k1k benchmarking
        SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=2048 \
        MC_TE_METRIC=true \
        SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 \
        SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
        SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
        SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True \
        MC_FORCE_MNNVL=1 \
        NCCL_MNNVL_ENABLE=1 \
        NCCL_CUMEM_ENABLE=1 \
        SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
        SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
        PYTHONUNBUFFERED=1 \
        python3 -m dynamo.sglang.worker \
            --served-model-name deepseek-ai/DeepSeek-R1 \
            --model-path /model/ \
            --skip-tokenizer-init \
            --trust-remote-code \
            --disaggregation-mode prefill \
            --dist-init-addr "$HOST_IP:$PORT" \
            --disaggregation-bootstrap-port 30001 \
            --nnodes "$TOTAL_NODES" \
            --node-rank "$RANK" \
            --tp-size "$TOTAL_GPUS" \
            --dp-size "$TOTAL_GPUS" \
            --enable-dp-attention \
            --host 0.0.0.0 \
            --decode-log-interval 1 \
            --max-running-requests 6144 \
            --context-length 2716 \
            --disable-radix-cache \
            --enable-deepep-moe \
            --deepep-mode low_latency \
            --moe-dense-tp-size 1 \
            --enable-dp-lm-head \
            --disable-shared-experts-fusion \
            --ep-num-redundant-experts 32 \
            --ep-dispatch-algorithm static \
            --eplb-algorithm deepseek \
            --attention-backend cutlass_mla \
            --watchdog-timeout 1000000 \
            --disable-cuda-graph \
            --chunked-prefill-size 16384 \
            --max-total-tokens 32768 \
            --mem-fraction-static 0.8 \
            --log-level debug

    elif [ "$cmd" = "sglang" ]; then
        # GB200 sglang prefill command
        # We are not using a init-expert-location file for e2e benchmarking
        # We also don't currently have a --deepep-config file for GB200
        # Need to increase --context-length to 10k for 8k1k benchmarking
        SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=2048 \
        MC_TE_METRIC=true \
        SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 \
        SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
        SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
        SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True \
        NCCL_MNNVL_ENABLE=1 \
        MC_FORCE_MNNVL=1 \
        NCCL_CUMEM_ENABLE=1 \
        SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
        SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
        PYTHONUNBUFFERED=1 \
        python3 -m sglang.launch_server \
            --served-model-name deepseek-ai/DeepSeek-R1 \
            --model-path /model/ \
            --trust-remote-code \
            --disaggregation-mode prefill \
            --dist-init-addr "$HOST_IP:$PORT" \
            --disaggregation-bootstrap-port 30001 \
            --nnodes "$TOTAL_NODES" \
            --node-rank "$RANK" \
            --tp-size "$TOTAL_GPUS" \
            --dp-size "$TOTAL_GPUS" \
            --enable-dp-attention \
            --host 0.0.0.0 \
            --decode-log-interval 1 \
            --max-running-requests 6144 \
            --context-length 2716 \
            --disable-radix-cache \
            --enable-deepep-moe \
            --deepep-mode low_latency \
            --moe-dense-tp-size 1 \
            --enable-dp-lm-head \
            --disable-shared-experts-fusion \
            --ep-num-redundant-experts 32 \
            --ep-dispatch-algorithm static \
            --eplb-algorithm deepseek \
            --attention-backend cutlass_mla \
            --watchdog-timeout 1000000 \
            --disable-cuda-graph \
            --chunked-prefill-size 16384 \
            --max-total-tokens 32768 \
            --mem-fraction-static 0.8 \
            --log-level debug
    fi
elif [ "$mode" = "decode" ]; then
    if [ "$cmd" = "dynamo" ]; then
        # Need to increase --context-length to 10k for 8k1k benchmarking
        # We are not using a init-expert-location file for e2e benchmarking
        SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=768 \
        MC_TE_METRIC=true \
        SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 \
        SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
        SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
        SGLANG_HACK_SEQ_BOOTSTRAP_ROOM=1 \
        SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True \
        NCCL_MNNVL_ENABLE=1 \
        MC_FORCE_MNNVL=1 \
        NCCL_CUMEM_ENABLE=1 \
        SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
        SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
        PYTHONUNBUFFERED=1 \
        python3 -m dynamo.sglang.decode_worker \
            --served-model-name deepseek-ai/DeepSeek-R1 \
            --model-path /model/ \
            --skip-tokenizer-init \
            --trust-remote-code \
            --disaggregation-mode decode \
            --dist-init-addr "$HOST_IP:$PORT" \
            --disaggregation-bootstrap-port 30001 \
            --nnodes "$TOTAL_NODES" \
            --node-rank "$RANK" \
            --tp-size "$TOTAL_GPUS" \
            --dp-size "$TOTAL_GPUS" \
            --enable-dp-attention \
            --host 0.0.0.0 \
            --decode-log-interval 1 \
            --max-running-requests 36864 \
            --context-length 2716 \
            --disable-radix-cache \
            --enable-deepep-moe \
            --deepep-mode low_latency \
            --moe-dense-tp-size 1 \
            --enable-dp-lm-head \
            --cuda-graph-bs 768 \
            --disable-shared-experts-fusion \
            --ep-num-redundant-experts 32 \
            --ep-dispatch-algorithm static \
            --eplb-algorithm deepseek \
            --attention-backend cutlass_mla \
            --watchdog-timeout 1000000 \
            --chunked-prefill-size 36864 \
            --mem-fraction-static 0.82 \
            --log-level debug

    elif [ "$cmd" = "sglang" ]; then
        # GB200 sglang decode command
        # Need to increase --context-length to 10k for 8k1k benchmarking
        # We are not using a init-expert-location file for e2e benchmarking
        SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=768 \
        MC_TE_METRIC=true \
        SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 \
        SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
        SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
        SGLANG_HACK_SEQ_BOOTSTRAP_ROOM=1 \
        SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True \
        NCCL_MNNVL_ENABLE=1 \
        MC_FORCE_MNNVL=1 \
        NCCL_CUMEM_ENABLE=1 \
        SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
        SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
        PYTHONUNBUFFERED=1 \
        python3 -m sglang.launch_server \
            --model-path /model/ \
            --trust-remote-code \
            --disaggregation-mode decode \
            --dist-init-addr "$HOST_IP:$PORT" \
            --disaggregation-bootstrap-port 30001 \
            --nnodes "$TOTAL_NODES" \
            --node-rank "$RANK" \
            --tp-size "$TOTAL_GPUS" \
            --dp-size "$TOTAL_GPUS" \
            --enable-dp-attention \
            --host 0.0.0.0 \
            --decode-log-interval 1 \
            --max-running-requests 36864 \
            --context-length 2716 \
            --disable-radix-cache \
            --enable-deepep-moe \
            --deepep-mode low_latency \
            --moe-dense-tp-size 1 \
            --enable-dp-lm-head \
            --cuda-graph-bs 768 \
            --disable-shared-experts-fusion \
            --ep-num-redundant-experts 32 \
            --ep-dispatch-algorithm static \
            --eplb-algorithm deepseek \
            --attention-backend cutlass_mla \
            --watchdog-timeout 1000000 \
            --chunked-prefill-size 36864 \
            --mem-fraction-static 0.82 \
            --log-level debug
    fi
fi
