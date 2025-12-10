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

if [ -z "$RUN_IN_CI" ]; then
    echo "Error: RUN_IN_CI environment variable is not set"
    exit 1
fi

# Construct command based on mode
if [ "$mode" = "prefill" ]; then
    set -x
    if [[ "${RUN_IN_CI,,}" == "true" ]]; then
        python3 -m pip install /configs/ai_dynamo_runtime-0.7.0-cp310-abi3-manylinux_2_28_aarch64.whl
        python3 -m pip install /configs/ai_dynamo-0.7.0-py3-none-any.whl
    fi
    export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800

    command_suffix=""
    if [[ -n "${DUMP_CONFIG_PATH}" ]]; then command_suffix="${command_suffix} --dump-config-to ${DUMP_CONFIG_PATH}"; fi

    PYTHONUNBUFFERED=1 \
    DYN_SKIP_SGLANG_LOG_FORMATTING=1 \
    SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
    SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
    SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 \
    SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
    SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
    SGLANG_DECODE_BOOTSTRAP_TIMEOUT=1000 \
    MC_FORCE_MNNVL=1 \
    NCCL_MNNVL_ENABLE=1 \
    NCCL_CUMEM_ENABLE=1 \
    SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True \
    SGLANG_ENABLE_JIT_DEEPGEMM=false \
    SGLANG_ENABLE_FLASHINFER_GEMM=true \
    python3 -m dynamo.sglang \
        --served-model-name deepseek-ai/DeepSeek-R1 \
        --model-path /model/ \
        --trust-remote-code \
        --disable-radix-cache \
        --kv-cache-dtype fp8_e4m3 \
        --attention-backend trtllm_mla \
        --quantization modelopt_fp4 \
        --moe-runner-backend flashinfer_trtllm \
        --stream-interval 50 \
        --watchdog-timeout 1000000 \
        --context-length 9600 \
        --mem-fraction-static 0.95 \
        --max-total-tokens 32768 \
        --chunked-prefill-size 24576 \
        --cuda-graph-max-bs 256 \
        --max-running-requests 512 \
        --scheduler-recv-interval 10 \
        --enable-symm-mem \
        --moe-dense-tp-size 1 \
        --load-balance-method round_robin \
        --disaggregation-bootstrap-port 30001 \
        --disaggregation-mode prefill \
        --dp-size 1 \
        --tp-size "$TOTAL_GPUS" \
        --ep-size 1 \
        --dist-init-addr "$HOST_IP_MACHINE:$PORT" \
        --nnodes "$TOTAL_NODES" \
        --node-rank "$RANK" \
        --host 0.0.0.0 ${command_suffix}

elif [ "$mode" = "decode" ]; then
    set -x
    if [[ "${RUN_IN_CI,,}" == "true" ]]; then
        python3 -m pip install /configs/ai_dynamo_runtime-0.7.0-cp310-abi3-manylinux_2_28_aarch64.whl
        python3 -m pip install /configs/ai_dynamo-0.7.0-py3-none-any.whl
    fi
    export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800

    command_suffix=""
    if [[ -n "${DUMP_CONFIG_PATH}" ]]; then command_suffix="${command_suffix} --dump-config-to ${DUMP_CONFIG_PATH}"; fi

    PYTHONUNBUFFERED=1 \
    DYN_SKIP_SGLANG_LOG_FORMATTING=1 \
    SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
    SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
    SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 \
    SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
    SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
    SGLANG_DECODE_BOOTSTRAP_TIMEOUT=1000 \
    MC_FORCE_MNNVL=1 \
    NCCL_MNNVL_ENABLE=1 \
    NCCL_CUMEM_ENABLE=1 \
    SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True \
    SGLANG_ENABLE_JIT_DEEPGEMM=false \
    SGLANG_ENABLE_FLASHINFER_GEMM=true \
    python3 -m dynamo.sglang \
        --served-model-name deepseek-ai/DeepSeek-R1 \
        --model-path /model/ \
        --prefill-round-robin-balance \
        --trust-remote-code \
        --disable-radix-cache \
        --kv-cache-dtype fp8_e4m3 \
        --attention-backend trtllm_mla \
        --quantization modelopt_fp4 \
        --moe-runner-backend flashinfer_trtllm \
        --disaggregation-bootstrap-port 30001 \
        --disaggregation-mode decode \
        --stream-interval 50 \
        --watchdog-timeout 1000000 \
        --context-length 9600 \
        --mem-fraction-static 0.95 \
        --chunked-prefill-size 8192 \
        --cuda-graph-max-bs 256 \
        --scheduler-recv-interval 10 \
        --enable-symm-mem \
        --moe-dense-tp-size 1 \
        --dp-size 1 \
        --tp-size "$TOTAL_GPUS" \
        --ep-size 1 \
        --dist-init-addr "$HOST_IP_MACHINE:$PORT" \
        --nnodes "$TOTAL_NODES" \
        --node-rank "$RANK" \
        --host 0.0.0.0 ${command_suffix}
fi

