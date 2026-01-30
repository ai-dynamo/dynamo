#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN=1 \
    SGLANG_PER_TOKEN_GROUP_QUANT_8BIT_V2=1 \
    SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 \
    SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
    SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
    SGLANG_HACK_SEQ_BOOTSTRAP_ROOM=1 \
    MC_TE_METRIC=true \
    MC_FORCE_MNNVL=1 \
    NCCL_MNNVL_ENABLE=1 \
    NCCL_CUMEM_ENABLE=1 \
    SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True \
    SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
    SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
    python3 -m dynamo.sglang \
        --served-model-name deepseek-ai/DeepSeek-R1 \
        --model-path /model/ \
        --trust-remote-code \
        --kv-cache-dtype fp8_e4m3 \
        --attention-backend trtllm_mla \
        --quantization modelopt_fp4 \
        --moe-runner-backend flashinfer_cutlass \
        --disable-radix-cache \
        --disable-chunked-prefix-cache \
        --stream-interval 50 \
        --decode-log-interval 1000 \
        --watchdog-timeout 1000000 \
        --context-length 2176 \
        --disable-shared-experts-fusion \
        --eplb-algorithm deepseek \
        --disaggregation-bootstrap-port 30001 \
        --disaggregation-mode prefill \
        --mem-fraction-static 0.84 \
        --max-total-tokens 131072 \
        --max-prefill-tokens 32768 \
        --chunked-prefill-size 65536 \
        --enable-single-batch-overlap \
        --max-running-requests 30000 \
        --load-balance-method round_robin \
        --disable-cuda-graph \
        --enable-dp-attention \
        --tp-size "$TOTAL_GPUS" \
        --dp-size "$TOTAL_GPUS" \
        --ep-size "$TOTAL_GPUS" \
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
    SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN=1 \
    SGLANG_PER_TOKEN_GROUP_QUANT_8BIT_V2=1 \
    SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 \
    SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
    SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
    SGLANG_HACK_SEQ_BOOTSTRAP_ROOM=1 \
    MC_TE_METRIC=true \
    MC_FORCE_MNNVL=1 \
    NCCL_MNNVL_ENABLE=1 \
    NCCL_CUMEM_ENABLE=1 \
    SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True \
    SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
    SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
    SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024 \
    SGLANG_CUTEDSL_MOE_NVFP4_DISPATCH=1 \
    SGLANG_FLASHINFER_FP4_GEMM_BACKEND=cutlass \
    python3 -m dynamo.sglang \
        --served-model-name deepseek-ai/DeepSeek-R1 \
        --model-path /model/ \
        --trust-remote-code \
        --kv-cache-dtype fp8_e4m3 \
        --attention-backend trtllm_mla \
        --quantization modelopt_fp4 \
        --moe-runner-backend flashinfer_cutedsl \
        --disable-radix-cache \
        --disable-chunked-prefix-cache \
        --stream-interval 50 \
        --decode-log-interval 1000 \
        --watchdog-timeout 1000000 \
        --context-length 2176 \
        --disable-shared-experts-fusion \
        --eplb-algorithm deepseek \
        --disaggregation-bootstrap-port 30001 \
        --disaggregation-mode decode \
        --mem-fraction-static 0.83 \
        --max-total-tokens 3122380 \
        --chunked-prefill-size 786432 \
        --max-running-requests 67584 \
        --moe-a2a-backend deepep \
        --deepep-mode low_latency \
        --ep-dispatch-algorithm static \
        --ep-num-redundant-experts 32 \
        --cuda-graph-bs 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160 168 176 184 192 200 208 216 224 232 240 248 256 264 272 280 288 296 304 312 320 328 336 344 352 360 368 376 384 416 448 480 512 544 576 608 640 672 704 736 768 1024 \
        --num-reserved-decode-tokens 112 \
        --moe-dense-tp-size 1 \
        --enable-dp-lm-head \
        --prefill-round-robin-balance \
        --enable-dp-attention \
        --tp-size "$TOTAL_GPUS" \
        --dp-size "$TOTAL_GPUS" \
        --ep-size "$TOTAL_GPUS" \
        --dist-init-addr "$HOST_IP_MACHINE:$PORT" \
        --nnodes "$TOTAL_NODES" \
        --node-rank "$RANK" \
        --host 0.0.0.0 ${command_suffix}
fi