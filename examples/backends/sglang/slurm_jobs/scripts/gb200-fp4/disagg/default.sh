#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This comes from https://github.com/sgl-project/sglang/issues/10903 and uses the low-prec decode setup

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
    # no expert locations collected for fp4 yet
    command_suffix=""
    if [[ "${USE_INIT_LOCATIONS,,}" == "true" ]]; then command_suffix=" "; fi
    if [[ -n "${DUMP_CONFIG_PATH}" ]]; then command_suffix="${command_suffix} --dump-config-to ${DUMP_CONFIG_PATH}"; fi

    # we have to install pre-release cutedsl for a integer overflow fix
    python3 -m pip install --no-cache-dir --upgrade --pre nvidia-cutlass-dsl

    # set your own cache variables here
    export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800
    export SGLANG_DG_CACHE_DIR="/configs/dg-10212025"
    export FLASHINFER_WORKSPACE_BASE="/configs/flashinfer-cache"

    DYN_SKIP_SGLANG_LOG_FORMATTING=1 \
    SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN=1 \
    SGLANG_PER_TOKEN_GROUP_QUANT_8BIT_V2=1 \
    SGL_JIT_DEEPGEMM_PRECOMPILE=0 \
    SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 \
    SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
    SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
    SGLANG_HACK_SEQ_BOOTSTRAP_ROOM=1 \
    MC_TE_METRIC=true \
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
        --disaggregation-mode prefill \
        --decode-log-interval 1000 \
        --max-running-requests 5632 \
        --context-length 2176 \
        --disable-radix-cache \
        --disable-shared-experts-fusion \
        --watchdog-timeout 1000000 \
        --disable-chunked-prefix-cache \
        --attention-backend trtllm_mla \
        --kv-cache-dtype fp8_e4m3 \
        --enable-single-batch-overlap \
        --chunked-prefill-size 65536 \
        --eplb-algorithm deepseek \
        --trust-remote-code \
        --disable-cuda-graph \
        --mem-fraction-static 0.84 \
        --max-total-tokens 131072 \
        --max-prefill-tokens 16384 \
        --load-balance-method round_robin \
        --quantization modelopt_fp4 \
        --moe-runner-backend flashinfer_cutlass \
        --dist-init-addr "$HOST_IP_MACHINE:$PORT" \
        --disaggregation-bootstrap-port 30001 \
        --nnodes "$TOTAL_NODES" \
        --node-rank "$RANK" \
        --ep-size "$TOTAL_GPUS" \
        --tp-size "$TOTAL_GPUS" \
        --dp-size "$TOTAL_GPUS" \
        --enable-dp-attention \
        --host 0.0.0.0 \
        --stream-interval 50 \
        --log-level debug ${command_suffix}

# For now we must keep SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK and cuda-graph-bs at 1024 until
# DeepEP merges in https://github.com/deepseek-ai/DeepEP/pull/440
# the nvidia-cutlass-dsl install fixes https://github.com/flashinfer-ai/flashinfer/issues/1830#issuecomment-3380074018
# which was previously limiting us to DISPATCH_TOKENS and cuda-graph-bs == 384
# For now use 12 nodes for fp4 since flashinfer_cutedsl requires experts per gpu < 8
# We have 288 (256 + 32 redundant) => 288/48 = 6

elif [ "$mode" = "decode" ]; then
    set -x
    # no expert locations collected for fp4 yet
    command_suffix=""
    if [[ "${USE_INIT_LOCATIONS,,}" == "true" ]]; then command_suffix=" "; fi
    if [[ -n "${DUMP_CONFIG_PATH}" ]]; then command_suffix="${command_suffix} --dump-config-to ${DUMP_CONFIG_PATH}"; fi

    # set your own cache variables here
    export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800
    export SGLANG_DG_CACHE_DIR="/configs/dg-10212025"
    export FLASHINFER_WORKSPACE_BASE="/configs/flashinfer-cache"

    # we have to install pre-release cutedsl for a integer overflow fix
    python3 -m pip install --no-cache-dir --upgrade --pre nvidia-cutlass-dsl

    SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN=1 \
    SGLANG_PER_TOKEN_GROUP_QUANT_8BIT_V2=1 \
    SGL_JIT_DEEPGEMM_PRECOMPILE=0 \
    MC_TE_METRIC=true \
    SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 \
    SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
    SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
    SGLANG_HACK_SEQ_BOOTSTRAP_ROOM=1 \
    SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True \
    SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=384 \
    SGLANG_CUTEDSL_MOE_NVFP4_DISPATCH=1 \
    SGLANG_FP4_GEMM_BACKEND=cutlass \
    DYN_SKIP_SGLANG_LOG_FORMATTING=1 \
    PYTHONUNBUFFERED=1 \
    python3 -m dynamo.sglang \
        --served-model-name deepseek-ai/DeepSeek-R1 \
        --model-path /model/ \
        --skip-tokenizer-init \
        --trust-remote-code \
        --disaggregation-mode decode \
        --host 0.0.0.0 \
        --decode-log-interval 1 \
        --max-running-requests 67584 \
        --context-length 2176 \
        --disable-radix-cache \
        --disable-shared-experts-fusion \
        --watchdog-timeout 1000000 \
        --disable-chunked-prefix-cache \
        --attention-backend trtllm_mla \
        --kv-cache-dtype fp8_e4m3 \
        --enable-dp-attention \
        --chunked-prefill-size 786432 \
        --mem-fraction-static 0.83 \
        --moe-a2a-backend deepep \
        --deepep-mode low_latency \
        --ep-dispatch-algorithm static \
        --cuda-graph-bs 384 \
        --num-reserved-decode-tokens 112 \
        --ep-num-redundant-experts 32 \
        --eplb-algorithm deepseek \
        --moe-dense-tp-size 1 \
        --enable-dp-lm-head \
        --prefill-round-robin-balance \
        --max-total-tokens 3122380 \
        --quantization modelopt_fp4 \
        --moe-runner-backend flashinfer_cutedsl \
        --dist-init-addr "$HOST_IP_MACHINE:$PORT" \
        --disaggregation-bootstrap-port 30001 \
        --nnodes "$TOTAL_NODES" \
        --node-rank "$RANK" \
        --tp-size "$TOTAL_GPUS" \
        --ep-size "$TOTAL_GPUS" \
        --dp-size "$TOTAL_GPUS" \
        --enable-single-batch-overlap \
        --enable-dp-attention \
        --stream-interval 50 \
        --mem-fraction-static 0.82 ${command_suffix}
fi
