#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This comes from https://github.com/sgl-project/sglang/issues/10903 and uses the low-prec prefill setup - seems like prefill is offloaded

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
    export SGLANG_DG_CACHE_DIR="/configs/deepgemm-kernels-10212025-ddcba74b"
    export FLASHINFER_WORKSPACE_BASE="/configs/flashinfer-cache"

    # temp we need to install newest cutedsl
    python3 -m pip install --no-cache-dir --upgrade --pre nvidia-cutlass-dsl

    # no expert locations collected for fp4 yet
    command_suffix=""
    if [[ "${USE_INIT_LOCATIONS,,}" == "true" ]]; then command_suffix=" "; fi
    if [[ -n "${DUMP_CONFIG_PATH}" ]]; then command_suffix="${command_suffix} --dump-config-to ${DUMP_CONFIG_PATH}"; fi

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
        --disaggregation-transfer-backend nixl \
        --served-model-name deepseek-ai/DeepSeek-R1 \
        --model-path /model/ \
        --skip-tokenizer-init \
        --disaggregation-mode prefill \
        --decode-log-interval 1000 \
        --max-running-requests 768 \
        --context-length 4224 \
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
        --offload-mode cpu  \
        --offload-group-size 2  \
        --offload-num-in-group 1 \
        --offload-prefetch-step 1 \
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
        --ep-size "$((TOTAL_GPUS - 2))" \
        --tp-size "$((TOTAL_GPUS - 2))" \
        --dp-size "$((TOTAL_GPUS - 2))" \
        --enable-dp-attention \
        --host 0.0.0.0 \
        --stream-interval 2 \
        --log-level debug ${command_suffix}

elif [ "$mode" = "decode" ]; then
    set -x
    command_suffix=""
    if [[ "${USE_INIT_LOCATIONS,,}" == "true" ]]; then command_suffix=" "; fi

    export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800
    export SGLANG_DG_CACHE_DIR="/configs/deepgemm-kernels-10212025-ddcba74b"
    export FLASHINFER_WORKSPACE_BASE="/configs/flashinfer-cache"

    # temp we need to install newest cutedsl
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
    SGLANG_CUTEDSL_MOE_NVFP4_DISPATCH=1 \
    SGLANG_FP4_GEMM_BACKEND=cutlass \
    SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=384 \
    DYN_SKIP_SGLANG_LOG_FORMATTING=1 \
    PYTHONUNBUFFERED=1 \
    python3 -m dynamo.sglang \
        --disaggregation-transfer-backend nixl \
        --served-model-name deepseek-ai/DeepSeek-R1 \
        --model-path /model/ \
        --skip-tokenizer-init \
        --trust-remote-code \
        --disaggregation-mode decode \
        --host 0.0.0.0 \
        --decode-log-interval 1 \
        --max-running-requests 18432 \
        --context-length 4224 \
        --disable-radix-cache \
        --disable-shared-experts-fusion \
        --watchdog-timeout 1000000 \
        --disable-chunked-prefix-cache \
        --attention-backend trtllm_mla \
        --kv-cache-dtype fp8_e4m3 \
        --enable-dp-attention \
        --chunked-prefill-size 1572864 \
        --mem-fraction-static 0.83 \
        --moe-a2a-backend deepep \
        --deepep-mode low_latency \
        --ep-dispatch-algorithm static \
        --cuda-graph-bs 384 \
        --num-reserved-decode-tokens 128 \
        --ep-num-redundant-experts 32 \
        --eplb-algorithm deepseek \
        --moe-dense-tp-size 1 \
        --enable-dp-lm-head \
        --prefill-round-robin-balance \
        --max-total-tokens 1703116 \
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
        --stream-interval 2 \
        --mem-fraction-static 0.83 ${command_suffix}
fi