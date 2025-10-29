#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -ex

# Default values
NUM_NODES=""
NODE_RANK=""
GPUS_PER_NODE=""
MASTER_ADDR="localhost"
LOG_DIR="./logs"
MODEL="deepseek-ai/DeepSeek-R1"
SERVED_MODEL_NAME="deepseek-ai/DeepSeek-R1"
IS_PREFILL_WORKER="false"
# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-nodes)
            NUM_NODES="$2"
            shift 2
            ;;
        --node-rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --gpus-per-node)
            GPUS_PER_NODE="$2"
            shift 2
            ;;
        --master-addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --served-model-name)
            SERVED_MODEL_NAME="$2"
            shift 2
            ;;
        --is-prefill-worker)
            IS_PREFILL_WORKER=true
            shift 1
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --num-nodes N         Number of nodes in the cluster (required, int)"
            echo "  --node-rank M         Rank of this node (0-based, required, int)"
            echo "  --gpus-per-node L     Number of GPUs per node (required, int)"
            echo "  --master-addr ADDR    Master node address (default: localhost)"
            echo "  --log-dir DIR         Directory for log files (default: ./logs)"
            echo "  --model MODEL         Model name to use (default: ${MODEL})"
            echo "  --served-model-name SERVED_MODEL_NAME         Served model name to use (default: ${SERVED_MODEL_NAME})"
            echo "  --is-prefill-worker   Mark this worker as a prefill worker (flag)"
            echo "  -h, --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$NUM_NODES" ] || [ -z "$NODE_RANK" ] || [ -z "$GPUS_PER_NODE" ]; then
    echo "Error: Missing required arguments"
    echo "Required: --num-nodes, --node-rank, --gpus-per-node"
    echo "Use --help for usage information"
    exit 1
fi

# Calculate data parallel size
DATA_PARALLEL_SIZE=$((NUM_NODES * GPUS_PER_NODE))

echo "Configuration:"
echo "  Number of nodes: $NUM_NODES"
echo "  Node rank: $NODE_RANK"
echo "  GPUs per node: $GPUS_PER_NODE"
echo "  Data parallel size: $DATA_PARALLEL_SIZE"
echo "  Master address: $MASTER_ADDR"
echo "  Log directory: $LOG_DIR"
echo "  Model name: $MODEL"
echo "  Served model name: $SERVED_MODEL_NAME"
echo "  Is prefill worker: $IS_PREFILL_WORKER"

cleanup() {
    echo "Cleaning up..."
    set +e
    # Terminate background jobs started by this script
    jobs -p | xargs -r kill 2>/dev/null || true
    # Also terminate any direct child processes of this script
    pkill -P $$ 2>/dev/null || true
    # Give processes a moment to exit gracefully, then force kill leftovers
    sleep 1
    jobs -p | xargs -r kill -9 2>/dev/null || true
}

trap 'cleanup' EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

# run ingress if it's node 0 and is not a prefill worker
if [ "$NODE_RANK" -eq 0 ] && [ "$IS_PREFILL_WORKER" = "false" ]; then
    python -m dynamo.frontend --router-mode kv --http-port=8000 2>&1 | tee $LOG_DIR/dsr1_dep_ingress.log &
fi

mkdir -p $LOG_DIR


export GLOO_SOCKET_IFNAME=eth3 # this has to be non-IB network interface
export NCCL_SOCKET_IFNAME=eth3
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=eth3
export NVIDIA_GDRCOPY=enabled
export NVSHMEM_REMOTE_TRANSPORT=ibgda
export NVSHMEM_IB_ENABLE_IBGDA="true"
export VLLM_USE_DEEP_GEMM=1
export VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1
export VLLM_SKIP_P2P_CHECK=1

# Data Parallel Attention / Expert Parallelism
# Routing to DP workers managed by Dynamo
for ((i=0; i<GPUS_PER_NODE; i++)); do
    dp_rank=$((i + NODE_RANK * GPUS_PER_NODE))
    if [ "$IS_PREFILL_WORKER" = "true" ]; then
        CUDA_VISIBLE_DEVICES=$i \
            VLLM_ALL2ALL_BACKEND=deepep_high_throughput \
            python3 -m dynamo.vllm \
            --model $MODEL \
            --served-model-name $SERVED_MODEL_NAME \
            --tensor-parallel-size 1 \
            --data_parallel_size $DATA_PARALLEL_SIZE \
            --data-parallel-address $MASTER_ADDR \
            --data-parallel-rpc-port 13345 \
            --data-parallel-rank $dp_rank \
            --max-model-len 1048 \
            --gpu-memory-utilization 0.8 \
            --enable-expert-parallel \
            --data-parallel-hybrid-lb \
            --async-scheduling \
            --enable-dbo \
            --dbo-decode-token-threshold 32 \
            --enable-eplb \
            --eplb-config '{"window_size":"1000",
                            "step_interval":"3000",
                            "num_redundant_experts":"32",
                            "log_balancedness":"False"}' \
            --is-prefill-worker 2>&1 | tee $LOG_DIR/dsr1_dep_${dp_rank}_prefill.log &
    else
        CUDA_VISIBLE_DEVICES=$i \
            VLLM_ALL2ALL_BACKEND=deepep_low_latency \
            python3 -m dynamo.vllm \
            --model $MODEL \
            --served-model-name $SERVED_MODEL_NAME \
            --tensor-parallel-size 1 \
            --data_parallel_size $DATA_PARALLEL_SIZE \
            --data-parallel-address $MASTER_ADDR \
            --data-parallel-rpc-port 13345 \
            --data-parallel-rank $dp_rank \
            --max-model-len 2560 \
            --gpu-memory-utilization 0.85 \
            --enable-expert-parallel \
            --data-parallel-hybrid-lb \
            --async-scheduling \
            --enable-dbo \
            --dbo-decode-token-threshold 32 \
            --enable-eplb \
            --eplb-config '{"window_size":"1000",
                            "step_interval":"3000",
                            "num_redundant_experts":"32",
                            "log_balancedness":"False"}' \
            --compilation_config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' 2>&1 | tee $LOG_DIR/dsr1_dep_${dp_rank}_decode.log &
        fi
done

echo "All workers starting. (press Ctrl+C to stop)..."
wait
