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
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --num-nodes N         Number of nodes in the cluster (required, int)"
            echo "  --node-rank M         Rank of this node (0-based, required, int)"
            echo "  --gpus-per-node L     Number of GPUs per node (required, int)"
            echo "  --master-addr ADDR    Master node address (default: localhost)"
            echo "  --log-dir DIR         Directory for log files (default: ./logs)"
            echo "  --model MODEL         Model name to use (default: ${MODEL})"
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

trap 'echo Cleaning up...; kill 0' EXIT

# run ingress if it's node 0
if [ $NODE_RANK -eq 0 ]; then
    DYN_LOG=debug python -m dynamo.frontend --router-mode kv --http-port=8000 2>&1 | tee $LOG_DIR/dsr1_dep_ingress.log &
fi

mkdir -p $LOG_DIR

# Data Parallel Attention / Expert Parallelism
# Routing to DP workers managed by Dynamo
for ((i=0; i<GPUS_PER_NODE; i++)); do
    dp_rank=$((i + NODE_RANK * GPUS_PER_NODE))
    CUDA_VISIBLE_DEVICES=$i \
        VLLM_ALL2ALL_BACKEND="deepep_low_latency" \
        VLLM_USE_DEEP_GEMM=1 \
        VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1 \
        python3 -m dynamo.vllm \
        --model $MODEL \
        --data_parallel_size $DATA_PARALLEL_SIZE \
        --data-parallel-rank $dp_rank \
        --enable-expert-parallel \
        --max-model-len 10240 \
        --data-parallel-address $MASTER_ADDR \
        --data-parallel-rpc-port 13345 \
        --gpu-memory-utilization 0.95 \
        --enforce-eager 2>&1 | tee $LOG_DIR/dsr1_dep_${dp_rank}.log &
done

echo "All workers starting. (press Ctrl+C to stop)..."
wait
