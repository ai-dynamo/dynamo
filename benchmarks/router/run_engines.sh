#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Parse command-line arguments
NUM_WORKERS=8
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
TENSOR_PARALLEL_SIZE=1
USE_MOCKERS=false
USE_PREFILLS=false
BASE_GPU_OFFSET=0
EXTRA_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --tensor-parallel-size)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --mockers)
            USE_MOCKERS=true
            shift
            ;;
        --prefills)
            USE_PREFILLS=true
            shift
            ;;
        --base-gpu-offset)
            BASE_GPU_OFFSET="$2"
            shift 2
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            # Collect all other arguments as vLLM/mocker arguments
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# If no extra args provided, use defaults
if [ ${#EXTRA_ARGS[@]} -eq 0 ]; then
    if [ "$USE_MOCKERS" = true ]; then
        # Default args for mocker engine (only block-size needed as others are defaults)
        EXTRA_ARGS=(
            "--block-size" "64"
        )
    else
        # Default args for vLLM engine (explicitly include block-size)
        EXTRA_ARGS=(
            "--enforce-eager"
            "--max-num-batched-tokens" "16384"
            "--max-model-len" "32768"
            "--block-size" "64"
        )
    fi
fi

# Validate arguments
if ! [[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || [ "$NUM_WORKERS" -lt 1 ]; then
    echo "Error: NUM_WORKERS must be a positive integer"
    exit 1
fi

if ! [[ "$TENSOR_PARALLEL_SIZE" =~ ^[0-9]+$ ]] || [ "$TENSOR_PARALLEL_SIZE" -lt 1 ]; then
    echo "Error: TENSOR_PARALLEL_SIZE must be a positive integer"
    exit 1
fi

if ! [[ "$BASE_GPU_OFFSET" =~ ^[0-9]+$ ]]; then
    echo "Error: BASE_GPU_OFFSET must be a non-negative integer"
    exit 1
fi

# Calculate total GPUs needed
TOTAL_GPUS_NEEDED=$((NUM_WORKERS * TENSOR_PARALLEL_SIZE))
LAST_GPU=$((BASE_GPU_OFFSET + TOTAL_GPUS_NEEDED - 1))
echo "Configuration:"
echo "  Engine Type: $([ "$USE_MOCKERS" = true ] && echo "Mocker" || echo "vLLM")"
echo "  Worker Type: $([ "$USE_PREFILLS" = true ] && echo "Prefill" || echo "Decode")"
echo "  Workers: $NUM_WORKERS"
echo "  Model: $MODEL_PATH"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "  Total GPUs needed: $TOTAL_GPUS_NEEDED"
echo "  GPU Range: $BASE_GPU_OFFSET-$LAST_GPU"
echo "  Engine args: ${EXTRA_ARGS[*]}"
echo ""

PIDS=()

cleanup() {
    echo -e "\nStopping all workers..."
    kill "${PIDS[@]}" 2>/dev/null
    wait
    exit 0
}

trap cleanup SIGINT SIGTERM

WORKER_TYPE=$([ "$USE_PREFILLS" = true ] && echo "prefill" || echo "decode")
echo "Starting $NUM_WORKERS $WORKER_TYPE workers..."

for i in $(seq 1 $NUM_WORKERS); do
    {
        echo "[${WORKER_TYPE^} Worker-$i] Starting..."

        # Calculate GPU indices for this worker (with base offset)
        START_GPU=$(( BASE_GPU_OFFSET + (i - 1) * TENSOR_PARALLEL_SIZE ))
        END_GPU=$(( START_GPU + TENSOR_PARALLEL_SIZE - 1 ))

        # Build CUDA_VISIBLE_DEVICES string
        if [ "$TENSOR_PARALLEL_SIZE" -eq 1 ]; then
            GPU_DEVICES="$START_GPU"
        else
            GPU_DEVICES=""
            for gpu in $(seq $START_GPU $END_GPU); do
                if [ -n "$GPU_DEVICES" ]; then
                    GPU_DEVICES="${GPU_DEVICES},$gpu"
                else
                    GPU_DEVICES="$gpu"
                fi
            done
        fi

        if [ "$USE_MOCKERS" = true ]; then
            # Run mocker engine (no GPU assignment needed)
            exec python -m dynamo.mocker \
                --model-path "$MODEL_PATH" \
                --endpoint dyn://test.mocker.generate \
                "${EXTRA_ARGS[@]}"
        else
            echo "[${WORKER_TYPE^} Worker-$i] Using GPUs: $GPU_DEVICES"
            # Run vLLM engine with PYTHONHASHSEED=0 for deterministic event IDs in KV-aware routing
            VLLM_ARGS=()
            VLLM_ARGS+=("--model" "$MODEL_PATH")
            VLLM_ARGS+=("--tensor-parallel-size" "$TENSOR_PARALLEL_SIZE")
            if [ "$USE_PREFILLS" = true ]; then
                VLLM_ARGS+=("--is-prefill-worker")
            fi
            VLLM_ARGS+=("${EXTRA_ARGS[@]}")

            exec env PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=$GPU_DEVICES python -m dynamo.vllm \
                "${VLLM_ARGS[@]}"
        fi
    } &
    PIDS+=($!)
    echo "Started $WORKER_TYPE worker $i (PID: $!)"
done

echo "All workers started. Press Ctrl+C to stop."
wait
echo "All workers completed."