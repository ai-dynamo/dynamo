#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Parse command-line arguments
export DYNAMO_HOME=${DYNAMO_HOME:-"/workspace"}
NUM_WORKERS=8
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
ENGINE_CONFIG_PATH="$DYNAMO_HOME/examples/backends/trtllm/engine_configs/deepseek-r1-distill-llama-8b"
TENSOR_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=1
USE_MOCKERS=false
USE_TRTLLM=false
MODE="agg"  # Options: agg (default), decode, prefill
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
        --data-parallel-size)
            DATA_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --mockers)
            USE_MOCKERS=true
            shift
            ;;
        --trtllm)
            USE_TRTLLM=true
            shift
            ;;
        --prefill)
            MODE="prefill"
            shift
            ;;
        --decode)
            MODE="decode"
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
            # Collect all other arguments as vLLM/mocker/trtllm arguments
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Validate that only one engine type is selected
ENGINE_COUNT=0
[ "$USE_MOCKERS" = true ] && ((ENGINE_COUNT++))
[ "$USE_TRTLLM" = true ] && ((ENGINE_COUNT++))
if [ "$ENGINE_COUNT" -gt 1 ]; then
    echo "Error: Only one engine type (--mockers, --trtllm, or default vLLM) can be specified"
    exit 1
fi

# If no extra args provided, use defaults
if [ ${#EXTRA_ARGS[@]} -eq 0 ]; then
    if [ "$USE_MOCKERS" = true ]; then
        # Default args for mocker engine (only block-size needed as others are defaults)
        EXTRA_ARGS=(
            "--block-size" "64"
        )
    elif [ "$USE_TRTLLM" = true ]; then
        # Default args for TensorRT-LLM engine using predefined YAML configs
        # Config files located at: $ENGINE_CONFIG_PATH/{agg,decode,prefill}.yaml
        if [ "$MODE" = "prefill" ]; then
            ENGINE_CONFIG="$ENGINE_CONFIG_PATH/prefill.yaml"
        elif [ "$MODE" = "decode" ]; then
            ENGINE_CONFIG="$ENGINE_CONFIG_PATH/decode.yaml"
        else
            ENGINE_CONFIG="$ENGINE_CONFIG_PATH/agg.yaml"
        fi

        EXTRA_ARGS=(
            "--extra-engine-args" "$ENGINE_CONFIG"
            "--publish-events-and-metrics"
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

if ! [[ "$DATA_PARALLEL_SIZE" =~ ^[0-9]+$ ]] || [ "$DATA_PARALLEL_SIZE" -lt 1 ]; then
    echo "Error: DATA_PARALLEL_SIZE must be a positive integer"
    exit 1
fi

if ! [[ "$BASE_GPU_OFFSET" =~ ^[0-9]+$ ]]; then
    echo "Error: BASE_GPU_OFFSET must be a non-negative integer"
    exit 1
fi

# Calculate total GPUs needed (TP * DP per worker)
GPUS_PER_WORKER=$((TENSOR_PARALLEL_SIZE * DATA_PARALLEL_SIZE))
TOTAL_GPUS_NEEDED=$((NUM_WORKERS * GPUS_PER_WORKER))
LAST_GPU=$((BASE_GPU_OFFSET + TOTAL_GPUS_NEEDED - 1))
echo "Configuration:"
if [ "$USE_MOCKERS" = true ]; then
    ENGINE_TYPE="Mocker"
elif [ "$USE_TRTLLM" = true ]; then
    ENGINE_TYPE="TensorRT-LLM"
else
    ENGINE_TYPE="vLLM"
fi
echo "  Engine Type: $ENGINE_TYPE"
echo "  Mode: $MODE"
echo "  Workers: $NUM_WORKERS"
echo "  Model: $MODEL_PATH"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "  Data Parallel Size: $DATA_PARALLEL_SIZE"
echo "  GPUs per worker: $GPUS_PER_WORKER"
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

echo "Starting $NUM_WORKERS $MODE workers..."

if [ "$USE_MOCKERS" = true ]; then
    # For mockers, launch a single process with --num-workers
    # All workers share the same tokio runtime and thread pool
    MODE_CAPITALIZED=$(echo "$MODE" | sed 's/\(.\)/\U\1/')
    echo "[$MODE_CAPITALIZED Mocker] Starting $NUM_WORKERS workers in single process..."

    MOCKER_ARGS=()
    MOCKER_ARGS+=("--model-path" "$MODEL_PATH")
    MOCKER_ARGS+=("--num-workers" "$NUM_WORKERS")

    # Set endpoint based on worker mode
    if [ "$MODE" = "prefill" ]; then
        MOCKER_ARGS+=("--endpoint" "dyn://test.prefill.generate")
        MOCKER_ARGS+=("--is-prefill-worker")
    elif [ "$MODE" = "decode" ]; then
        MOCKER_ARGS+=("--endpoint" "dyn://test.mocker.generate")
        MOCKER_ARGS+=("--is-decode-worker")
    else
        MOCKER_ARGS+=("--endpoint" "dyn://test.mocker.generate")
    fi

    if [ "$DATA_PARALLEL_SIZE" -gt 1 ]; then
        MOCKER_ARGS+=("--data-parallel-size" "$DATA_PARALLEL_SIZE")
    fi
    MOCKER_ARGS+=("${EXTRA_ARGS[@]}")

    python -m dynamo.mocker "${MOCKER_ARGS[@]}" &
    PIDS+=($!)
    echo "Started mocker with $NUM_WORKERS workers (PID: $!)"
else
    # For vLLM and TensorRT-LLM, use the original loop to launch separate processes
    for i in $(seq 1 $NUM_WORKERS); do
        {
            MODE_CAPITALIZED=$(echo "$MODE" | sed 's/\(.\)/\U\1/')
            echo "[$MODE_CAPITALIZED Worker-$i] Starting..."

            # Calculate GPU indices for this worker (with base offset)
            # Each worker needs TP * DP GPUs
            START_GPU=$(( BASE_GPU_OFFSET + (i - 1) * GPUS_PER_WORKER ))
            END_GPU=$(( START_GPU + GPUS_PER_WORKER - 1 ))

            # Build CUDA_VISIBLE_DEVICES string for all GPUs (TP * DP)
            if [ "$GPUS_PER_WORKER" -eq 1 ]; then
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

            if [ "$USE_TRTLLM" = true ]; then
                echo "[$MODE_CAPITALIZED Worker-$i] Using GPUs: $GPU_DEVICES"
                # Run TensorRT-LLM engine
                TRTLLM_ARGS=()
                TRTLLM_ARGS+=("--model-path" "$MODEL_PATH")
                TRTLLM_ARGS+=("--tensor-parallel-size" "$TENSOR_PARALLEL_SIZE")
                if [ "$MODE" != "agg" ]; then
                    TRTLLM_ARGS+=("--disaggregation-mode" "$MODE")
                fi
                TRTLLM_ARGS+=("${EXTRA_ARGS[@]}")

                exec env CUDA_VISIBLE_DEVICES=$GPU_DEVICES trtllm-llmapi-launch python3 -m dynamo.trtllm \
                    "${TRTLLM_ARGS[@]}"
            else
                echo "[$MODE_CAPITALIZED Worker-$i] Using GPUs: $GPU_DEVICES"
                # Run vLLM engine with PYTHONHASHSEED=0 for deterministic event IDs in KV-aware routing
                VLLM_ARGS=()
                VLLM_ARGS+=("--model" "$MODEL_PATH")
                VLLM_ARGS+=("--tensor-parallel-size" "$TENSOR_PARALLEL_SIZE")
                if [ "$DATA_PARALLEL_SIZE" -gt 1 ]; then
                    VLLM_ARGS+=("--data-parallel-size" "$DATA_PARALLEL_SIZE")
                fi
                if [ "$MODE" = "prefill" ]; then
                    VLLM_ARGS+=("--is-prefill-worker")
                elif [ "$MODE" = "decode" ]; then
                    VLLM_ARGS+=("--is-decode-worker")
                fi
                VLLM_ARGS+=("${EXTRA_ARGS[@]}")

                exec env PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=$GPU_DEVICES python3 -m dynamo.vllm \
                    "${VLLM_ARGS[@]}"
            fi
        } &
        PIDS+=($!)
        echo "Started $MODE worker $i (PID: $!)"

        # Add delay between TensorRT-LLM worker launches to avoid MPI initialization conflicts
        if [ "$USE_TRTLLM" = true ] && [ "$i" -lt "$NUM_WORKERS" ]; then
            echo "Waiting 2 seconds before launching next TensorRT-LLM worker..."
            sleep 2
        fi
    done
fi

echo "All workers started. Press Ctrl+C to stop."
wait
echo "All workers completed."