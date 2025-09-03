#!/bin/bash

# Parse command-line arguments
NUM_WORKERS=8
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
TENSOR_PARALLEL_SIZE=1
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
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            # Collect all other arguments as vLLM arguments
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# If no extra args provided, use defaults
if [ ${#EXTRA_ARGS[@]} -eq 0 ]; then
    EXTRA_ARGS=(
        "--enforce-eager"
        "--max-num-batched-tokens" "16384"
        "--max-model-len" "32768"
        "--block-size" "64"
    )
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

# Calculate total GPUs needed
TOTAL_GPUS_NEEDED=$((NUM_WORKERS * TENSOR_PARALLEL_SIZE))
echo "Configuration:"
echo "  Workers: $NUM_WORKERS"
echo "  Model: $MODEL_PATH"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "  Total GPUs needed: $TOTAL_GPUS_NEEDED"
echo "  vLLM args: ${EXTRA_ARGS[*]}"
echo ""

PIDS=()

cleanup() {
    echo -e "\nStopping all workers..."
    kill "${PIDS[@]}" 2>/dev/null
    wait
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "Starting $NUM_WORKERS workers..."

for i in $(seq 1 $NUM_WORKERS); do
    {
        echo "[Worker-$i] Starting..."

        # Calculate GPU indices for this worker
        START_GPU=$(( (i - 1) * TENSOR_PARALLEL_SIZE ))
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

        echo "[Worker-$i] Using GPUs: $GPU_DEVICES"

        CUDA_VISIBLE_DEVICES=$GPU_DEVICES python -m dynamo.vllm \
            --model "$MODEL_PATH" \
            --endpoint dyn://test.vllm.generate \
            --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
            "${EXTRA_ARGS[@]}"
        echo "[Worker-$i] Finished"
    } &
    PIDS+=($!)
    echo "Started worker $i (PID: $!)"
done

echo "All workers started. Press Ctrl+C to stop."
wait
echo "All workers completed."