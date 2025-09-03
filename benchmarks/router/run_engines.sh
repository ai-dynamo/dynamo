#!/bin/bash

NUM_WORKERS=8
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
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

        CUDA_VISIBLE_DEVICES=$((i-1)) python -m dynamo.vllm \
            --model "$MODEL_PATH" \
            --endpoint dyn://test.vllm.generate \
            --enforce-eager \
            --max-num-batched-tokens 16384 \
            --max-model-len 32768 \
            --block-size 64
        echo "[Worker-$i] Finished"
    } &
    PIDS+=($!)
    echo "Started worker $i (PID: $!)"
done

echo "All workers started. Press Ctrl+C to stop."
wait
echo "All workers completed."