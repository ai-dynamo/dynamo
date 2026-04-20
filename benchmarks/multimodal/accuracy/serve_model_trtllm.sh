#!/bin/bash
set -euo pipefail

MODEL="Qwen/Qwen3-VL-2B-Instruct"
PORT="${TRTLLM_PORT:-8000}"
CONTAINER_NAME="trtllm-accuracy-server"
IMAGE="nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.0.0"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

echo "=== Starting TRT-LLM server for ${MODEL} ==="
echo "Port: ${PORT}"
echo "Container: ${CONTAINER_NAME}"
echo "HF cache: ${HF_CACHE}"

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

# Host driver is 565.x (CUDA 12.7); image needs CUDA 13. Force-load the
# forward-compat libcuda AND ptxjitcompiler from /usr/local/cuda/compat —
# loading only libcuda segfaults because it mismatches the host-565
# ptxjitcompiler that the nvidia container runtime injects.
docker run -d \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    --ipc=host \
    --network host \
    -v "${HF_CACHE}":/root/.cache/huggingface \
    -v "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)":/cfg:ro \
    -e LD_PRELOAD="/usr/local/cuda/compat/lib.real/libcuda.so.1 /usr/local/cuda/compat/lib.real/libnvidia-ptxjitcompiler.so.1" \
    ${HF_TOKEN:+-e HF_TOKEN="${HF_TOKEN}"} \
    --entrypoint trtllm-serve \
    "${IMAGE}" \
    serve "${MODEL}" \
        --backend pytorch \
        --trust_remote_code \
        --host 0.0.0.0 \
        --port "${PORT}" \
        --max_seq_len 8192 \
        --max_batch_size 4 \
        --free_gpu_memory_fraction 0.55 \
        --extra_llm_api_options /cfg/trtllm_extra_config.yaml

echo "=== Waiting for server to be ready ==="
# TRT-LLM startup is slower than vLLM (MPI init + torch.compile warmup)
MAX_WAIT=900
ELAPSED=0
until curl -sf http://localhost:${PORT}/health > /dev/null 2>&1; do
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "ERROR: container exited before becoming healthy"
        docker logs "${CONTAINER_NAME}" --tail 80
        exit 1
    fi
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "ERROR: Server did not become healthy within ${MAX_WAIT}s"
        docker logs "${CONTAINER_NAME}" --tail 80
        exit 1
    fi
    sleep 10
    ELAPSED=$((ELAPSED + 10))
    echo "  Waiting... (${ELAPSED}s / ${MAX_WAIT}s)"
done

echo "=== Server is ready on port ${PORT} ==="
curl -s http://localhost:${PORT}/v1/models | python3 -m json.tool || true
