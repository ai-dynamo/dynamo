#!/bin/bash
set -euo pipefail

MODEL="Qwen/Qwen3-VL-2B-Instruct"
PORT="${VLLM_PORT:-8000}"
CONTAINER_NAME="vllm-accuracy-server"
IMAGE="nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.0"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

echo "=== Starting vLLM server for ${MODEL} ==="
echo "Port: ${PORT}"
echo "Container: ${CONTAINER_NAME}"
echo "HF cache: ${HF_CACHE}"

# Stop any existing container with the same name
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

docker run -d \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    --ipc=host \
    --network host \
    -v "${HF_CACHE}":/root/.cache/huggingface \
    ${HF_TOKEN:+-e HF_TOKEN="${HF_TOKEN}"} \
    --entrypoint vllm \
    "${IMAGE}" \
    serve "${MODEL}" \
        --host 0.0.0.0 \
        --port "${PORT}" \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.40 \
        --limit-mm-per-prompt '{"image":4,"video":0}' \
        --trust-remote-code

echo "=== Waiting for server to be ready ==="
MAX_WAIT=300
ELAPSED=0
until curl -sf http://localhost:${PORT}/health > /dev/null 2>&1; do
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "ERROR: Server did not become healthy within ${MAX_WAIT}s"
        echo "=== Container logs ==="
        docker logs "${CONTAINER_NAME}" --tail 50
        exit 1
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "  Waiting... (${ELAPSED}s / ${MAX_WAIT}s)"
done

echo "=== Server is ready on port ${PORT} ==="
curl -s http://localhost:${PORT}/v1/models | python3 -m json.tool
