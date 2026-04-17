#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end security test runner. Launches the TRT-LLM multimodal server
# inside a container with the security-patched code, then runs the test suite.
#
# Usage:
#   bash tests/security/run_security_tests.sh
#
# Environment:
#   IMAGE          - Container image (default: dynamo:trtllm-flashcache)
#   MODEL_PATH     - HF model path (default: Qwen/Qwen3-VL-2B-Instruct)
#   HF_HOME        - HuggingFace cache directory

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

IMAGE="${IMAGE:-dynamo:trtllm-flashcache}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-2B-Instruct}"
HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
CONTAINER_NAME="dynamo-security-test-$$"
FRONTEND_PORT="${FRONTEND_PORT:-8000}"
FRONTEND_DECODING="${FRONTEND_DECODING:-false}"

echo "================================================================="
echo "  Dynamo TRT-LLM Multimodal Security Test Runner"
echo "================================================================="
echo "Image:     ${IMAGE}"
echo "Model:     ${MODEL_PATH}"
echo "Container: ${CONTAINER_NAME}"
echo "Port:      ${FRONTEND_PORT}"
echo "HF_HOME:   ${HF_HOME}"
echo "Frontend decoding: ${FRONTEND_DECODING}"
echo ""

cleanup() {
    echo "[*] Cleaning up container ${CONTAINER_NAME}..."
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Build the overlay of modified files to mount into the container
MODIFIED_FILES=(
    "components/src/dynamo/trtllm/multimodal_processor.py"
    "components/src/dynamo/trtllm/encode_helper.py"
    "components/src/dynamo/trtllm/workers/llm_worker.py"
    "components/src/dynamo/trtllm/backend_args.py"
)

# Build volume mount args for modified source files
MOUNT_ARGS=""
for f in "${MODIFIED_FILES[@]}"; do
    HOST_PATH="${REPO_ROOT}/${f}"
    # Mount into both workspace and installed package
    PKG_PATH="/opt/dynamo/venv/lib/python3.12/site-packages/dynamo/trtllm/${f#components/src/dynamo/trtllm/}"
    WS_PATH="/workspace/${f}"
    MOUNT_ARGS="${MOUNT_ARGS} -v ${HOST_PATH}:${PKG_PATH}:ro"
    MOUNT_ARGS="${MOUNT_ARGS} -v ${HOST_PATH}:${WS_PATH}:ro"
done

# Mount test scripts and launch scripts
MOUNT_ARGS="${MOUNT_ARGS} -v ${REPO_ROOT}/tests/security:/workspace/tests/security:ro"
MOUNT_ARGS="${MOUNT_ARGS} -v ${REPO_ROOT}/examples/backends/trtllm/launch/agg_multimodal_qwen3vl.sh:/workspace/examples/backends/trtllm/launch/agg_multimodal_qwen3vl.sh:ro"

# Mount HuggingFace cache
MOUNT_ARGS="${MOUNT_ARGS} -v ${HF_HOME}:/home/dynamo/.cache/huggingface"

echo "[*] Starting server container..."
docker run -d \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --network host \
    -e ETCD_ENDPOINTS="${ETCD_ENDPOINTS:-http://localhost:2379}" \
    -e NATS_SERVER="${NATS_SERVER:-nats://localhost:4222}" \
    -e HF_HOME=/home/dynamo/.cache/huggingface \
    -e MODEL_PATH="${MODEL_PATH}" \
    -e SERVED_MODEL_NAME="${MODEL_PATH}" \
    -e DYN_HTTP_PORT="${FRONTEND_PORT}" \
    -e FRONTEND_DECODING="${FRONTEND_DECODING}" \
    -e PYTHONUNBUFFERED=1 \
    ${MOUNT_ARGS} \
    "${IMAGE}" \
    bash -lc "bash /workspace/examples/backends/trtllm/launch/agg_multimodal_qwen3vl.sh"

echo "[*] Waiting for model to register on :${FRONTEND_PORT}..."
READY=0
for i in $(seq 1 180); do
    MODELS="$(curl -sf "http://127.0.0.1:${FRONTEND_PORT}/v1/models" 2>/dev/null || true)"
    if echo "${MODELS}" | grep -q "${MODEL_PATH}"; then
        READY=1
        echo "[+] Model registered after ${i}s"
        break
    fi
    sleep 2
done

if [ "${READY}" != "1" ]; then
    echo "[!] Model not ready after 360s. Container logs:"
    docker logs "${CONTAINER_NAME}" 2>&1 | tail -100
    exit 1
fi

echo ""
echo "[*] Running security tests..."
echo ""

# Generate random safetensors for test 6
docker exec "${CONTAINER_NAME}" bash -c \
    "python3 /workspace/tests/security/generate_safetensors_embeddings.py \
        --random --hidden-size 1536 --seq-len 256 \
        --output /tmp/test_embeddings.safetensors" || true

# Run the test suite
FRONTEND_URL="http://127.0.0.1:${FRONTEND_PORT}" \
SERVED_MODEL_NAME="${MODEL_PATH}" \
FRONTEND_DECODING="${FRONTEND_DECODING}" \
bash "${REPO_ROOT}/tests/security/test_multimodal_security.sh"

echo ""
echo "[*] Container logs (last 50 lines):"
docker logs "${CONTAINER_NAME}" 2>&1 | tail -50
