#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Interactive launcher for the prompt-enhance multimodal generation example.
# Set MODE=video or MODE=image to choose the diffusion backend.
# Run this on a local workstation or an already allocated node.

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NS="${NS:-prompt-enhance}"
PYTHON="${PYTHON:-python}"
MODE="${MODE:-video}"

LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3-0.6B}"

FE_LLM_PORT="${FE_LLM_PORT:-8000}"

ENHANCER_GPU="${ENHANCER_GPU:-0}"

case "${MODE}" in
  video)
    BACKEND_MODEL="${BACKEND_MODEL:-${T2V_MODEL:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}}"
    FE_BACKEND_PORT="${FE_BACKEND_PORT:-${FE_T2V_PORT:-8001}}"
    BACKEND_GPUS="${BACKEND_GPUS:-${T2V_GPUS:-1,2,3}}"
    BACKEND_MASTER_PORT_BASE="${BACKEND_MASTER_PORT_BASE:-${T2V_MASTER_PORT_BASE:-30110}}"
    BACKEND_MASTER_PORT_STRIDE="${BACKEND_MASTER_PORT_STRIDE:-${T2V_MASTER_PORT_STRIDE:-10}}"
    BACKEND_NAMESPACE="${NS}-t2v"
    BACKEND_LABEL="T2V"
    BACKEND_ROUTE="/v1/videos"
    BACKEND_WORKER_FLAG=(--video-generation-worker)
    BACKEND_MODEL_ARG=(--model "${BACKEND_MODEL}")
    BACKEND_EXTRA_ARGS=()
    ;;
  image)
    BACKEND_MODEL="${BACKEND_MODEL:-${T2I_MODEL:-black-forest-labs/FLUX.1-dev}}"
    FE_BACKEND_PORT="${FE_BACKEND_PORT:-${FE_T2I_PORT:-8001}}"
    BACKEND_GPUS="${BACKEND_GPUS:-${T2I_GPUS:-1}}"
    BACKEND_MASTER_PORT_BASE="${BACKEND_MASTER_PORT_BASE:-${T2I_MASTER_PORT_BASE:-30110}}"
    BACKEND_MASTER_PORT_STRIDE="${BACKEND_MASTER_PORT_STRIDE:-${T2I_MASTER_PORT_STRIDE:-10}}"
    BACKEND_NAMESPACE="${NS}-t2i"
    BACKEND_LABEL="T2I"
    BACKEND_ROUTE="/v1/images/generations"
    BACKEND_WORKER_FLAG=(--image-diffusion-worker)
    BACKEND_MODEL_ARG=(--model-path "${BACKEND_MODEL}")
    BACKEND_EXTRA_ARGS=(--trust-remote-code --skip-tokenizer-init)
    ;;
  *)
    echo "MODE must be 'video' or 'image'." >&2
    exit 1
    ;;
esac

MEDIA_DIR="${MEDIA_DIR:-${THIS_DIR}/media}"
mkdir -p "${MEDIA_DIR}"

if [[ -n "${DYN_FILE_KV_ROOT:-}" ]]; then
  CLEAN_DYN_FILE_KV_ROOT=0
else
  DYN_FILE_KV_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/dynamo-${NS}.XXXXXX")"
  CLEAN_DYN_FILE_KV_ROOT=1
fi

DYN_KV_LLM="${DYN_FILE_KV_ROOT}/llm"
DYN_KV_BACKEND="${DYN_FILE_KV_ROOT}/${MODE}"
rm -rf "${DYN_KV_LLM}" "${DYN_KV_BACKEND}"
mkdir -p "${DYN_KV_LLM}" "${DYN_KV_BACKEND}"

IFS=',' read -r -a BACKEND_GPU_LIST <<< "${BACKEND_GPUS}"
if [[ "${#BACKEND_GPU_LIST[@]}" -eq 0 ]]; then
  echo "BACKEND_GPUS must contain at least one GPU id" >&2
  exit 1
fi

pids=()
cleanup() {
  set +e
  for p in "${pids[@]:-}"; do kill "${p}" 2>/dev/null || true; done
  sleep 3
  for p in "${pids[@]:-}"; do kill -9 "${p}" 2>/dev/null || true; done
  if [[ "${CLEAN_DYN_FILE_KV_ROOT}" == "1" ]]; then
    rm -rf "${DYN_FILE_KV_ROOT}"
  fi
}
trap cleanup EXIT INT TERM

echo "Media output: ${MEDIA_DIR}"
echo "Discovery state: ${DYN_FILE_KV_ROOT}"
echo "LLM frontend: http://127.0.0.1:${FE_LLM_PORT}"
echo "${BACKEND_LABEL} frontend: http://127.0.0.1:${FE_BACKEND_PORT}${BACKEND_ROUTE}"

DYN_FILE_KV="${DYN_KV_LLM}" "${PYTHON}" -m dynamo.frontend \
  --http-port "${FE_LLM_PORT}" --namespace "${NS}-llm" \
  --discovery-backend file --request-plane tcp --event-plane zmq &
pids+=($!)

DYN_FILE_KV="${DYN_KV_BACKEND}" "${PYTHON}" -m dynamo.frontend \
  --http-port "${FE_BACKEND_PORT}" --namespace "${BACKEND_NAMESPACE}" \
  --discovery-backend file --request-plane tcp --event-plane zmq &
pids+=($!)

sleep 5

CUDA_VISIBLE_DEVICES="${ENHANCER_GPU}" DYN_FILE_KV="${DYN_KV_LLM}" \
"${PYTHON}" -m dynamo.sglang \
  --model "${LLM_MODEL}" --served-model-name "${LLM_MODEL}" \
  --namespace "${NS}-llm" \
  --discovery-backend file --request-plane tcp --event-plane zmq &
pids+=($!)

for i in "${!BACKEND_GPU_LIST[@]}"; do
  gpu="${BACKEND_GPU_LIST[$i]}"
  port=$((BACKEND_MASTER_PORT_BASE + i * BACKEND_MASTER_PORT_STRIDE))
  CUDA_VISIBLE_DEVICES="${gpu}" DYN_FILE_KV="${DYN_KV_BACKEND}" \
  "${PYTHON}" -m dynamo.sglang \
    "${BACKEND_WORKER_FLAG[@]}" \
    "${BACKEND_MODEL_ARG[@]}" --served-model-name "${BACKEND_MODEL}" \
    --namespace "${BACKEND_NAMESPACE}" \
    --media-output-fs-url "file://${MEDIA_DIR}" \
    --master-port "${port}" \
    "${BACKEND_EXTRA_ARGS[@]}" \
    --discovery-backend file --request-plane tcp --event-plane zmq &
  pids+=($!)
done

echo "Started ${#BACKEND_GPU_LIST[@]} ${BACKEND_LABEL} replica(s). Press Ctrl-C to stop."
wait
