#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# In-container bring-up for the prompt-enhanced text-to-video example.
# Starts:
#   * one dynamo.frontend on FE_LLM_PORT serving the enhancer LLM
#   * one dynamo.frontend on FE_T2V_PORT serving the diffusion backend
#   * one enhancer worker on GPU 0
#   * N diffusion replicas on GPUs 1..N (default N=3 for the GB200 layout)
#
# Each diffusion replica is launched with a distinct --master-port so
# sglang's multimodal_gen does not collide on 30005.

set -euo pipefail

JOB_ID="${SLURM_JOB_ID:-manual}"
NS="${NS:-prompt-enhanced-t2v}"

LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3-0.6B}"
T2V_MODEL="${T2V_MODEL:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"

FE_LLM_PORT="${FE_LLM_PORT:-8000}"
FE_T2V_PORT="${FE_T2V_PORT:-8001}"

T2V_REPLICAS="${T2V_REPLICAS:-3}"
T2V_MASTER_PORT_BASE="${T2V_MASTER_PORT_BASE:-30110}"
T2V_MASTER_PORT_STRIDE="${T2V_MASTER_PORT_STRIDE:-10}"

LOG_DIR="${LOG_DIR:-/scratch/exp-prompt-enhanced-t2v/logs/${JOB_ID}}"
MEDIA_DIR="${MEDIA_DIR:-/scratch/exp-prompt-enhanced-t2v/media/${JOB_ID}}"
mkdir -p "${LOG_DIR}" "${MEDIA_DIR}"

DYN_KV_LLM="/tmp/dyn-kv-llm-${JOB_ID}"
DYN_KV_T2V="/tmp/dyn-kv-t2v-${JOB_ID}"
rm -rf "${DYN_KV_LLM}" "${DYN_KV_T2V}"
mkdir -p "${DYN_KV_LLM}" "${DYN_KV_T2V}"

pids=()
cleanup() {
  set +e
  for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null || true; done
  sleep 3
  for p in "${pids[@]:-}"; do kill -9 "$p" 2>/dev/null || true; done
}
trap cleanup EXIT

DYN_FILE_KV="${DYN_KV_LLM}" python3 -m dynamo.frontend \
  --http-port "${FE_LLM_PORT}" --namespace "${NS}-llm" \
  --discovery-backend file --request-plane tcp --event-plane zmq \
  > "${LOG_DIR}/frontend_llm.log" 2>&1 &
pids+=($!)

DYN_FILE_KV="${DYN_KV_T2V}" python3 -m dynamo.frontend \
  --http-port "${FE_T2V_PORT}" --namespace "${NS}-t2v" \
  --discovery-backend file --request-plane tcp --event-plane zmq \
  > "${LOG_DIR}/frontend_t2v.log" 2>&1 &
pids+=($!)

sleep 5

CUDA_VISIBLE_DEVICES=0 DYN_FILE_KV="${DYN_KV_LLM}" \
python3 -m dynamo.sglang \
  --model "${LLM_MODEL}" --served-model-name "${LLM_MODEL}" \
  --namespace "${NS}-llm" \
  --discovery-backend file --request-plane tcp --event-plane zmq \
  > "${LOG_DIR}/enhancer.log" 2>&1 &
pids+=($!)

for ((g=1; g<=T2V_REPLICAS; g++)); do
  port=$((T2V_MASTER_PORT_BASE + (g - 1) * T2V_MASTER_PORT_STRIDE))
  CUDA_VISIBLE_DEVICES="${g}" DYN_FILE_KV="${DYN_KV_T2V}" \
  python3 -m dynamo.sglang \
    --video-generation-worker \
    --model "${T2V_MODEL}" --served-model-name "${T2V_MODEL}" \
    --namespace "${NS}-t2v" \
    --media-output-fs-url "file://${MEDIA_DIR}" \
    --master-port "${port}" \
    --discovery-backend file --request-plane tcp --event-plane zmq \
    > "${LOG_DIR}/t2v_g${g}.log" 2>&1 &
  pids+=($!)
done

wait
