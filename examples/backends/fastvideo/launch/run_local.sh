#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${EXAMPLE_DIR}/../../.." && pwd)"

: "${PYTHON_BIN:=python3}"
: "${MODEL:=Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
: "${NUM_GPUS:=1}"
: "${HTTP_PORT:=8000}"
: "${DISCOVERY_DIR:=${SCRIPT_DIR}/.runtime/discovery}"
: "${LOG_DIR:=${SCRIPT_DIR}/.runtime/logs}"
: "${WORKER_EXTRA_ARGS:=}"
: "${FRONTEND_EXTRA_ARGS:=}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "error: ${PYTHON_BIN} not found"
  exit 1
fi

mkdir -p "${DISCOVERY_DIR}" "${LOG_DIR}"

export DYN_DISCOVERY_BACKEND=file
export DYN_FILE_KV="${DYN_FILE_KV:-${DISCOVERY_DIR}}"
export PYTHONPATH="${REPO_DIR}/components/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${EXAMPLE_DIR}"

worker_cmd=("${PYTHON_BIN}" -m dynamo.fastvideo --model "${MODEL}" --num-gpus "${NUM_GPUS}")
if [[ -n "${WORKER_EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  worker_extra=( ${WORKER_EXTRA_ARGS} )
  worker_cmd+=("${worker_extra[@]}")
fi

frontend_cmd=("${PYTHON_BIN}" -m dynamo.frontend --http-port "${HTTP_PORT}" --discovery-backend file)
if [[ -n "${FRONTEND_EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  frontend_extra=( ${FRONTEND_EXTRA_ARGS} )
  frontend_cmd+=("${frontend_extra[@]}")
fi

cleanup() {
  echo
  echo "Stopping local processes..."
  kill "${frontend_pid:-}" "${worker_pid:-}" 2>/dev/null || true
  wait "${frontend_pid:-}" "${worker_pid:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting worker: ${worker_cmd[*]}"
"${worker_cmd[@]}" >"${LOG_DIR}/worker.log" 2>&1 &
worker_pid=$!

echo "Starting frontend: ${frontend_cmd[*]}"
"${frontend_cmd[@]}" >"${LOG_DIR}/frontend.log" 2>&1 &
frontend_pid=$!

echo ""
echo "Worker log:   ${LOG_DIR}/worker.log"
echo "Frontend log: ${LOG_DIR}/frontend.log"
echo ""
echo "API endpoint: http://localhost:${HTTP_PORT}/v1/videos"
echo ""
echo "Example request:"
echo "curl -s -X POST http://localhost:${HTTP_PORT}/v1/videos -H 'Content-Type: application/json' -d '{\"model\":\"${MODEL}\",\"prompt\":\"A cinematic drone shot over snowy mountains at sunrise\",\"size\":\"1920x1088\",\"response_format\":\"b64_json\",\"seconds\":5,\"nvext\":{\"fps\":24,\"num_frames\":121,\"num_inference_steps\":5,\"guidance_scale\":1.0,\"seed\":10}}' > response.json"
echo ""

wait -n "${worker_pid}" "${frontend_pid}"
