#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${EXAMPLE_DIR}/../.." && pwd)"

source "${REPO_DIR}/examples/common/launch_utils.sh" # print_launch_banner, print_curl_footer, wait_any_exit
source "${REPO_DIR}/examples/common/gpu_utils.sh"

: "${PYTHON_BIN:=python3}"
: "${MODEL:=FastVideo/FastWan2.1-T2V-1.3B-Diffusers}"
: "${NUM_GPUS:=1}"
: "${DYN_HTTP_PORT:=${HTTP_PORT:-8000}}"
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

cd "${EXAMPLE_DIR}"

worker_cmd=("${PYTHON_BIN}" worker.py --model "${MODEL}" --num-gpus "${NUM_GPUS}")
if [[ -n "${WORKER_EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  worker_extra=( ${WORKER_EXTRA_ARGS} )
  worker_cmd+=("${worker_extra[@]}")
fi

frontend_cmd=("${PYTHON_BIN}" -m dynamo.frontend --http-port "${DYN_HTTP_PORT}" --discovery-backend file)
if [[ -n "${FRONTEND_EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  frontend_extra=( ${FRONTEND_EXTRA_ARGS} )
  frontend_cmd+=("${frontend_extra[@]}")
fi

trap dynamo_exit_trap EXIT

print_launch_banner --no-curl "Launching FastVideo Diffusers Example" "${MODEL}" "${DYN_HTTP_PORT}" \
  "Worker log:   ${LOG_DIR}/worker.log" \
  "Frontend log: ${LOG_DIR}/frontend.log"
print_curl_footer <<CURL
  curl -s -X POST http://localhost:${DYN_HTTP_PORT}/v1/videos \\
    -H 'Content-Type: application/json' \\
    -d '{
      "model": "${MODEL}",
      "prompt": "A cinematic drone shot over snowy mountains at sunrise",
      "size": "256x256",
      "response_format": "b64_json",
      "nvext": {
        "fps": 8,
        "num_frames": 8,
        "num_inference_steps": 1,
        "guidance_scale": 1.0,
        "seed": 10
      }
    }' > response.json
CURL

echo "Starting worker: ${worker_cmd[*]}"
"${worker_cmd[@]}" >"${LOG_DIR}/worker.log" 2>&1 &
worker_pid=$!

echo "Starting frontend: ${frontend_cmd[*]}"
"${frontend_cmd[@]}" >"${LOG_DIR}/frontend.log" 2>&1 &
frontend_pid=$!

wait_any_exit
