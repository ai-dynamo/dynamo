#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch SDXL image generation on Intel XPU locally (frontend + worker).
#
# Environment variables (all optional):
#   PYTHON_BIN          Python interpreter (default: python3)
#   MODEL               HuggingFace model (default: stabilityai/stable-diffusion-xl-base-1.0)
#   HTTP_PORT           Frontend port (default: 8000)
#   DTYPE               bf16 or fp16 (default: bf16)
#   WORKER_EXTRA_ARGS   Extra flags for worker_xpu.py
#   FRONTEND_EXTRA_ARGS Extra flags for dynamo.frontend

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${PYTHON_BIN:=python3}"
: "${MODEL:=stabilityai/stable-diffusion-xl-base-1.0}"
: "${HTTP_PORT:=8000}"
: "${DTYPE:=bf16}"
: "${DISCOVERY_DIR:=${SCRIPT_DIR}/.runtime/discovery}"
: "${LOG_DIR:=${SCRIPT_DIR}/.runtime/logs}"
: "${WORKER_EXTRA_ARGS:=}"
: "${FRONTEND_EXTRA_ARGS:=}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "error: ${PYTHON_BIN} not found"
  exit 1
fi

# Verify XPU is visible before launching
if ! "${PYTHON_BIN}" -c "import torch; assert torch.xpu.is_available(), 'No XPU device found'" 2>/dev/null; then
  echo "error: Intel XPU not detected by PyTorch."
  echo "  Check drivers (level-zero) and PyTorch XPU install."
  exit 1
fi

mkdir -p "${DISCOVERY_DIR}" "${LOG_DIR}"

export DYN_DISCOVERY_BACKEND=file
export DYN_FILE_KV="${DYN_FILE_KV:-${DISCOVERY_DIR}}"

worker_cmd=("${PYTHON_BIN}" "${SCRIPT_DIR}/worker_xpu.py" --model "${MODEL}" --dtype "${DTYPE}")
if [[ -n "${WORKER_EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  worker_extra=( ${WORKER_EXTRA_ARGS} )
  worker_cmd+=("${worker_extra[@]}")
fi

frontend_cmd=("${PYTHON_BIN}" -m dynamo.frontend --http-port "${HTTP_PORT}")
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
echo "API endpoint: http://localhost:${HTTP_PORT}/v1/images/generations"
echo ""
echo "Example request:"
echo "  curl -s -X POST http://localhost:${HTTP_PORT}/v1/images/generations \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"${MODEL}\",\"prompt\":\"A photo of a cat sitting on a windowsill at sunset\",\"size\":\"1024x1024\",\"nvext\":{\"num_inference_steps\":30,\"guidance_scale\":7.5,\"seed\":42}}' \\"
echo "    | python3 -c \"import sys,json,base64; d=json.load(sys.stdin); open('output.png','wb').write(base64.b64decode(d['data'][0]['b64_json'])); print('Saved output.png')\""
echo ""

wait -n "${worker_pid}" "${frontend_pid}"
