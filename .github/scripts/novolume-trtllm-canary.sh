#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail

readonly CONTAINER_WORKSPACE=/workspace
readonly TEST_RESULTS_DIR="${GITHUB_WORKSPACE}/test-results"

mkdir -p "${TEST_RESULTS_DIR}"
chmod 777 "${TEST_RESULTS_DIR}"

echo "::group::Sanity and service checks"
python "${CONTAINER_WORKSPACE}/dev/sanity_check.py" --runtime-check --no-gpu-check

for attempt in $(seq 1 30); do
  if curl -sf http://localhost:9000/minio/health/live >/dev/null 2>&1; then
    echo "MinIO is ready (attempt ${attempt})"
    break
  fi
  if [[ "${attempt}" == 30 ]]; then
    echo "MinIO failed to start within 30 seconds" >&2
    exit 1
  fi
  sleep 1
done

python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    sys.exit("GPU tests requested but CUDA is unavailable")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
PY
echo "::endgroup::"

if [[ -d /scratch ]]; then
  readonly WORK_DIR=/scratch
else
  readonly WORK_DIR=/tmp
fi

if mkdir -p /models 2>/dev/null && [[ -w /models ]]; then
  export HF_HOME=/models
else
  export HF_HOME="${WORK_DIR}/.cache/huggingface"
  mkdir -p "${HF_HOME}"
fi

export HOME="${GITHUB_WORKSPACE}"
export XDG_CACHE_HOME="${WORK_DIR}/.cache"
mkdir -p "${XDG_CACHE_HOME}"

run_suite() {
  local suite_name=$1
  local marks=$2
  local workers=$3
  shift 3

  local suite_results="${TEST_RESULTS_DIR}/${suite_name}"
  mkdir -p "${suite_results}/allure-results"

  local -a command=(
    python -m pytest
    -n "${workers}"
    "$@"
    --continue-on-collection-errors
    -v
    --tb=short
    "--basetemp=${WORK_DIR}/pytest_temp-${suite_name}"
    -o "cache_dir=${WORK_DIR}/.pytest_cache"
    "--junitxml=${suite_results}/pytest_test_report.xml"
    "--alluredir=${suite_results}/allure-results"
    --durations=10
    -m "${marks}"
  )

  echo "::group::${suite_name}"
  printf 'Running:'
  printf ' %q' "${command[@]}"
  printf '\n'
  (
    cd "${CONTAINER_WORKSPACE}"
    "${command[@]}"
  )
  local exit_code=$?
  echo "${suite_name} exit code: ${exit_code}"
  echo "::endgroup::"
  return "${exit_code}"
}

overall_status=0

run_suite \
  pre_merge_cpu \
  'pre_merge and trtllm and gpu_0' \
  auto \
  --dist=loadscope || overall_status=1

run_suite \
  pre_merge_gpu_parallel \
  'pre_merge and trtllm and gpu_1' \
  auto \
  --max-vram-gib=24 || overall_status=1

run_suite \
  pre_merge_gpu \
  '(pre_merge and trtllm and gpu_1) and not profiled_vram_gib' \
  0 \
  --dist=loadscope || overall_status=1

exit "${overall_status}"
