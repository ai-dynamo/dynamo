#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail

readonly TEST_RESULTS_DIR="${GITHUB_WORKSPACE}/test-results"
readonly ALLURE_RESULTS_DIR="${TEST_RESULTS_DIR}/allure-results"

mkdir -p "${ALLURE_RESULTS_DIR}"
chmod 777 "${TEST_RESULTS_DIR}" "${ALLURE_RESULTS_DIR}"

if [[ -d "${NVME_DIR}" ]]; then
  readonly WORK_DIR="${NVME_DIR}"
  echo "NVMe scratch available at ${WORK_DIR}"
else
  readonly WORK_DIR=/tmp
fi

if
  [[ -n "${REQUESTED_HF_HOME}" ]] &&
  mkdir -p "${REQUESTED_HF_HOME}" 2>/dev/null &&
  [[ -w "${REQUESTED_HF_HOME}" ]]
then
  export HF_HOME="${REQUESTED_HF_HOME}"
else
  export HF_HOME="${WORK_DIR}/.cache/huggingface"
  mkdir -p "${HF_HOME}"
fi

export HOME="${GITHUB_WORKSPACE}"
export XDG_CACHE_HOME="${WORK_DIR}/.cache"
mkdir -p "${XDG_CACHE_HOME}"

if [[ ! -d "${CONTAINER_WORKSPACE}" ]]; then
  echo "container_workspace '${CONTAINER_WORKSPACE}' not found" >&2
  exit 1
fi

run_sanity_check() {
  echo "::group::Runtime image sanity check"
  (
    cd "${CONTAINER_WORKSPACE}"
    python ./dev/sanity_check.py --runtime-check --no-gpu-check
  )
  local exit_code=$?
  echo "::endgroup::"
  return "${exit_code}"
}

verify_gpu_services() {
  echo "::group::GPU and service checks"
  for attempt in $(seq 1 30); do
    if curl -sf http://localhost:9000/minio/health/live >/dev/null 2>&1; then
      echo "MinIO is ready (attempt ${attempt})"
      break
    fi
    if [[ "${attempt}" == 30 ]]; then
      echo "MinIO failed to start within 30 seconds" >&2
      echo "::endgroup::"
      return 1
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
  local exit_code=$?
  echo "::endgroup::"
  return "${exit_code}"
}

run_suite() {
  local suite_name=$1
  local marks=$2
  local parallel_mode=$3
  local max_vram_gib=$4

  local -a parallel_options
  case "${parallel_mode}" in
    auto)
      parallel_options=(-n auto)
      ;;
    none|0)
      parallel_options=(-n 0)
      ;;
    *)
      parallel_options=(-n "${parallel_mode}")
      ;;
  esac

  local -a distribution_options=(--dist=loadscope)
  local -a vram_options=()
  if [[ -n "${max_vram_gib}" ]]; then
    distribution_options=()
    vram_options=("--max-vram-gib=${max_vram_gib}")
  fi

  local -a coverage_options=()
  if [[ "${ENABLE_COVERAGE}" == true ]]; then
    coverage_options=(
      --cov=components/src/dynamo
      --cov=lib/bindings/python/src/dynamo
      --cov-report=
    )
    export COVERAGE_FILE="${TEST_RESULTS_DIR}/.coverage.${suite_name}"
    export PYTHONPATH="${CONTAINER_WORKSPACE}/components/src:${CONTAINER_WORKSPACE}/lib/bindings/python/src${PYTHONPATH:+:${PYTHONPATH}}"
  fi

  local junit_name="pytest_test_report_${TEST_SUITE_NAME}_${suite_name}_${PLATFORM_ARCH}_${GITHUB_RUN_ID}_${GITHUB_JOB}.xml"
  local -a command=(
    python -m pytest
    "${parallel_options[@]}"
    "${vram_options[@]}"
    "${distribution_options[@]}"
    --continue-on-collection-errors
    -v
    --tb=short
    "--basetemp=${WORK_DIR}/pytest_temp-${suite_name}"
    -o "cache_dir=${WORK_DIR}/.pytest_cache"
    "--junitxml=${TEST_RESULTS_DIR}/${junit_name}"
    "--alluredir=${ALLURE_RESULTS_DIR}"
    --durations=10
    "${coverage_options[@]}"
    -m "${marks}"
  )

  echo "::group::${suite_name}"
  printf 'Running:'
  printf ' %q' "${command[@]}"
  printf '\n'
  (
    cd "${CONTAINER_WORKSPACE}"
    DYNAMO_TEST_TYPE="${suite_name}" \
      "${command[@]}"
  )
  local exit_code=$?

  if [[ -f "${TEST_RESULTS_DIR}/${junit_name}" ]]; then
    echo "JUnit XML: ${junit_name}"
  else
    echo "JUnit XML was not generated for ${suite_name}" >&2
    [[ "${exit_code}" -ne 0 ]] || exit_code=1
  fi
  echo "${suite_name} exit code: ${exit_code}"
  echo "::endgroup::"
  return "${exit_code}"
}

case "${TEST_STAGE}" in
  sanity)
    run_sanity_check
    ;;
  cpu)
    run_suite pre_merge_cpu "${PYTEST_MARKS}" "${PARALLEL_MODE}" ''
    ;;
  gpu-parallel)
    verify_gpu_services &&
      run_suite pre_merge_gpu_parallel "${PYTEST_MARKS}" auto "${MAX_VRAM_GIB}"
    ;;
  gpu-sequential)
    verify_gpu_services &&
      run_suite pre_merge_gpu "${PYTEST_MARKS}" none ''
    ;;
  *)
    echo "Unknown TEST_STAGE '${TEST_STAGE}'" >&2
    exit 2
    ;;
esac
