#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove this experiment after the requested CI evidence is collected.
set -euo pipefail

comparison_dir="${GITHUB_WORKSPACE}/test-results/ucx-comparison"
mkdir -p "${comparison_dir}" "${GITHUB_WORKSPACE}/test-results/status"
status_file="${GITHUB_WORKSPACE}/test-results/status/${TEST_TYPE}.exitcode"
: > "${status_file}"

# Both variants must use the requested hardware in the same container/job.
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > "${comparison_dir}/gpu.csv"
cat "${comparison_dir}/gpu.csv"
if ! grep -q 'RTX PRO 4500 Blackwell' "${comparison_dir}/gpu.csv"; then
  echo '::error::UCX comparison requires an RTX PRO 4500 Blackwell runner.'
  echo 1 > "${status_file}"
  exit 1
fi

export PYTEST_NODE_ID='tests/serve/test_sidecar.py::test_serve_deployment[vllm_disaggregated-4]'
export PYTEST_DD_FLAKY_RETRY_ENABLED=false
export UCX_LOG_LEVEL=info UCX_PROTO_INFO=y NIXL_LOG_LEVEL=DEBUG
export VLLM_LOGGING_LEVEL=DEBUG DYN_LOG=debug
export PYTHONPATH="${GITHUB_WORKSPACE}/.github/scripts/sidecar-ucx-trace${PYTHONPATH:+:${PYTHONPATH}}"
printf 'attempt,variant,exit_code\n' > "${comparison_dir}/results.csv"

# Each new pytest process uses the original fixture, payloads, and warm-up.
# Interleave three pairs, reversing the middle pair to limit order bias.
variants=(baseline ipc_get ipc_get baseline baseline ipc_get)
failed=0
attempt=0
for variant in "${variants[@]}"; do
  attempt=$((attempt + 1))
  attempt_dir="${comparison_dir}/${attempt}-${variant}"
  mkdir -p "${attempt_dir}"
  export DYN_TEST_OUTPUT_PATH="${attempt_dir}/engine-logs"
  export NURSERY_UCX_ATTEMPT="${attempt}-${variant}"
  unset UCX_CUDA_IPC_ENABLE_GET_ZCOPY
  if [[ "${variant}" == ipc_get ]]; then
    export UCX_CUDA_IPC_ENABLE_GET_ZCOPY=y
  fi
  printf 'attempt=%s variant=%s UCX_CUDA_IPC_ENABLE_GET_ZCOPY=%s\n' \
    "${attempt}" "${variant}" "${UCX_CUDA_IPC_ENABLE_GET_ZCOPY:-unset}" \
    > "${attempt_dir}/settings.txt"
  cat "${attempt_dir}/settings.txt"
  code=0
  bash "${GITHUB_WORKSPACE}/.github/scripts/pytest-local.sh" \
    > "${attempt_dir}/pytest.log" 2>&1 || code=$?
  cat "${attempt_dir}/pytest.log"
  # pytest-local.sh writes one stage status/JUnit name; preserve each attempt
  # before the next one overwrites it. Engine logs already have unique roots.
  cp "${status_file}" "${attempt_dir}/stage.exitcode"
  stage_code=''
  read -r stage_code < "${status_file}" || true
  if [[ "${stage_code}" != '0' && "${code}" == '0' ]]; then
    echo '::error::An attempt did not record a passing pytest stage result.'
    code=1
  fi
  printf '%s,%s,%s\n' "${attempt}" "${variant}" "${code}" >> "${comparison_dir}/results.csv"
  shopt -s nullglob
  reports=("${GITHUB_WORKSPACE}"/test-results/pytest_test_report_*.xml)
  if (( ${#reports[@]} )); then
    mv "${reports[@]}" "${attempt_dir}/"
  fi
  if (( code != 0 )); then
    failed=1
  fi
done

# Do not turn a failed attempt into a passing job because a later one passed.
cat "${comparison_dir}/results.csv"
echo "${failed}" > "${status_file}"
exit "${failed}"
