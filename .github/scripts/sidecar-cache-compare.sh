#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove after the requested CUDA cache comparison has run in CI.
set -euo pipefail

comparison_dir="${GITHUB_WORKSPACE}/test-results/cache-comparison"
mkdir -p "${comparison_dir}" "${GITHUB_WORKSPACE}/test-results/status"
status_file="${GITHUB_WORKSPACE}/test-results/status/${TEST_TYPE}.exitcode"
echo 1 > "${status_file}"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > "${comparison_dir}/gpu.csv"
cat "${comparison_dir}/gpu.csv"
if ! grep -q 'RTX PRO 4500 Blackwell' "${comparison_dir}/gpu.csv"; then
  echo '::error::The CUDA cache comparison requires an RTX PRO 4500 Blackwell runner.'
  exit 1
fi

export PYTEST_NODE_ID='tests/serve/test_sidecar.py::test_serve_deployment[vllm_disaggregated-4]'
export PYTEST_DD_FLAKY_RETRY_ENABLED=false
export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} -o log_cli=true --log-cli-level=INFO"
printf 'attempt,exit_code\n' > "${comparison_dir}/results.csv"

cache_snapshot() {
  python - <<'PY'
import json
import os
from pathlib import Path

# pytest-local.sh pins HOME to GITHUB_WORKSPACE; CUDA uses this default
# when the runner has not supplied CUDA_CACHE_PATH. Do not clear a shared cache.
cache = Path(
    os.environ.get("CUDA_CACHE_PATH")
    or str(Path(os.environ["GITHUB_WORKSPACE"]) / ".nv/ComputeCache")
).resolve()
mounts = []
for line in Path("/proc/self/mountinfo").read_text().splitlines():
    fields, filesystem = line.split(" - ", 1)
    mount = Path(fields.split()[4])
    if cache == mount or mount in cache.parents:
        mounts.append((len(mount.parts), str(mount), filesystem.split()[0]))
_, mount, filesystem = max(mounts)
files = []
if cache.is_dir():
    for path in sorted(cache.rglob("*")):
        if path.is_file():
            stat = path.stat()
            files.append([str(path.relative_to(cache)), stat.st_size, stat.st_mtime_ns])
print(json.dumps({
    "cache_path": str(cache),
    "mount": mount,
    "filesystem": filesystem,
    "nfs_backed": filesystem in {"nfs", "nfs4"},
    "cache_writable": cache.is_dir() and os.access(cache, os.W_OK),
    "file_count": len(files),
    "total_bytes": sum(item[1] for item in files),
    "CUDA_CACHE_DISABLE": os.environ.get("CUDA_CACHE_DISABLE", "unset"),
    "CUDA_CACHE_MAXSIZE": os.environ.get("CUDA_CACHE_MAXSIZE", "unset"),
    "files": files,
}, indent=2))
PY
}

# The first attempt uses the cache as found. Later attempts retain it.
# This does not claim an empty cache or isolate activity from other runners.
failed=0
for attempt in 1 2 3; do
  attempt_dir="${comparison_dir}/attempt-${attempt}"
  mkdir -p "${attempt_dir}"
  export DYN_TEST_OUTPUT_PATH="${attempt_dir}/engine-logs"
  printf "Starting cache comparison attempt %s at " "${attempt}"
  date -u +%Y-%m-%dT%H:%M:%SZ
  cache_snapshot > "${attempt_dir}/cache-before.json"
  cat "${attempt_dir}/cache-before.json"
  code=0
  bash "${GITHUB_WORKSPACE}/.github/scripts/pytest-local.sh" \
    > "${attempt_dir}/pytest.log" 2>&1 || code=$?
  cat "${attempt_dir}/pytest.log"
  cp "${status_file}" "${attempt_dir}/stage.exitcode"
  stage_code=''
  read -r stage_code < "${status_file}" || true
  if [[ "${stage_code}" != '0' && "${code}" == '0' ]]; then
    code=1
  fi
  printf '%s,%s\n' "${attempt}" "${code}" >> "${comparison_dir}/results.csv"
  cache_snapshot > "${attempt_dir}/cache-after.json"
  cat "${attempt_dir}/cache-after.json"
  report="${GITHUB_WORKSPACE}/test-results/pytest_test_report_${TEST_SUITE_NAME}_${TEST_TYPE}_${PLATFORM_ARCH}_${GITHUB_RUN_ID}_${JOB_CHECK_RUN_ID}.xml"
  if [[ -f "${report}" ]]; then
    cp "${report}" "${attempt_dir}/"
    mv "${report}" "${report%.xml}_attempt-${attempt}.xml"
  fi
  if (( code != 0 )); then
    failed=1
  fi
done

# Preserve every failure even if subsequent attempts pass.
cat "${comparison_dir}/results.csv"
echo "${failed}" > "${status_file}"
exit "${failed}"
