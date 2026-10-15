#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Runs one pytest stage of a shared-test.yml job, inside the job container.
# It replaces the former .github/actions/pytest-local composite action: loading
# that action made the runner download the whole repository for every job, and
# the k8s-novolume hook then copied it into the GPU pod file by file.
#
# Inputs come from the step environment; never interpolate workflow expressions
# here (zizmor "template-injection"). Names and defaults match the old inputs.
set -eo pipefail

# Defaults apply only when the caller did not set a variable, as action input
# defaults did; an explicitly empty value stays empty.
: "${DRY_RUN=false}" "${ENABLE_COVERAGE=false}" "${PARALLEL_MODE=auto}"
: "${MAX_VRAM_GIB=}" "${NVME_DIR=/scratch}" "${REQUESTED_HF_HOME=/models}"
: "${GPU_TESTS=false}" "${VERIFY_GPU=true}" "${PYTEST_DD_FLAKY_RETRY_ENABLED=}"
: "${HF_TOKEN=}" "${TRITON_ARTIFACTORY_USER=}" "${TRITON_ARTIFACTORY_TOKEN=}"
: "${DYNAMO_TEST_FRAMEWORK=unknown}"
export DRY_RUN ENABLE_COVERAGE PARALLEL_MODE MAX_VRAM_GIB NVME_DIR \
  REQUESTED_HF_HOME GPU_TESTS VERIFY_GPU PYTEST_DD_FLAKY_RETRY_ENABLED \
  HF_TOKEN TRITON_ARTIFACTORY_USER TRITON_ARTIFACTORY_TOKEN DYNAMO_TEST_FRAMEWORK

# The action derived these from workflow context; the runner provides the same
# values as GITHUB_* variables in the job container.
export RUN_ID="${GITHUB_RUN_ID}"
export CONTAINER_ID="test_${GITHUB_RUN_ID}_${GITHUB_RUN_ATTEMPT}_${GITHUB_JOB}"
export PYTEST_XML_FILE=pytest_test_report.xml
export DYNAMO_TEST_PLATFORM="${PLATFORM_ARCH}"
export DYNAMO_TEST_TYPE="${TEST_TYPE}"
export DYNAMO_TEST_WORKFLOW="${GITHUB_WORKFLOW}"
export HF_XET_HIGH_PERFORMANCE=1

# Sanitize test_type for the JUnit and coverage filenames this step
# writes: remove commas and spaces.
STR_TEST_TYPE=$(echo "${TEST_TYPE}" | tr ', ' '_')

if [[ "${GPU_TESTS}" == "true" && "${VERIFY_GPU}" != "false" && "${PYTEST_MARKS}" != *"triton_qa"* ]]; then
  # Only GPU tests require MinIO (LoRA tests). Wait for the job service.
  echo "⏳ Waiting for MinIO to be ready..."
  MAX_ATTEMPTS=30
  ATTEMPT=0
  while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if curl -sf http://localhost:9000/minio/health/live > /dev/null 2>&1; then
      echo "✅ MinIO is ready (attempt $((ATTEMPT + 1)))"
      break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
      echo "❌ MinIO failed to start within ${MAX_ATTEMPTS}s"
      exit 1
    fi
    sleep 1
  done

  # Verify GPU visibility before spending time on collection.
  python - <<'PY'
import sys
import torch
available = torch.cuda.is_available()
count = torch.cuda.device_count() if available else 0
name = torch.cuda.get_device_name(0) if available else "None"
print(f"Python version: {sys.version}")
print(f"Is CUDA available? {available}")
print(f"GPU count: {count}")
print(f"GPU Name: {name}")
if not available:
    sys.exit("GPU tests requested but torch.cuda.is_available() is False")
PY
fi

# TRT-LLM starts MPI ranks and several runtime processes per test.
# Higher slot counts can starve startup even when VRAM fits.
# xdist resolves `-n auto` from this variable before tests/conftest.py
# saves the result as the GPU scheduler slot count.
if [[ -n "${MAX_VRAM_GIB}" && "${TEST_SUITE_NAME}" == trtllm* ]]; then
  export PYTEST_XDIST_AUTO_NUM_WORKERS=2
  echo "📊 TRT-LLM GPU parallelism: capping at 2 concurrent tests"
fi

# Run pytest with detailed output and JUnit XML
set +e  # Don't exit on test failures

if [[ -n "${PYTEST_DD_FLAKY_RETRY_ENABLED}" ]]; then
  export DD_CIVISIBILITY_FLAKY_RETRY_ENABLED="${PYTEST_DD_FLAKY_RETRY_ENABLED}"
fi
echo "Datadog flaky retries enabled: ${DD_CIVISIBILITY_FLAKY_RETRY_ENABLED:-unset}"

# Get absolute path for test-results directory and ensure it has proper permissions.
# Artifacts go to the GitHub Actions checkout so the per-job upload in
# shared-test.yml can find them.
TEST_RESULTS_DIR="${GITHUB_WORKSPACE}/test-results"
mkdir -p "${TEST_RESULTS_DIR}/status"
chmod 777 "${TEST_RESULTS_DIR}"
echo "📁 Test results will be saved to: ${TEST_RESULTS_DIR}"

# Emptied up front so the file read at the end of the job can only have
# come from this run of this stage. Truncated rather than removed: the
# distroless planner test image ships no rm (see the tool list in
# container/Dockerfile.test), and an empty file is the better signal
# anyway - absent means the stage never started, empty means it started
# and was cut off before recording an exit code.
STATUS_FILE="${TEST_RESULTS_DIR}/status/${STR_TEST_TYPE}.exitcode"
: > "${STATUS_FILE}"

# NVMe guard: GPU runners mount ${NVME_DIR} (nvme-scratch emptyDir, NVMe-backed via
# instanceStorePolicy: RAID0). CPU/ARM runners have no ${NVME_DIR} mount.
if [[ -d "${NVME_DIR}" ]]; then
  WORK_DIR="${NVME_DIR}"
  echo "✅ NVMe scratch available"
else
  WORK_DIR="/tmp"
fi

FALLBACK_HF_HOME="${WORK_DIR}/.cache/huggingface"
# Prefer the requested cache path when it is usable in this job container.
if
  [[ -n "${REQUESTED_HF_HOME}" ]] &&          # caller provided a preferred HF cache path
  mkdir -p "${REQUESTED_HF_HOME}" 2>/dev/null && # path exists or can be created; hide expected permission noise
  [[ -w "${REQUESTED_HF_HOME}" ]]             # current runner uid can write model/cache metadata there
then
  export HF_HOME="${REQUESTED_HF_HOME}"
  echo "✅ HF cache available at ${HF_HOME}"
else
  export HF_HOME="${FALLBACK_HF_HOME}"
  mkdir -p "${HF_HOME}"
  echo "⚠️  HF cache '${REQUESTED_HF_HOME}' unavailable or not writable; using ${HF_HOME}"
fi
echo "HF_HOME=${HF_HOME}" >> $GITHUB_ENV

# HOME must stay at GITHUB_WORKSPACE: ARC, upload-artifact, and checkout require it.
# The ARC runner uid (1001) may have no /etc/passwd entry, causing the shell
# to set HOME=/ by default. Pin HOME to GITHUB_WORKSPACE so downstream code
# that reads HOME (git, getpass.getuser) gets a writable path.
export HOME="${GITHUB_WORKSPACE}"

# XDG cache (flashinfer JIT, torch compile) → NVMe when available
export XDG_CACHE_HOME="${WORK_DIR}/.cache"
mkdir -p "${XDG_CACHE_HOME}"
echo "XDG_CACHE_HOME=${XDG_CACHE_HOME}" >> $GITHUB_ENV

if [[ ! -d "${CONTAINER_WORKSPACE}" ]]; then
  echo "❌ container_workspace '${CONTAINER_WORKSPACE}' not found inside container"
  exit 1
fi
echo "📦 Running pytest from container workspace: ${CONTAINER_WORKSPACE}"

# Determine pytest command based on dry_run mode
if [[ "${DRY_RUN}" == "true" ]]; then
  echo "🔍 Running pytest in dry-run mode (collect-only, no GPU required)"
  PYTEST_CMD="python -m pytest -v --collect-only -m \"${PYTEST_MARKS}\""
else
  echo "🚀 Running pytest in normal mode"
  # Determine parallelization based on parallel_mode input
  case "${PARALLEL_MODE}" in
    "auto")
      PARALLEL_OPTS="-n auto"
      echo "📊 Parallelization: auto (use all available cores)"
      ;;
    "none"|"0")
      PARALLEL_OPTS="-n 0"
      echo "📊 Parallelization: disabled (sequential execution) for GPU runs"
      ;;
    *)
      PARALLEL_OPTS="-n ${PARALLEL_MODE}"
      echo "📊 Parallelization: ${PARALLEL_MODE} workers"
      ;;
  esac

  # When max_vram_gib is set, the GPU-parallel orchestrator
  # (tests/utils/pytest_parallel_gpu.py via tests/conftest.py) takes over
  # parallelism. We pass `--max-vram-gib=N` plus `-n auto`; the conftest
  # consumes both, disables xdist, and runs tests as VRAM-aware
  # subprocesses. --dist=loadscope is omitted because xdist is bypassed.
  VRAM_OPTS=""
  if [[ -n "${MAX_VRAM_GIB}" ]]; then
    VRAM_OPTS="--max-vram-gib=${MAX_VRAM_GIB}"
    DIST_OPTS=""
    echo "🧮 GPU-parallel mode: --max-vram-gib=${MAX_VRAM_GIB} (xdist replaced by VRAM-aware orchestrator)"
  else
    # --dist=loadscope groups tests by module/class to prevent race conditions in stateful tests
    DIST_OPTS="--dist=loadscope"
  fi

  # Add coverage flags if enabled
  COVERAGE_FLAGS=""
  if [[ "${ENABLE_COVERAGE}" == "true" ]]; then
    echo "📊 Coverage collection enabled"
    COVERAGE_FLAGS="--cov=components/src/dynamo --cov=lib/bindings/python/src/dynamo --cov-report="
    # Suffix the data file per stage: the job uploads test-results/ once
    # at the end, so stages must not overwrite each other. The nightly
    # merge globs `.coverage*` and combines whatever it finds.
    export COVERAGE_FILE="${TEST_RESULTS_DIR}/.coverage.${STR_TEST_TYPE}"
    export PYTHONPATH="${CONTAINER_WORKSPACE}/components/src:${CONTAINER_WORKSPACE}/lib/bindings/python/src${PYTHONPATH:+:${PYTHONPATH}}"
  fi

  PYTEST_CMD="python -m pytest ${PARALLEL_OPTS} ${VRAM_OPTS} ${DIST_OPTS} --continue-on-collection-errors -v --tb=short --basetemp=${WORK_DIR}/pytest_temp -o cache_dir=${WORK_DIR}/.pytest_cache --junitxml=${TEST_RESULTS_DIR}/${PYTEST_XML_FILE} --alluredir=${TEST_RESULTS_DIR}/allure-results --durations=10 ${COVERAGE_FLAGS} -m \"${PYTEST_MARKS}\""
fi

echo "▶️ Executing (cwd=${CONTAINER_WORKSPACE}): ${PYTEST_CMD}"

umask 0022
cd "${CONTAINER_WORKSPACE}"
bash -c "${PYTEST_CMD}"

TEST_EXIT_CODE=$?
echo "🧪 Tests completed with exit code: ${TEST_EXIT_CODE}"
# Back to fail-fast: a failure while processing results must fail the
# step instead of being masked by the pytest status returned below.
set -e

# Process test results. The JUnit file gets a unique per-stage name so
# several stages can share one test-results/ directory and one upload.
JUNIT_FILE="${TEST_RESULTS_DIR}/${PYTEST_XML_FILE}"
if [[ -f "$JUNIT_FILE" ]]; then
  echo "✅ JUnit XML generated successfully"
  TOTAL_TESTS=$(grep -o 'tests="[0-9]*"' "$JUNIT_FILE" | grep -o '[0-9]*' | head -1 || echo "0")
  FAILED_TESTS=$(grep -o 'failures="[0-9]*"' "$JUNIT_FILE" | grep -o '[0-9]*' | head -1 || echo "0")
  ERROR_TESTS=$(grep -o 'errors="[0-9]*"' "$JUNIT_FILE" | grep -o '[0-9]*' | head -1 || echo "0")
  echo "📊 ${TOTAL_TESTS} tests completed (${FAILED_TESTS} failed, ${ERROR_TESTS} errors)"

  JUNIT_NAME="pytest_test_report_${TEST_SUITE_NAME}_${STR_TEST_TYPE}_${PLATFORM_ARCH}_${RUN_ID}_${JOB_CHECK_RUN_ID}.xml"
  mv "$JUNIT_FILE" "${TEST_RESULTS_DIR}/${JUNIT_NAME}"
  echo "📝 Renamed XML file to: ${JUNIT_NAME}"
elif [[ "${DRY_RUN}" != "true" ]]; then
  echo "⚠️  JUnit XML file not found - test results may not be available for upload"
fi

# Checked by "Verify test stages completed" in shared-test.yml, which
# cannot use the exit below or $GITHUB_ENV: a truncated script is still
# reported as a successful step and its $GITHUB_ENV writes are dropped.
# Keep this last - its absence is what proves the script was truncated.
echo "${TEST_EXIT_CODE}" > "${STATUS_FILE}"

exit ${TEST_EXIT_CODE}
