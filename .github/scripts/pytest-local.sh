#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Runs one pytest stage of a shared-test.yml job, inside the job container.
# It replaces the former .github/actions/pytest-local composite action: loading
# that action made the runner download the whole repository for every job, and
# the k8s-novolume hook then copied it into the GPU pod file by file.
#
# Configured through the step environment. Never interpolate workflow
# expressions into this file (zizmor "template-injection").
#   Required: PYTEST_MARKS, CONTAINER_WORKSPACE, TEST_SUITE_NAME, TEST_TYPE,
#             PLATFORM_ARCH, PARALLEL_MODE (auto | none | 0 | <workers>),
#             JOB_CHECK_RUN_ID
#   Optional: ENABLE_COVERAGE=true, GPU_TESTS=true, VERIFY_GPU=false (skips the
#             MinIO and GPU checks), MAX_VRAM_GIB, PYTEST_DD_FLAKY_RETRY_ENABLED,
#             SAFE_SUBSET_MARKER, SAFE_SUBSET_PATHS (space-separated)
#
# SAFE_SUBSET_MARKER tests under SAFE_SUBSET_PATHS run first on 2 xdist workers,
# then the rest; merging their JUnit reports needs python in the image.
#
# The distroless planner test image ships only the tools listed in
# container/Dockerfile.test, so stay on bash builtins and those tools.
set -euo pipefail

for var in PYTEST_MARKS CONTAINER_WORKSPACE TEST_SUITE_NAME TEST_TYPE \
  PLATFORM_ARCH PARALLEL_MODE JOB_CHECK_RUN_ID; do
  if [[ -z "${!var:-}" ]]; then
    echo "::error::pytest-local.sh: ${var} is not set"
    exit 1
  fi
done

# Read by tests/conftest.py for Allure labels.
export DYNAMO_TEST_PLATFORM="${PLATFORM_ARCH}"
export DYNAMO_TEST_TYPE="${TEST_TYPE}"
export DYNAMO_TEST_WORKFLOW="${GITHUB_WORKFLOW}"
export HF_XET_HIGH_PERFORMANCE=1

if [[ "${GPU_TESTS:-}" == "true" && "${VERIFY_GPU:-}" != "false" && "${PYTEST_MARKS}" != *"triton_qa"* ]]; then
  # Only GPU tests require MinIO (LoRA tests). Wait for the job service.
  echo "⏳ Waiting for MinIO to be ready..."
  for attempt in {1..30}; do
    curl -sf http://localhost:9000/minio/health/live > /dev/null 2>&1 && break
    if (( attempt == 30 )); then
      echo "❌ MinIO failed to start within 30s"
      exit 1
    fi
    sleep 1
  done
  echo "✅ MinIO is ready (attempt ${attempt})"

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
if [[ -n "${MAX_VRAM_GIB:-}" && "${TEST_SUITE_NAME}" == trtllm* ]]; then
  export PYTEST_XDIST_AUTO_NUM_WORKERS=2
  echo "📊 TRT-LLM GPU parallelism: capping at 2 concurrent tests"
fi

if [[ -n "${PYTEST_DD_FLAKY_RETRY_ENABLED:-}" ]]; then
  export DD_CIVISIBILITY_FLAKY_RETRY_ENABLED="${PYTEST_DD_FLAKY_RETRY_ENABLED}"
fi
echo "Datadog flaky retries enabled: ${DD_CIVISIBILITY_FLAKY_RETRY_ENABLED:-unset}"

# Artifacts go to the GitHub Actions checkout so the uploads in
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
STATUS_FILE="${TEST_RESULTS_DIR}/status/${TEST_TYPE}.exitcode"
: > "${STATUS_FILE}"

# GPU runners mount /scratch (nvme-scratch emptyDir, NVMe-backed via
# instanceStorePolicy: RAID0). CPU/ARM runners have no /scratch mount.
WORK_DIR=/tmp
if [[ -d /scratch ]]; then
  WORK_DIR=/scratch
  echo "✅ NVMe scratch available"
fi

# Prefer the shared model cache at /models when this job container can
# write it; mkdir errors are expected on runners without that mount.
if mkdir -p /models 2>/dev/null && [[ -w /models ]]; then
  export HF_HOME=/models
  echo "✅ HF cache available at ${HF_HOME}"
else
  export HF_HOME="${WORK_DIR}/.cache/huggingface"
  mkdir -p "${HF_HOME}"
  echo "⚠️  HF cache '/models' unavailable or not writable; using ${HF_HOME}"
fi
echo "HF_HOME=${HF_HOME}" >> "${GITHUB_ENV}"

# HOME must stay at GITHUB_WORKSPACE: ARC, upload-artifact, and checkout require it.
# The ARC runner uid (1001) may have no /etc/passwd entry, causing the shell
# to set HOME=/ by default. Pin HOME to GITHUB_WORKSPACE so downstream code
# that reads HOME (git, getpass.getuser) gets a writable path.
export HOME="${GITHUB_WORKSPACE}"

# XDG cache (flashinfer JIT, torch compile) → NVMe when available
export XDG_CACHE_HOME="${WORK_DIR}/.cache"
mkdir -p "${XDG_CACHE_HOME}"
echo "XDG_CACHE_HOME=${XDG_CACHE_HOME}" >> "${GITHUB_ENV}"

if [[ ! -d "${CONTAINER_WORKSPACE}" ]]; then
  echo "❌ container_workspace '${CONTAINER_WORKSPACE}' not found inside container"
  exit 1
fi
echo "📦 Running pytest from container workspace: ${CONTAINER_WORKSPACE}"

case "${PARALLEL_MODE}" in
  auto)
    pytest_args=(-n auto)
    echo "📊 Parallelization: auto (use all available cores)"
    ;;
  none | 0)
    pytest_args=(-n 0)
    echo "📊 Parallelization: disabled (sequential execution)"
    ;;
  *)
    pytest_args=(-n "${PARALLEL_MODE}")
    echo "📊 Parallelization: ${PARALLEL_MODE} workers"
    ;;
esac

if [[ -n "${MAX_VRAM_GIB:-}" ]]; then
  # The GPU-parallel orchestrator (tests/utils/pytest_parallel_gpu.py via
  # tests/conftest.py) consumes --max-vram-gib plus -n, disables xdist, and
  # runs tests as VRAM-aware subprocesses, so --dist=loadscope is omitted.
  pytest_args+=("--max-vram-gib=${MAX_VRAM_GIB}")
  echo "🧮 GPU-parallel mode: --max-vram-gib=${MAX_VRAM_GIB} (xdist replaced by VRAM-aware orchestrator)"
else
  # --dist=loadscope groups tests by module/class to prevent race conditions in stateful tests
  pytest_args+=(--dist=loadscope)
fi

JUNIT_FILE="${TEST_RESULTS_DIR}/pytest_test_report.xml"
REMAINDER_MARKS="${PYTEST_MARKS}"
if [[ -n "${SAFE_SUBSET_MARKER:-}" ]]; then
  # Phase reports stay out of the test-results/pytest_test_report_*.xml upload
  # glob; only the merged report below matches it.
  PHASE_DIR="${TEST_RESULTS_DIR}/phases"
  mkdir -p "${PHASE_DIR}"
  rm -f "${PHASE_DIR}/safe_subset.xml" "${PHASE_DIR}/remainder.xml"
  JUNIT_FILE="${PHASE_DIR}/remainder.xml"
  REMAINDER_MARKS="(${PYTEST_MARKS}) and not (${SAFE_SUBSET_MARKER})"
fi
pytest_args+=(
  --continue-on-collection-errors -v --tb=short
  "--basetemp=${WORK_DIR}/pytest_temp"
  -o "cache_dir=${WORK_DIR}/.pytest_cache"
  "--junitxml=${JUNIT_FILE}"
  "--alluredir=${TEST_RESULTS_DIR}/allure-results"
  --durations=10
)

if [[ "${ENABLE_COVERAGE:-}" == "true" ]]; then
  echo "📊 Coverage collection enabled"
  cov_args=(--cov=components/src/dynamo --cov=lib/bindings/python/src/dynamo --cov-report=)
  pytest_args+=("${cov_args[@]}")
  # Suffix the data file per stage: the job uploads test-results/ once
  # at the end, so stages must not overwrite each other. The nightly
  # merge globs `.coverage*` and combines whatever it finds.
  export COVERAGE_FILE="${TEST_RESULTS_DIR}/.coverage.${TEST_TYPE}"
  export PYTHONPATH="${CONTAINER_WORKSPACE}/components/src:${CONTAINER_WORKSPACE}/lib/bindings/python/src${PYTHONPATH:+:${PYTHONPATH}}"
fi

pytest_args+=(-m "${REMAINDER_MARKS}")

umask 0022
cd "${CONTAINER_WORKSPACE}"
# Record pytest's exit code rather than failing here, so results are still
# renamed and the status file is written.
TEST_EXIT_CODE=0
JUNIT_NAME="pytest_test_report_${TEST_SUITE_NAME}_${TEST_TYPE}_${PLATFORM_ARCH}_${GITHUB_RUN_ID}_${JOB_CHECK_RUN_ID}.xml"

if [[ -z "${SAFE_SUBSET_MARKER:-}" ]]; then
  echo "▶️ Executing (cwd=${CONTAINER_WORKSPACE}): python -m pytest ${pytest_args[*]}"
  python -m pytest "${pytest_args[@]}" || TEST_EXIT_CODE=$?
  echo "🧪 Tests completed with exit code: ${TEST_EXIT_CODE}"

  # Unique per-stage name so several stages can share one test-results/
  # directory and one upload.
  if [[ -f "${JUNIT_FILE}" ]]; then
    mv "${JUNIT_FILE}" "${TEST_RESULTS_DIR}/${JUNIT_NAME}"
    echo "📝 Renamed JUnit XML to: ${JUNIT_NAME}"
  else
    echo "⚠️  JUnit XML file not found - test results may not be available for upload"
  fi
else
  # Every SAFE_SUBSET_MARKER test must live under SAFE_SUBSET_PATHS, or it runs
  # in neither phase; the paths avoid a second full-repository collection.
  read -ra safe_paths <<< "${SAFE_SUBSET_PATHS:-}"
  # --dist=load, not loadscope: loadscope would keep the whole module on one
  # worker.
  safe_args=(
    -n 2 --dist=load
    --continue-on-collection-errors -v --tb=short
    "--basetemp=${WORK_DIR}/pytest_temp_safe_subset"
    -o "cache_dir=${WORK_DIR}/.pytest_cache_safe_subset"
    "--junitxml=${PHASE_DIR}/safe_subset.xml"
    "--alluredir=${TEST_RESULTS_DIR}/allure-results"
    --durations=10
  )
  safe_env=()
  if [[ "${ENABLE_COVERAGE:-}" == "true" ]]; then
    safe_args+=("${cov_args[@]}")
    # Must not start with "${COVERAGE_FILE}.": pytest-cov combines (and
    # deletes) files matching <data_file>.* at the end of the remainder phase.
    safe_env=("COVERAGE_FILE=${TEST_RESULTS_DIR}/.coverage.${TEST_TYPE}-safe_subset")
  fi
  safe_args+=(-m "(${PYTEST_MARKS}) and (${SAFE_SUBSET_MARKER})" "${safe_paths[@]}")

  # Exit code 5 (nothing collected) is deliberately not special-cased: an
  # empty safe subset fails the stage, as an empty single phase does.
  SAFE_EXIT_CODE=0
  echo "▶️ Executing safe-subset phase (cwd=${CONTAINER_WORKSPACE}): ${safe_env[*]:+${safe_env[*]} }python -m pytest ${safe_args[*]}"
  PHASE_START=${SECONDS}
  env "${safe_env[@]}" python -m pytest "${safe_args[@]}" || SAFE_EXIT_CODE=$?
  echo "🧪 Safe-subset phase completed with exit code ${SAFE_EXIT_CODE} in $((SECONDS - PHASE_START))s"

  REMAINDER_EXIT_CODE=0
  echo "▶️ Executing remainder phase (cwd=${CONTAINER_WORKSPACE}): python -m pytest ${pytest_args[*]}"
  PHASE_START=${SECONDS}
  python -m pytest "${pytest_args[@]}" || REMAINDER_EXIT_CODE=$?
  echo "🧪 Remainder phase completed with exit code ${REMAINDER_EXIT_CODE} in $((SECONDS - PHASE_START))s"

  TEST_EXIT_CODE=${SAFE_EXIT_CODE}
  if (( TEST_EXIT_CODE == 0 )); then
    TEST_EXIT_CODE=${REMAINDER_EXIT_CODE}
  fi
  echo "🧪 Tests completed with exit code: ${TEST_EXIT_CODE}"

  phase_reports=()
  for report in "${PHASE_DIR}/safe_subset.xml" "${PHASE_DIR}/remainder.xml"; do
    if [[ -f "${report}" ]]; then
      phase_reports+=("${report}")
    else
      echo "⚠️  JUnit XML ${report} not found"
    fi
  done
  if (( ${#phase_reports[@]} > 0 )); then
    # One report per stage: the metrics uploader and test-results consumers
    # expect one file, and read every <testsuite> under the root.
    MERGE_EXIT_CODE=0
    python - "${TEST_RESULTS_DIR}/${JUNIT_NAME}" "${phase_reports[@]}" <<'PY' || MERGE_EXIT_CODE=$?
import sys
import xml.etree.ElementTree as ET

out, *inputs = sys.argv[1:]
merged = ET.Element("testsuites", name="pytest tests")
for path in inputs:
    root = ET.parse(path).getroot()
    merged.extend([root] if root.tag == "testsuite" else root.findall("testsuite"))
ET.ElementTree(merged).write(out, encoding="utf-8", xml_declaration=True)
PY
    if (( MERGE_EXIT_CODE == 0 )); then
      echo "📝 Merged ${#phase_reports[@]} phase JUnit XML file(s) into: ${JUNIT_NAME}"
    else
      echo "❌ Merging phase JUnit XML failed with exit code ${MERGE_EXIT_CODE}"
      if (( TEST_EXIT_CODE == 0 )); then
        TEST_EXIT_CODE=${MERGE_EXIT_CODE}
      fi
    fi
  else
    echo "⚠️  JUnit XML file not found - test results may not be available for upload"
  fi
fi

# Checked by "Verify test stages completed" in shared-test.yml, which
# cannot use the exit below or $GITHUB_ENV: a truncated script is still
# reported as a successful step and its $GITHUB_ENV writes are dropped.
# Keep this last - its absence is what proves the script was truncated.
echo "${TEST_EXIT_CODE}" > "${STATUS_FILE}"

exit "${TEST_EXIT_CODE}"
