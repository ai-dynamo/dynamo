#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

require_value() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "${name} must be set" >&2
    exit 2
  fi
}

require_value DYNAMO_BASE_URL
require_value DYNAMO_MODEL_ALIAS
require_value TASK_IDS_FILE
require_value DYN_REQUEST_TRACE_OUTPUT_PATH

HARBOR_COMMAND="${HARBOR_COMMAND:-harbor}"
HARBOR_VERSION="${HARBOR_VERSION:-0.21.0}"
HARBOR_DATASET="${HARBOR_DATASET:-swebenchpro@1.0}"
HARBOR_CONCURRENCY="${HARBOR_CONCURRENCY:-1}"
EXPECTED_TASK_COUNT="${EXPECTED_TASK_COUNT:-5}"
MINIMUM_REQUESTS_PER_SESSION="${MINIMUM_REQUESTS_PER_SESSION:-4}"
TRACE_VALIDATION_TIMEOUT_SECONDS="${TRACE_VALIDATION_TIMEOUT_SECONDS:-30}"
RESULTS_DIR="${RESULTS_DIR:-${PWD}/agent-harness-nightly-results}"
HARBOR_JOBS_DIR="${HARBOR_JOBS_DIR:-${RESULTS_DIR}/harbor-jobs}"
DYNAMO_API_KEY="${DYNAMO_API_KEY:-dummy}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-$(date -u +%Y%m%dT%H%M%SZ)}"

if ! command -v "${HARBOR_COMMAND}" >/dev/null 2>&1; then
  echo "Harbor executable not found: ${HARBOR_COMMAND}" >&2
  exit 2
fi
harbor_version_output="$("${HARBOR_COMMAND}" --version 2>&1)"
if [[ ! "${harbor_version_output}" =~ (^|[^0-9])${HARBOR_VERSION//./\.}([^0-9]|$) ]]; then
  echo "Expected Harbor ${HARBOR_VERSION}, got: ${harbor_version_output}" >&2
  exit 2
fi
if [[ ! -r "${TASK_IDS_FILE}" ]]; then
  echo "TASK_IDS_FILE is not readable: ${TASK_IDS_FILE}" >&2
  exit 2
fi
if [[ ! "${HARBOR_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]]; then
  echo "HARBOR_CONCURRENCY must be a positive integer" >&2
  exit 2
fi
if [[ ! "${EXPECTED_TASK_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EXPECTED_TASK_COUNT must be a positive integer" >&2
  exit 2
fi
if [[ ! "${MINIMUM_REQUESTS_PER_SESSION}" =~ ^[0-9]+$ ]] || \
   (( MINIMUM_REQUESTS_PER_SESSION < 2 )); then
  echo "MINIMUM_REQUESTS_PER_SESSION must be an integer of at least 2" >&2
  exit 2
fi
if [[ ! "${TRACE_VALIDATION_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRACE_VALIDATION_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 2
fi
if [[ "${DYNAMO_MODEL_ALIAS}" == */* ]]; then
  echo "DYNAMO_MODEL_ALIAS must not contain '/'; Codex strips provider prefixes" >&2
  exit 2
fi
if [[ ! "${DYNAMO_MODEL_ALIAS}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "DYNAMO_MODEL_ALIAS may contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
fi
if [[ ! "${RUN_NAME_SUFFIX}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_NAME_SUFFIX may contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
fi

TASK_IDS=()
while IFS= read -r task_id; do
  TASK_IDS+=("${task_id}")
done < <(
  sed \
    -e 's/[[:space:]]*#.*$//' \
    -e 's/^[[:space:]]*//' \
    -e 's/[[:space:]]*$//' \
    -e '/^$/d' \
    "${TASK_IDS_FILE}"
)
if [[ ${#TASK_IDS[@]} -eq 0 ]]; then
  echo "TASK_IDS_FILE did not contain any task IDs" >&2
  exit 2
fi
if [[ ${#TASK_IDS[@]} -ne ${EXPECTED_TASK_COUNT} ]]; then
  echo "Expected ${EXPECTED_TASK_COUNT} task IDs, found ${#TASK_IDS[@]}" >&2
  exit 2
fi
unique_task_count="$(printf '%s\n' "${TASK_IDS[@]}" | sort -u | wc -l)"
if [[ ${unique_task_count} -ne ${#TASK_IDS[@]} ]]; then
  echo "TASK_IDS_FILE contains duplicate task IDs" >&2
  exit 2
fi

mkdir -p "${RESULTS_DIR}" "${HARBOR_JOBS_DIR}"
OPENAI_BASE_URL="${DYNAMO_BASE_URL%/}"
if [[ "${OPENAI_BASE_URL}" != */v1 ]]; then
  OPENAI_BASE_URL="${OPENAI_BASE_URL}/v1"
fi
ANTHROPIC_BASE_URL="${OPENAI_BASE_URL%/v1}"
DYNAMO_ALLOWED_HOST="$(python3 - "${DYNAMO_BASE_URL}" <<'PY'
import sys
from urllib.parse import urlparse

hostname = urlparse(sys.argv[1]).hostname
if not hostname:
    raise SystemExit(f"Could not determine endpoint hostname from {sys.argv[1]!r}")
print(hostname)
PY
)"

python3 - "${OPENAI_BASE_URL}" "${DYNAMO_MODEL_ALIAS}" "${DYNAMO_API_KEY}" <<'PY'
import json
import sys
import urllib.request

base_url, expected_model, api_key = sys.argv[1:]
request = urllib.request.Request(
    f"{base_url}/models",
    headers={"Authorization": f"Bearer {api_key}"},
)
with urllib.request.urlopen(request, timeout=30) as response:
    payload = json.load(response)
models = {
    item.get("id")
    for item in payload.get("data", [])
    if isinstance(item, dict)
}
if expected_model not in models:
    raise SystemExit(
        f"Dynamo endpoint does not advertise {expected_model!r}; found {sorted(models)}"
    )
PY

task_args=()
for task_id in "${TASK_IDS[@]}"; do
  task_args+=(-i "${task_id}")
done

run_harness() {
  local harness="$1"
  local trace_start
  local job_name
  local log_path
  local summary_path
  local validation_log
  local validation_status
  local validation_deadline
  local -a agent_args

  trace_start=0
  if [[ -f "${DYN_REQUEST_TRACE_OUTPUT_PATH}" ]]; then
    trace_start="$(wc -l < "${DYN_REQUEST_TRACE_OUTPUT_PATH}")"
  fi
  job_name="dynamo-nightly-${DYNAMO_MODEL_ALIAS}-${harness}-${RUN_NAME_SUFFIX}"
  log_path="${RESULTS_DIR}/${harness}.log"
  summary_path="${RESULTS_DIR}/${harness}-request-trace-summary.json"
  validation_log="${RESULTS_DIR}/${harness}-request-trace-validation.log"

  case "${harness}" in
    claude-code)
      agent_args=(
        -a claude-code
        --ae "ANTHROPIC_BASE_URL=${ANTHROPIC_BASE_URL}"
        --ae "ANTHROPIC_API_KEY=${DYNAMO_API_KEY}"
        --ae "ANTHROPIC_AUTH_TOKEN=${DYNAMO_API_KEY}"
        --ae "CLAUDE_CODE_MAX_OUTPUT_TOKENS=4096"
        --ae "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1"
      )
      ;;
    codex)
      agent_args=(
        -a codex
        --ae "OPENAI_BASE_URL=${OPENAI_BASE_URL}"
        --ae "OPENAI_API_KEY=${DYNAMO_API_KEY}"
      )
      ;;
    *)
      echo "unsupported harness: ${harness}" >&2
      return 2
      ;;
  esac

  "${HARBOR_COMMAND}" run \
    -d "${HARBOR_DATASET}" \
    "${task_args[@]}" \
    "${agent_args[@]}" \
    -m "${DYNAMO_MODEL_ALIAS}" \
    --allow-agent-host "${DYNAMO_ALLOWED_HOST}" \
    --n-concurrent "${HARBOR_CONCURRENCY}" \
    --agent-setup-timeout-multiplier 10 \
    --no-delete \
    --job-name "${job_name}" \
    --jobs-dir "${HARBOR_JOBS_DIR}" \
    -y 2>&1 | tee "${log_path}"

  validation_deadline=$((SECONDS + TRACE_VALIDATION_TIMEOUT_SECONDS))
  while true; do
    set +e
    python3 "${SCRIPT_DIR}/validate_request_trace.py" \
      "${DYN_REQUEST_TRACE_OUTPUT_PATH}" \
      --start-line "${trace_start}" \
      --expected-model "${DYNAMO_MODEL_ALIAS}" \
      --minimum-root-sessions "${#TASK_IDS[@]}" \
      --minimum-requests-per-session "${MINIMUM_REQUESTS_PER_SESSION}" \
      --output "${summary_path}" >"${validation_log}" 2>&1
    validation_status=$?
    set -e
    if [[ ${validation_status} -eq 0 ]]; then
      cat "${validation_log}"
      break
    fi
    if (( SECONDS >= validation_deadline )); then
      cat "${validation_log}" >&2
      return "${validation_status}"
    fi
    sleep 1
  done
}

run_harness claude-code
run_harness codex
