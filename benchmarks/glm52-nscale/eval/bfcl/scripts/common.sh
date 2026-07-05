#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/upstream.env"

BFCL_CHECKOUT_DIR="${BFCL_CHECKOUT_DIR:-${ROOT_DIR}/.cache/gorilla}"
BFCL_VENV_DIR="${BFCL_VENV_DIR:-${ROOT_DIR}/.venv}"
BFCL_PYTHON="${BFCL_VENV_DIR}/bin/python"
BFCL_BIN="${BFCL_VENV_DIR}/bin/bfcl"
BFCL_MODEL="${BFCL_MODEL:-zai-org/GLM-5.2-FC}"
BFCL_ARTIFACT_ROOT="${BFCL_ARTIFACT_ROOT:-${ROOT_DIR}/outputs}"
BFCL_NUM_THREADS="${BFCL_NUM_THREADS:-16}"
BFCL_TEMPERATURE="${BFCL_TEMPERATURE:-0}"
BFCL_MAX_TOKENS="${BFCL_MAX_TOKENS:-64000}"
BFCL_INCLUDE_INPUT_LOG="${BFCL_INCLUDE_INPUT_LOG:-1}"
CAMPAIGN_SOURCE_METADATA="${CAMPAIGN_SOURCE_METADATA:-/workspace/source-provenance.json}"
CAMPAIGN_SOURCE_ROOT="${CAMPAIGN_SOURCE_ROOT:-/workspace}"
export BFCL_MODEL BFCL_NUM_THREADS BFCL_TEMPERATURE BFCL_MAX_TOKENS
export GLM52_OPENAI_MAX_TOKENS="${BFCL_MAX_TOKENS}"

validate_variant() {
  case "$1" in
    dynamo-vllm|vllm-serve|dynamo-sglang|sglang-serve) ;;
    *)
      echo "Unknown variant '$1'; expected dynamo-vllm, vllm-serve, dynamo-sglang, or sglang-serve." >&2
      exit 2
      ;;
  esac
}

validate_campaign_phase() {
  local mode="$1"
  local phase="$2"
  case "${mode}:${phase}" in
    full:ab|full:ba|smoke:ab|smoke:ba|smoke:validation) ;;
    *)
      echo "Invalid ${mode} BFCL campaign phase '${phase}'." >&2
      exit 2
      ;;
  esac
}

require_install() {
  local current_commit
  if [[ ! -x "${BFCL_BIN}" ]]; then
    echo "BFCL is not bootstrapped. Run ${SCRIPT_DIR}/bootstrap.sh first." >&2
    exit 1
  fi

  current_commit="$(git -C "${BFCL_CHECKOUT_DIR}" rev-parse HEAD)"
  if [[ "${current_commit}" != "${BFCL_GORILLA_COMMIT}" ]]; then
    echo "BFCL commit mismatch: expected ${BFCL_GORILLA_COMMIT}, got ${current_commit}." >&2
    exit 1
  fi
  if ! git -C "${BFCL_CHECKOUT_DIR}" apply --reverse --check \
    "${ROOT_DIR}/patches/0001-glm52-openai-chat-completions.patch" 2>/dev/null; then
    echo "BFCL checkout does not contain the exact campaign adapter patch; rerun bootstrap.sh." >&2
    exit 1
  fi
  "${BFCL_PYTHON}" "${SCRIPT_DIR}/capture_metadata.py" \
    --checkout "${BFCL_CHECKOUT_DIR}" \
    --patch "${ROOT_DIR}/patches/0001-glm52-openai-chat-completions.patch" \
    --verify-only >/dev/null
}

require_endpoint() {
  if [[ -z "${GLM52_OPENAI_BASE_URL:-}" ]]; then
    echo "GLM52_OPENAI_BASE_URL is required and must include /v1." >&2
    exit 1
  fi

  case "${GLM52_OPENAI_BASE_URL%/}" in
    */v1) ;;
    *)
      echo "GLM52_OPENAI_BASE_URL must end in /v1: ${GLM52_OPENAI_BASE_URL}" >&2
      exit 1
      ;;
  esac
}

new_run_dir() {
  local variant="$1"
  local mode="$2"
  local phase="$3"
  local requested="${4:-}"
  validate_campaign_phase "${mode}" "${phase}"
  if [[ -n "${requested}" ]]; then
    case "/${requested//_/-}/" in
      *"-${phase}-"*|*"/${phase}/"*) ;;
      *)
        echo "Requested RUN_DIR must contain campaign phase token '${phase}': ${requested}" >&2
        exit 2
        ;;
    esac
    printf '%s\n' "${requested}"
  else
    printf '%s/%s/%s-%s-%s\n' \
      "${BFCL_ARTIFACT_ROOT}" "${variant}" "${mode}" "${phase}" "$(date -u +%Y%m%dT%H%M%SZ)"
  fi
}

prepare_run() {
  local variant="$1"
  local mode="$2"
  local phase="$3"
  local categories="$4"
  local run_dir="$5"
  local runtime_binding="${BFCL_RUNTIME_BINDING:-/artifacts/glm52-nscale/runtime-bindings/${variant}/active.json}"

  validate_campaign_phase "${mode}" "${phase}"
  if [[ ! -s "${runtime_binding}" ]]; then
    echo "Runtime binding is missing: ${runtime_binding}" >&2
    exit 1
  fi

  mkdir -p "${run_dir}/logs"
  export BFCL_PROJECT_ROOT="${run_dir}"

  "${BFCL_PYTHON}" "${SCRIPT_DIR}/verify_environment_lock.py" \
    --lock "${ROOT_DIR}/constraints.lock" \
    --freeze-output "${run_dir}/environment.freeze.txt" \
    --metadata-output "${run_dir}/environment-lock.json"

  "${BFCL_PYTHON}" "${SCRIPT_DIR}/endpoint_preflight.py" \
    --model "zai-org/GLM-5.2" \
    --expected-context-window 409600 \
    --output "${run_dir}/endpoint-models.json"

  "${BFCL_PYTHON}" "${SCRIPT_DIR}/capture_metadata.py" \
    --run-dir "${run_dir}" \
    --variant "${variant}" \
    --mode "${mode}" \
    --campaign-phase "${phase}" \
    --categories "${categories}" \
    --checkout "${BFCL_CHECKOUT_DIR}" \
    --patch "${ROOT_DIR}/patches/0001-glm52-openai-chat-completions.patch" \
    --endpoint-models "${run_dir}/endpoint-models.json" \
    --runtime-binding "${runtime_binding}" \
    --environment-lock "${run_dir}/environment-lock.json" \
    --campaign-source-metadata "${CAMPAIGN_SOURCE_METADATA}" \
    --campaign-source-root "${CAMPAIGN_SOURCE_ROOT}"
}

run_logged() {
  local log_file="$1"
  local start_epoch
  local status
  local wall_seconds
  shift
  start_epoch="$(date +%s)"
  printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${log_file}"
  printf 'command:' | tee -a "${log_file}"
  printf ' %q' "$@" | tee -a "${log_file}"
  printf '\n' | tee -a "${log_file}"
  set +e
  "$@" 2>&1 | tee -a "${log_file}"
  status="${PIPESTATUS[0]}"
  set -e
  wall_seconds="$(( $(date +%s) - start_epoch ))"
  printf '\nfinished_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${log_file}"
  printf 'exit_status=%s\nwall_seconds=%s\n' "${status}" "${wall_seconds}" | tee -a "${log_file}"
  return "${status}"
}

require_serpapi_for_categories() {
  local categories="$1"
  case ",${categories}," in
    *,all,*|*,all_scoring,*|*,agentic,*|*,web_search,*|*,web_search_base,*|*,web_search_no_snippet,*)
      if [[ -z "${SERPAPI_API_KEY:-}" ]]; then
        echo "SERPAPI_API_KEY is required for BFCL v4 web-search categories." >&2
        exit 1
      fi
      ;;
  esac
}
