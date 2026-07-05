# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared paths and immutable pins for the GLM-5.2 SWE-bench campaign.

SWEBENCH_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEBENCH_EVAL_DIR="$(cd "${SWEBENCH_SCRIPT_DIR}/.." && pwd)"

# shellcheck source=../pins.env
source "${SWEBENCH_EVAL_DIR}/pins.env"

assert_pin() {
  local name="$1"
  local expected="$2"
  local actual="${!name:-}"
  if [[ "${actual}" != "${expected}" ]]; then
    printf 'invalid %s: expected %s, got %s\n' "${name}" "${expected}" "${actual:-<unset>}" >&2
    return 1
  fi
}

# Fail closed if the campaign-wide pin file drifts from the audited revisions.
assert_pin MINI_SWE_AGENT_VERSION 2.4.4
assert_pin MINI_SWE_AGENT_COMMIT 4fe36a38941abde8a332bda950ca6c0de653a19f
assert_pin SWEBENCH_VERSION 4.1.0
assert_pin SWEBENCH_COMMIT 726c5461e2ef52d83cf1ea2107870a8bb3328d57
assert_pin SWEBENCH_PRO_COMMIT ca10a60a5fcae51e6948ffe1485d4153d421e6c5
assert_pin SWEBENCH_VERIFIED_REVISION 91aa3ed51b709be6457e12d00300a6a596d4c6a3
assert_pin SWEBENCH_MULTILINGUAL_REVISION 2b7aced941b4873e9cad3e76abbae93f481d1beb
assert_pin SWEBENCH_PRO_REVISION 7ab5114912baf22bb098818e604c02fe7ad2c11f
assert_pin SWEBENCH_VERIFIED_CASES 500
assert_pin SWEBENCH_MULTILINGUAL_CASES 300
assert_pin SWEBENCH_PRO_PUBLIC_CASES 731

SWEBENCH_WORK_ROOT="${SWEBENCH_WORK_ROOT:-${HOME}/.cache/dynamo-glm52/swebench}"
SWEBENCH_RESULTS_ROOT="${SWEBENCH_RESULTS_ROOT:-${SWEBENCH_WORK_ROOT}/results}"
SWEBENCH_VENV="${SWEBENCH_WORK_ROOT}/.venv"
SWEBENCH_REPOS_DIR="${SWEBENCH_WORK_ROOT}/repos"
MINI_SWE_AGENT_REPO="${SWEBENCH_REPOS_DIR}/mini-swe-agent"
SWEBENCH_EVALUATOR_REPO="${SWEBENCH_REPOS_DIR}/SWE-bench"
SWEBENCH_PRO_REPO="${SWEBENCH_REPOS_DIR}/SWE-bench_Pro-os"
SWEBENCH_DATA_ROOT="${SWEBENCH_WORK_ROOT}/datasets"
SWEBENCH_MSWEA_CONFIG_DIR="${SWEBENCH_WORK_ROOT}/mini-swe-agent-config"

suite_expected_count() {
  case "$1" in
    verified) printf '%s\n' "${SWEBENCH_VERIFIED_CASES}" ;;
    multilingual) printf '%s\n' "${SWEBENCH_MULTILINGUAL_CASES}" ;;
    pro) printf '%s\n' "${SWEBENCH_PRO_PUBLIC_CASES}" ;;
    *) printf 'unknown SWE-bench suite: %s\n' "$1" >&2; return 2 ;;
  esac
}

suite_agent_dataset() {
  case "$1" in
    verified|multilingual|pro) printf '%s/agent/%s\n' "${SWEBENCH_DATA_ROOT}" "$1" ;;
    *) printf 'unknown SWE-bench suite: %s\n' "$1" >&2; return 2 ;;
  esac
}

suite_evaluator_dataset() {
  case "$1" in
    verified|multilingual|pro) printf '%s/evaluator/%s.jsonl\n' "${SWEBENCH_DATA_ROOT}" "$1" ;;
    *) printf 'unknown SWE-bench suite: %s\n' "$1" >&2; return 2 ;;
  esac
}

require_bootstrap() {
  local path
  for path in \
    "${SWEBENCH_VENV}/bin/python" \
    "${SWEBENCH_VENV}/bin/mini-extra" \
    "${MINI_SWE_AGENT_REPO}/.git" \
    "${SWEBENCH_EVALUATOR_REPO}/.git" \
    "${SWEBENCH_PRO_REPO}/.git" \
    "${SWEBENCH_DATA_ROOT}/provenance.json" \
    "${SWEBENCH_WORK_ROOT}/environment.freeze.txt" \
    "${SWEBENCH_WORK_ROOT}/environment.normalized.freeze.txt"; do
    if [[ ! -e "${path}" ]]; then
      printf 'missing bootstrap artifact %s; run %s/bootstrap.sh first\n' "${path}" "${SWEBENCH_SCRIPT_DIR}" >&2
      return 1
    fi
  done
}
