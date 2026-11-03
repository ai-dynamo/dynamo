#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Exercise the runner's EXIT handler without creating or deleting real clusters.

# shellcheck source-path=SCRIPTDIR
set -euo pipefail

RUNNER_TEST_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${RUNNER_TEST_SCRIPT_DIR}/lib.sh"
RUNNER_TEST_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mcs-runner-test.XXXXXX")"
trap 'rm -rf "${RUNNER_TEST_DIR}"' EXIT

runner_finishes() {
  local mode="$1" incoming="$2" keep="$3" expected_status="$4" expected_deleted="$5" failed="${6:-0}"
  local work="${RUNNER_TEST_DIR}/${mode}-${incoming}-${keep}-${failed}" status=0 deleted=0
  mkdir -p "${work}" || return 1
  case "${mode}" in
    directory) : >"${work}/reports" ;;
    summary) mkdir -p "${work}/reports/summary.txt" || return 1 ;;
  esac

  # Stub external operations; reports and the EXIT trap use the real runner.
  # shellcheck disable=SC2016  # Variables expand in the child bash process.
  E2E_WORK_DIR="${work}" E2E_KEEP="${keep}" RUNNER_INCOMING_STATUS="${incoming}" \
    RUNNER_FAILURE_MODE="${mode}" RUNNER_FAIL_COUNT="${failed}" \
    RUNNER_UNDER_TEST="${RUNNER_TEST_SCRIPT_DIR}/run.sh" \
    "${BASH}" -c '
      source "${RUNNER_UNDER_TEST}"
      CLUSTERS_READY=1
      CLUSTER_KEYS="hub a"
      FAIL_COUNT="${RUNNER_FAIL_COUNT}"
      kc() { printf "{}\n"; }
      delete_clusters() { : >"${E2E_WORK_DIR}/deleted"; }
      dump_diagnostics() { return 0; }
      if [ "${RUNNER_FAILURE_MODE}" = write ]; then
        RUNNER_SUMMARY_CALLS=0
        print_summary() {
          RUNNER_SUMMARY_CALLS=$((RUNNER_SUMMARY_CALLS + 1))
          if [ "${RUNNER_SUMMARY_CALLS}" -eq 1 ]; then return 18; fi
        }
      fi
      trap finish EXIT
      exit "${RUNNER_INCOMING_STATUS}"
    ' >"${work}/output.log" 2>&1 || status=$?
  [ ! -f "${work}/deleted" ] || deleted=1
  LAST_OBSERVED="exit ${status} (expected ${expected_status}); deletion ${deleted} (expected ${expected_deleted})"
  if [ "${status}" != "${expected_status}" ] || [ "${deleted}" != "${expected_deleted}" ]; then
    LAST_OBSERVED="${LAST_OBSERVED}; runner log: $(tail -n 8 "${work}/output.log")"
    return 1
  fi
}

once "successful run deletes clusters and exits zero" runner_finishes normal 0 0 0 1 || true
once "failed assertions delete clusters and fail the run" runner_finishes normal 0 0 1 1 1 || true
once "setup failure preserves its exit status and deletes clusters" runner_finishes normal 7 0 7 1 || true
once "report directory failure still deletes clusters and fails the run" runner_finishes directory 0 0 1 1 || true
once "summary redirection failure still deletes clusters and fails the run" runner_finishes summary 0 0 1 1 || true
once "summary write failure still deletes clusters and fails the run" runner_finishes write 0 0 1 1 || true
once "report failure preserves an existing nonzero exit status" runner_finishes directory 7 0 7 1 || true
once "E2E_KEEP=always preserves clusters after success" runner_finishes normal 0 always 0 0 || true
once "E2E_KEEP=1 preserves clusters after failure" runner_finishes normal 7 1 7 0 || true
print_summary
[ "${FAIL_COUNT}" -eq 0 ]
