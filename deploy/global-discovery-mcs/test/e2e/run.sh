#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end test of mcs-registry and mcs-controller (and, in phase 3, the
# Global Discovery Service) on three local kind clusters. See README.md for
# the prerequisites, the settings and the assertions of each phase. Run it
# with make -C deploy/global-discovery-mcs e2e.
#
#   E2E_PHASES="1 2 3 4" phases to run
#   E2E_KEEP=1           keep the clusters when the run fails (always: also after a pass)
#   E2E_SKIP_BUILD=1     reuse the images of an earlier run
#   run.sh down          delete the clusters

# shellcheck source-path=SCRIPTDIR
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"
# shellcheck source=setup.sh
source "${SCRIPT_DIR}/setup.sh"
# shellcheck source=checks.sh
source "${SCRIPT_DIR}/checks.sh"
# shellcheck source=phase1.sh
source "${SCRIPT_DIR}/phase1.sh"
# shellcheck source=phase2.sh
source "${SCRIPT_DIR}/phase2.sh"
# shellcheck source=phase3.sh
source "${SCRIPT_DIR}/phase3.sh"
# shellcheck source=phase4.sh
source "${SCRIPT_DIR}/phase4.sh"

START_SECONDS="${SECONDS}"
CLUSTERS_READY=0

dump_diagnostics() {
  local key
  for key in ${CLUSTER_KEYS}; do
    [ -s "$(kubeconfig_of "${key}")" ] || continue
    section "Pods in ${MCS_NAMESPACE} (${key})"
    kc "${key}" --request-timeout=10s -n "${MCS_NAMESPACE}" get pods -o wide 2>&1 || true
    section "Logs of the controller in ${key} (tail 60)"
    kc "${key}" --request-timeout=10s -n "${MCS_NAMESPACE}" logs -l app.kubernetes.io/component=controller --tail=60 2>&1 || true
    section "MCS objects in ${key}"
    kc "${key}" --request-timeout=10s get serviceexports,serviceimports,endpointslices -A \
      -l 'dynamo.nvidia.com/mcs-managed-by' -o wide 2>&1 || true
  done
  if [ -s "$(kubeconfig_of hub)" ]; then
    section "Registry log (tail 40)"
    kc hub --request-timeout=10s -n "${MCS_NAMESPACE}" logs -l app.kubernetes.io/component=registry --tail=40 2>&1 || true
    if kc hub --request-timeout=10s get namespace global-discovery >/dev/null 2>&1; then
      section "Global Discovery Service log (tail 40)"
      kc hub --request-timeout=10s -n global-discovery logs deploy/global-discovery-service --tail=40 2>&1 || true
    fi
  fi
}

# finish: runs on exit. Prints the summary, then deletes the clusters unless
# they are to be kept.
finish() {
  local rc=$?
  trap - EXIT
  if [ "${CLUSTERS_READY}" = "1" ]; then
    if [ "${FAIL_COUNT}" -gt 0 ] || [ "${rc}" -ne 0 ]; then
      dump_diagnostics
    fi
  fi
  print_summary "${rc}"
  printf 'Wall clock: %ds\n' "$((SECONDS - START_SECONDS))"
  if [ "${CLUSTERS_READY}" = "1" ]; then
    if [ "${E2E_KEEP}" = "always" ] || { [ "${E2E_KEEP}" = "1" ] && { [ "${FAIL_COUNT}" -gt 0 ] || [ "${rc}" -ne 0 ]; }; }; then
      log "keeping the kind clusters ${E2E_PREFIX}-{hub,a,b}; kubeconfigs in ${KUBECONFIG_DIR}; make e2e-down deletes them"
    else
      delete_clusters
    fi
  fi
  if [ "${rc}" -eq 0 ] && [ "${FAIL_COUNT}" -gt 0 ]; then
    rc=1
  fi
  exit "${rc}"
}

main() {
  if [ "${1:-}" = "down" ]; then
    preflight
    delete_clusters
    exit 0
  fi
  local phase
  for phase in ${E2E_PHASES//,/ }; do
    case "${phase}" in
      1 | 2 | 3 | 4) ;;
      *) die "unknown phase '${phase}'; E2E_PHASES takes 1, 2, 3 and 4" ;;
    esac
  done
  preflight
  trap finish EXIT
  section "Setup"
  build_images
  CLUSTERS_READY=1
  create_clusters
  install_releases
  for phase in ${E2E_PHASES//,/ }; do
    "run_phase_${phase}"
    end_phase "${phase}"
  done
}

main "$@"
