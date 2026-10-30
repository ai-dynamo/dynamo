#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end test of mcs-registry and mcs-controller (and, in phase 3, the
# Global Discovery Service) on three local kind clusters. See README.md for
# the prerequisites, the settings and the assertions of each phase. Run it
# with make -C deploy/global-discovery-mcs e2e.
#
#   E2E_PHASES="1 2 3"   phases to run
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
# shellcheck source=provider.sh
source "${SCRIPT_DIR}/provider.sh"
# shellcheck source=checks.sh
source "${SCRIPT_DIR}/checks.sh"
# shellcheck source=phase1.sh
source "${SCRIPT_DIR}/phase1.sh"
# shellcheck source=phase2.sh
source "${SCRIPT_DIR}/phase2.sh"
# shellcheck source=phase3.sh
source "${SCRIPT_DIR}/phase3.sh"

START_SECONDS="${SECONDS}"
CLUSTERS_READY=0

dump_diagnostics() {
  local key
  for key in ${CLUSTER_KEYS}; do
    [ -s "$(kubeconfig_of "${key}")" ] || continue
    section "Pods in ${key}"
    kc "${key}" --request-timeout=10s get pods -A -o wide 2>&1 || true
    if [ "${E2E_PROVIDER}" = cilium ]; then
      local workload
      for workload in daemonset/cilium deployment/cilium-operator deployment/clustermesh-apiserver; do
        kc "${key}" --request-timeout=10s -n kube-system logs "${workload}" --all-containers --tail=60 2>&1 || true
      done
    fi
    section "Logs of the controller in ${key} (tail 60)"
    kc "${key}" --request-timeout=10s -n "${MCS_NAMESPACE}" logs -l app.kubernetes.io/component=controller --tail=60 2>&1 || true
    section "MCS objects in ${key}"
    kc "${key}" --request-timeout=10s get serviceexports,serviceimports,endpointslices -A -o wide 2>&1 || true
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

  # Report failures must not interrupt teardown.
  if ! mkdir -p "${E2E_WORK_DIR}/reports" || ! print_summary "${rc}" >"${E2E_WORK_DIR}/reports/summary.txt"; then
    printf 'e2e: could not save the summary; continuing cleanup\n' >&2
    if [ "${rc}" -eq 0 ]; then rc=1; fi
  fi
  jq -n --arg provider "${E2E_PROVIDER}" --arg ciliumVersion "${CILIUM_VERSION}" \
    --arg gdsImage "${GDS_IMAGE_REPO}:${GDS_IMAGE_TAG}" --arg mcsImage "${MCS_IMAGE_REPO}:${MCS_IMAGE_TAG}" \
    --arg revision "$(git -C "${REPO_ROOT}" rev-parse HEAD)" \
    --argjson dirty "$(if [ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]; then printf true; else printf false; fi)" \
    '{provider: $provider, cilium_version: $ciliumVersion, gds_image: $gdsImage,
      bundled_mcs_image: (if $provider == "dynamo" then $mcsImage else null end),
      source_revision: $revision, source_dirty: $dirty}' \
    >"${E2E_WORK_DIR}/reports/environment.json" || true
  if [ "${CLUSTERS_READY}" = "1" ]; then
    local key
    for key in ${CLUSTER_KEYS}; do
      kc "${key}" --request-timeout=10s get serviceexports,serviceimports,endpointslices,pods -A -o json \
        >"${E2E_WORK_DIR}/reports/objects-${key}.json" 2>"${E2E_WORK_DIR}/reports/objects-${key}.err" || true
    done
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
    validate_reused_clusters
    delete_clusters
    exit 0
  fi
  local phase
  if [ "${E2E_PROVIDER}" = cilium ] && [ "${CLUSTER_KEYS%% *}" != hub ]; then
    die "E2E_CLUSTER_KEYS must start with hub for Cilium's shared test CA"
  fi
  for phase in ${E2E_PHASES//,/ }; do
    case "${phase}" in
      1 | 2 | 3) ;;
      *) die "unknown phase '${phase}'; E2E_PHASES takes 1, 2 and 3" ;;
    esac
    if [ "${E2E_PROVIDER}" = cilium ] && [ "${phase}" != 3 ]; then
      die "Cilium supports phase 3; phases 1 and 2 test the bundled registry and controller"
    fi
    if [ "${phase}" = 3 ] && { ! has_cluster hub || ! has_cluster a; }; then
      die "phase 3 requires both hub and a; discovery assertions must not be skipped"
    fi
  done
  preflight
  rm -f "${E2E_WORK_DIR}/reports/summary.txt" "${E2E_WORK_DIR}/reports/environment.json" \
    "${E2E_WORK_DIR}/reports/objects-"*.json "${E2E_WORK_DIR}/reports/objects-"*.err
  validate_reused_clusters
  trap finish EXIT
  section "Setup"
  build_images
  CLUSTERS_READY=1
  create_clusters
  install_provider
  end_phase setup
  for phase in ${E2E_PHASES//,/ }; do
    "run_phase_${phase}"
    end_phase "${phase}"
  done
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main "$@"
fi
