#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# shellcheck source-path=SCRIPTDIR
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Reuse the e2e deployment and routing helpers with a separate pair of clusters.
E2E_PREFIX="${CONFORMANCE_PREFIX:-mcs-conformance}"
E2E_WORK_DIR="${CONFORMANCE_WORK_DIR:-${MODULE_ROOT}/bin/conformance}"
E2E_CLUSTER_KEYS="hub a"
E2E_PHASES=1
MCS_IMAGE_REPO="${MCS_IMAGE_REPO:-global-discovery-mcs-conformance}"
# shellcheck source=../e2e/lib.sh
source "${SCRIPT_DIR}/../e2e/lib.sh"
# shellcheck source=../e2e/setup.sh
source "${SCRIPT_DIR}/../e2e/setup.sh"

MODE="${1:-run}"
CONFORMANCE_BINARY="${CONFORMANCE_BINARY:-${MODULE_ROOT}/bin/mcs-conformance.test}"
REPORT_DIR="${E2E_WORK_DIR}/reports"
COMBINED_KUBECONFIG="${E2E_WORK_DIR}/conformance.kubeconfig"
CLUSTERS_READY=0

# Retain controller logs even when upstream assertions fail, without exporting credentials.
diagnostics() {
  local key
  mkdir -p "${REPORT_DIR}"
  for key in ${CLUSTER_KEYS}; do
    [ -s "$(kubeconfig_of "${key}")" ] || continue
    kc "${key}" --request-timeout=10s -n "${MCS_NAMESPACE}" logs \
      -l app.kubernetes.io/component=controller --all-containers=true --tail=2000 \
      >"${REPORT_DIR}/controller-${key}.log" 2>&1 || true
    kc "${key}" --request-timeout=10s get serviceexports,serviceimports,endpointslices -A -o yaml \
      >"${REPORT_DIR}/objects-${key}.yaml" 2>&1 || true
  done
  if [ -s "$(kubeconfig_of hub)" ]; then
    kc hub --request-timeout=10s -n "${MCS_NAMESPACE}" logs \
      -l app.kubernetes.io/component=registry --tail=2000 \
      >"${REPORT_DIR}/registry.log" 2>&1 || true
  fi
}

finish() {
  local rc=$?
  trap - EXIT
  if [ "${CLUSTERS_READY}" = 1 ]; then
    diagnostics
    if [ "${MODE}" = run ] && [ "${CONFORMANCE_KEEP:-0}" != 1 ]; then
      delete_clusters
    fi
  fi
  exit "${rc}"
}

# Use the kubeconfig cluster names as cluster IDs: upstream checks source-cluster against them.
setup() {
  build_images
  CLUSTERS_READY=1
  create_clusters
  install_chart hub \
    --set registry.enabled=true --set registry.store=memory --set registry.replicas=1 \
    --set registry.service.type=NodePort --set controller.enabled=true \
    --set-string "controller.clusterId=kind-$(cluster_name hub)"
  registry_url || die "cannot determine the registry NodePort"
  install_chart a --set controller.enabled=true \
    --set-string "controller.registryUrls[0]=${REGISTRY_URL}" \
    --set-string "controller.clusterId=kind-$(cluster_name a)"
  KUBECONFIG="$(kubeconfig_of hub):$(kubeconfig_of a)" \
    kubectl config view --flatten --raw >"${COMBINED_KUBECONFIG}"
  chmod 600 "${COMBINED_KUBECONFIG}"
}

run_tests() {
  [ -x "${CONFORMANCE_BINARY}" ] || die "missing ${CONFORMANCE_BINARY}; run make conformance-bin"
  [ -s "${COMBINED_KUBECONFIG}" ] || die "missing kubeconfig; run make conformance-setup first"
  CLUSTERS_READY=1
  mkdir -p "${REPORT_DIR}"
  # Never leave a prior successful report beside a later timeout or early failure.
  rm -f "${REPORT_DIR}/report.yaml" "${REPORT_DIR}/report.html" \
    "${REPORT_DIR}/results.json" "${REPORT_DIR}/junit.xml" "${REPORT_DIR}/environment.json"
  local revision image observed key
  revision="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
  image="$(kc hub --request-timeout=10s -n "${MCS_NAMESPACE}" get deployment/mcs-controller \
    -o jsonpath='{.spec.template.spec.containers[0].image}')"
  for key in ${CLUSTER_KEYS}; do
    observed="$(kc "${key}" --request-timeout=10s -n "${MCS_NAMESPACE}" get deployment/mcs-controller \
      -o jsonpath='{.spec.template.spec.containers[0].image}')"
    [ "${observed}" = "${image}" ] || die "controllers use different images; redeploy before testing"
  done
  observed="$(kc hub --request-timeout=10s -n "${MCS_NAMESPACE}" get deployment/mcs-registry \
    -o jsonpath='{.spec.template.spec.containers[0].image}')"
  [ -n "${image}" ] && [ "${observed}" = "${image}" ] || die "registry/controller image mismatch"
  jq -n --arg image "${image}" --arg revision "${revision}" \
    --arg upstream 32672de8605f9719b9220992c75cb263ba6f3746 \
    '{implementation_image: $image, harness_checkout_revision: $revision, upstream_commit: $upstream}' \
    >"${REPORT_DIR}/environment.json"
  (
    cd "${REPORT_DIR}"
    "${CONFORMANCE_BINARY}" -test.v -test.timeout="${CONFORMANCE_TIMEOUT:-60m}" \
      --kubeconfig="${COMBINED_KUBECONFIG}" \
      --contexts="kind-$(cluster_name hub),kind-$(cluster_name a)" \
      --organization=NVIDIA --project=Dynamo-MCS --version="${image}" \
      --url=https://github.com/ai-dynamo/dynamo/tree/estroz/mcs-port-module/deploy/global-discovery-mcs \
      --ginkgo.no-color --ginkgo.json-report=results.json --ginkgo.junit-report=junit.xml
  ) 2>&1 | tee "${REPORT_DIR}/suite.log"
}

case "${MODE}" in
  run | setup | test | down) ;;
  *) die "usage: $0 [run|setup|test|down]" ;;
esac
preflight
if [ "${MODE}" = down ]; then
  delete_clusters
  exit 0
fi
trap finish EXIT
case "${MODE}" in
  run) setup; run_tests ;;
  setup) setup ;;
  test) run_tests ;;
esac
