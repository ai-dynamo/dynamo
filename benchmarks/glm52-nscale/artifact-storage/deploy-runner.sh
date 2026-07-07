#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/campaign.env"
glm52_require_cluster_env KUBE_CONTEXT NAMESPACE
kubectl() { command kubectl --context "${KUBE_CONTEXT}" "$@"; }

pod=glm52-eval-runner
ready_timeout_seconds="${EVAL_RUNNER_READY_TIMEOUT_SECONDS:-1800}"
manifest="$(mktemp "${TMPDIR:-/tmp}/glm52-output-runner.XXXXXX.yaml")"
cleanup() { rm -f "${manifest}"; }
trap cleanup EXIT

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

diagnose_runner() {
  local status=$?
  trap - ERR
  set +e
  kubectl get pod "${pod}" -n "${NAMESPACE}" -o wide >&2
  kubectl describe pod "${pod}" -n "${NAMESPACE}" >&2
  kubectl logs "${pod}" -n "${NAMESPACE}" -c runner --tail=100 >&2
  kubectl logs "${pod}" -n "${NAMESPACE}" -c dind --tail=100 >&2
  exit "${status}"
}

wait_for_runner() {
  local deadline=$((SECONDS + ready_timeout_seconds))
  local phase ready runner_exit dind_exit
  while ((SECONDS < deadline)); do
    phase="$(kubectl get pod "${pod}" -n "${NAMESPACE}" \
      -o jsonpath='{.status.phase}' 2>/dev/null || true)"
    runner_exit="$(kubectl get pod "${pod}" -n "${NAMESPACE}" \
      -o jsonpath='{.status.containerStatuses[?(@.name=="runner")].state.terminated.exitCode}' \
      2>/dev/null || true)"
    dind_exit="$(kubectl get pod "${pod}" -n "${NAMESPACE}" \
      -o jsonpath='{.status.containerStatuses[?(@.name=="dind")].state.terminated.exitCode}' \
      2>/dev/null || true)"
    if [[ -n "${runner_exit}" || -n "${dind_exit}" \
      || "${phase}" == Failed || "${phase}" == Succeeded ]]; then
      echo "Runner terminated before readiness (phase=${phase:-unknown}, runner=${runner_exit:-running}, dind=${dind_exit:-running})" >&2
      return 1
    fi
    ready="$(kubectl get pod "${pod}" -n "${NAMESPACE}" \
      -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' \
      2>/dev/null || true)"
    if [[ "${ready}" == True ]]; then
      kubectl exec "${pod}" -n "${NAMESPACE}" -c runner -- test -f /run/glm52-runner-ready
      kubectl exec "${pod}" -n "${NAMESPACE}" -c runner -- docker info >/dev/null
      kubectl exec "${pod}" -n "${NAMESPACE}" -c runner -- docker compose version >/dev/null
      return 0
    fi
    sleep 5
  done
  echo "Timed out after ${ready_timeout_seconds}s waiting for ${pod}" >&2
  return 1
}

"${ROOT_DIR}/eval/assert-pinned-source.sh" >/dev/null
"${SCRIPT_DIR}/assert-ready.sh" --storage-only >/dev/null
"${ROOT_DIR}/deploy/assert-clean-slate.sh" >/dev/null
command kubectl kustomize "${ROOT_DIR}" >"${manifest}"
manifest_sha256="$(sha256_file "${manifest}")"

current_sha256="$(kubectl get pod "${pod}" -n "${NAMESPACE}" \
  -o jsonpath='{.metadata.annotations.benchmarks\.nvidia\.com/manifest-sha256}' \
  2>/dev/null || true)"
if kubectl get pod "${pod}" -n "${NAMESPACE}" >/dev/null 2>&1; then
  current_phase="$(kubectl get pod "${pod}" -n "${NAMESPACE}" -o jsonpath='{.status.phase}')"
  if [[ "${current_phase}" == Failed || "${current_phase}" == Succeeded ]]; then
    kubectl delete pod "${pod}" -n "${NAMESPACE}" --wait=true
  elif [[ "${current_sha256}" != "${manifest_sha256}" ]]; then
    echo "Runner manifest changed; refusing implicit pod deletion." >&2
    echo "Export runner-local state, then run eval/teardown-runner.sh explicitly." >&2
    exit 1
  else
    "${ROOT_DIR}/eval/assert-runner-idle.sh"
  fi
fi

kubectl apply --server-side -n "${NAMESPACE}" -f "${manifest}"
kubectl wait --for=jsonpath='{.status.phase}'=Bound \
  pvc/glm52-benchmark-artifacts pvc/glm52-benchmark-output \
  -n "${NAMESPACE}" --timeout=600s
kubectl annotate pod "${pod}" -n "${NAMESPACE}" \
  benchmarks.nvidia.com/manifest-sha256="${manifest_sha256}" --overwrite
trap diagnose_runner ERR
wait_for_runner
trap - ERR

"${ROOT_DIR}/eval/sync-runner.sh"
"${ROOT_DIR}/cache/configure-runner.sh"
"${ROOT_DIR}/cache/assert-ready.sh"
"${SCRIPT_DIR}/assert-ready.sh"
"${ROOT_DIR}/eval/capture-runner-identity.sh"
echo "Evaluation runner ready with dedicated benchmark output: manifest_sha256=${manifest_sha256}"
