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
POD=glm52-eval-runner
PVC=glm52-benchmark-artifacts
READY_TIMEOUT_SECONDS="${EVAL_RUNNER_READY_TIMEOUT_SECONDS:-1800}"
"${SCRIPT_DIR}/assert-pinned-source.sh" >/dev/null

for command in kubectl awk; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "Required command not found: ${command}" >&2
    exit 1
  fi
done

if ! kubectl get secret hf-token-secret -n "${NAMESPACE}" >/dev/null 2>&1; then
  echo "Required dataset-access secret ${NAMESPACE}/hf-token-secret is missing" >&2
  exit 1
fi

diagnose_runner() {
  local exit_code=$?
  trap - ERR
  set +e
  echo "Evaluation runner did not become ready" >&2
  kubectl get pod "${POD}" -n "${NAMESPACE}" -o wide >&2
  kubectl describe pod "${POD}" -n "${NAMESPACE}" >&2
  echo "--- runner log (last 100 lines) ---" >&2
  kubectl logs "${POD}" -n "${NAMESPACE}" -c runner --tail=100 >&2
  echo "--- dind log (last 100 lines) ---" >&2
  kubectl logs "${POD}" -n "${NAMESPACE}" -c dind --tail=100 >&2
  exit "${exit_code}"
}

wait_for_runner() {
  local deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
  local next_status=${SECONDS}
  local phase ready runner_exit dind_exit

  while ((SECONDS < deadline)); do
    phase="$(kubectl get pod "${POD}" -n "${NAMESPACE}" \
      -o jsonpath='{.status.phase}' 2>/dev/null || true)"
    runner_exit="$(kubectl get pod "${POD}" -n "${NAMESPACE}" \
      -o jsonpath='{.status.containerStatuses[?(@.name=="runner")].state.terminated.exitCode}' \
      2>/dev/null || true)"
    dind_exit="$(kubectl get pod "${POD}" -n "${NAMESPACE}" \
      -o jsonpath='{.status.containerStatuses[?(@.name=="dind")].state.terminated.exitCode}' \
      2>/dev/null || true)"

    if [[ -n "${runner_exit}" || -n "${dind_exit}" \
      || "${phase}" == Failed || "${phase}" == Succeeded ]]; then
      echo "Runner terminated before readiness (phase=${phase:-unknown}, runner=${runner_exit:-running}, dind=${dind_exit:-running})" >&2
      return 1
    fi

    ready="$(kubectl get pod "${POD}" -n "${NAMESPACE}" \
      -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' \
      2>/dev/null || true)"
    if [[ "${ready}" == True ]]; then
      kubectl exec -n "${NAMESPACE}" "${POD}" -c runner -- \
        test -f /run/glm52-runner-ready
      kubectl exec -n "${NAMESPACE}" "${POD}" -c runner -- \
        docker info >/dev/null
      kubectl exec -n "${NAMESPACE}" "${POD}" -c runner -- \
        docker compose version >/dev/null
      return 0
    fi

    if ((SECONDS >= next_status)); then
      echo "Waiting for evaluation runner (phase=${phase:-unknown}, ready=${ready:-unknown})"
      next_status=$((SECONDS + 30))
    fi
    sleep 5
  done

  echo "Timed out after ${READY_TIMEOUT_SECONDS}s waiting for ${POD}" >&2
  return 1
}

if command -v sha256sum >/dev/null 2>&1; then
  manifest_sha256="$(sha256sum "${SCRIPT_DIR}/runner.yaml" | awk '{print $1}')"
else
  manifest_sha256="$(shasum -a 256 "${SCRIPT_DIR}/runner.yaml" | awk '{print $1}')"
fi
current_sha256="$(kubectl get pod "${POD}" -n "${NAMESPACE}" \
  -o jsonpath='{.metadata.annotations.benchmarks\.nvidia\.com/manifest-sha256}' \
  2>/dev/null || true)"

if kubectl get pod "${POD}" -n "${NAMESPACE}" >/dev/null 2>&1; then
  current_phase="$(kubectl get pod "${POD}" -n "${NAMESPACE}" \
    -o jsonpath='{.status.phase}')"
  if [[ "${current_phase}" == Failed || "${current_phase}" == Succeeded ]]; then
    echo "Recreating evaluation runner for manifest ${manifest_sha256}"
    kubectl delete pod "${POD}" -n "${NAMESPACE}" --wait=true
  elif [[ "${current_sha256}" != "${manifest_sha256}" ]]; then
    echo "Runner manifest changed; refusing implicit pod deletion." >&2
    echo "Wait for active work, then run teardown-runner.sh before deploy-runner.sh." >&2
    exit 1
  else
    "${SCRIPT_DIR}/assert-runner-idle.sh"
  fi
fi

kubectl apply --server-side -n "${NAMESPACE}" -f "${SCRIPT_DIR}/runner.yaml"
kubectl wait --for=jsonpath='{.status.phase}'=Bound "pvc/${PVC}" \
  -n "${NAMESPACE}" --timeout=600s
kubectl annotate pod "${POD}" -n "${NAMESPACE}" \
  benchmarks.nvidia.com/manifest-sha256="${manifest_sha256}" --overwrite
trap diagnose_runner ERR
wait_for_runner
trap - ERR
"${SCRIPT_DIR}/sync-runner.sh"
"${SCRIPT_DIR}/capture-runner-identity.sh"
echo "Evaluation runner ready"
