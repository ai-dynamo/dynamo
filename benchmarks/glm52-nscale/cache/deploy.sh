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

capacity_gib() {
  case "$1" in
    *Gi) printf '%s\n' "${1%Gi}" ;;
    *Ti) printf '%s\n' "$(( ${1%Ti} * 1024 ))" ;;
    *) return 1 ;;
  esac
}

if ! kubectl get pvc dockerhub-pull-cache-data -n "${NAMESPACE}" \
    >/dev/null 2>&1; then
  kubectl apply --server-side -n "${NAMESPACE}" -f "${SCRIPT_DIR}/data-pvc.yaml"
fi
kubectl get pvc dockerhub-pull-cache-data -n "${NAMESPACE}" -o json \
  | jq -e '.spec.storageClassName == "cinder"
    and .spec.accessModes == ["ReadWriteOnce"]' >/dev/null
requested="$(kubectl get pvc dockerhub-pull-cache-data -n "${NAMESPACE}" \
  -o jsonpath='{.spec.resources.requests.storage}')"
(( $(capacity_gib "${requested}") >= 500 ))
if kubectl get deployment dockerhub-pull-cache -n "${NAMESPACE}" \
    -o json 2>/dev/null \
    | jq -e '.spec.template.spec.volumes[]?
      | select(.persistentVolumeClaim.claimName == "glm52-benchmark-artifacts")' \
      >/dev/null; then
  echo "Cache still uses the artifact PVC; migrate it before deploy.sh." >&2
  exit 1
fi
marker_sha256="$(kubectl get configmap dockerhub-pull-cache-marker \
  -n "${NAMESPACE}" -o jsonpath='{.data.marker\.sha256}' 2>/dev/null || true)"
pvc_marker_sha256="$(kubectl get pvc dockerhub-pull-cache-data \
  -n "${NAMESPACE}" \
  -o jsonpath='{.metadata.annotations.benchmarks\.nvidia\.com/migration-marker-sha256}' \
  2>/dev/null || true)"
if [[ -z "${marker_sha256}" && "${pvc_marker_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
  kubectl create configmap dockerhub-pull-cache-marker -n "${NAMESPACE}" \
    --from-literal=marker.sha256="${pvc_marker_sha256}" \
    --dry-run=client -o yaml \
    | kubectl apply --server-side -f - >/dev/null
  marker_sha256="${pvc_marker_sha256}"
fi
if [[ ! "${marker_sha256}" =~ ^[0-9a-f]{64}$ \
  || "${pvc_marker_sha256}" != "${marker_sha256}" ]]; then
  echo "Dedicated cache PVC is not initialized; run migrate.sh or migrate.sh --initialize-empty." >&2
  exit 1
fi
volume_name="$(kubectl get pvc dockerhub-pull-cache-data -n "${NAMESPACE}" \
  -o jsonpath='{.spec.volumeName}')"
[[ -n "${volume_name}" ]]
[[ "$(kubectl get pv "${volume_name}" \
  -o jsonpath='{.spec.persistentVolumeReclaimPolicy}')" == Retain ]]

kubectl apply --server-side -n "${NAMESPACE}" -f "${SCRIPT_DIR}/registry-mirror.yaml"
kubectl rollout status deployment/dockerhub-pull-cache \
  -n "${NAMESPACE}" --timeout=600s

if kubectl get pod glm52-eval-runner -n "${NAMESPACE}" >/dev/null 2>&1; then
  kubectl exec glm52-eval-runner -n "${NAMESPACE}" -c runner -- \
    curl -fsS http://dockerhub-pull-cache:5000/v2/ >/dev/null
fi

echo "Docker Hub pull-through cache ready in ${NAMESPACE}"
