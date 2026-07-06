#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -gt 1 || ($# -eq 1 && "$1" != --skip-runner-mirror) ]]; then
  echo "usage: $0 [--skip-runner-mirror]" >&2
  exit 2
fi
skip_runner_mirror=0
if [[ ${1:-} == --skip-runner-mirror ]]; then
  skip_runner_mirror=1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/campaign.env"
glm52_require_cluster_env KUBE_CONTEXT NAMESPACE
kubectl() { command kubectl --context "${KUBE_CONTEXT}" "$@"; }

runner=glm52-eval-runner
mirror=http://dockerhub-pull-cache:5000/

capacity_gib() {
  case "$1" in
    *Gi) printf '%s\n' "${1%Gi}" ;;
    *Ti) printf '%s\n' "$(( ${1%Ti} * 1024 ))" ;;
    *) return 1 ;;
  esac
}

kubectl get pvc dockerhub-pull-cache-data -n "${NAMESPACE}" -o json \
  | jq -e '.status.phase == "Bound"
    and .spec.storageClassName == "cinder"
    and .spec.accessModes == ["ReadWriteOnce"]' >/dev/null
requested="$(kubectl get pvc dockerhub-pull-cache-data -n "${NAMESPACE}" \
  -o jsonpath='{.spec.resources.requests.storage}')"
capacity="$(kubectl get pvc dockerhub-pull-cache-data -n "${NAMESPACE}" \
  -o jsonpath='{.status.capacity.storage}')"
(( $(capacity_gib "${requested}") >= 500 ))
(( $(capacity_gib "${capacity}") >= 500 ))
volume_name="$(kubectl get pvc dockerhub-pull-cache-data -n "${NAMESPACE}" \
  -o jsonpath='{.spec.volumeName}')"
[[ -n "${volume_name}" ]]
[[ "$(kubectl get pv "${volume_name}" \
  -o jsonpath='{.spec.persistentVolumeReclaimPolicy}')" == Retain ]]
kubectl get deployment dockerhub-pull-cache -n "${NAMESPACE}" -o json \
  | jq -e '.status.observedGeneration == .metadata.generation
    and .status.readyReplicas == 1
    and .status.availableReplicas == 1
    and ([.spec.template.spec.volumes[]?
      | select(.persistentVolumeClaim.claimName == "dockerhub-pull-cache-data")]
      | length) == 1
    and ([.spec.template.spec.volumes[]?
      | select(.persistentVolumeClaim.claimName == "glm52-benchmark-artifacts")]
      | length) == 0
    and ([.spec.template.spec.containers[]
      | select(.name == "registry")
      | .volumeMounts[]?
      | select(.name == "cache-data" and .mountPath == "/artifacts/cache")]
      | length) == 1' >/dev/null
marker_sha256="$(kubectl get configmap dockerhub-pull-cache-marker \
  -n "${NAMESPACE}" -o jsonpath='{.data.marker\.sha256}')"
pvc_marker_sha256="$(kubectl get pvc dockerhub-pull-cache-data \
  -n "${NAMESPACE}" \
  -o jsonpath='{.metadata.annotations.benchmarks\.nvidia\.com/migration-marker-sha256}')"
actual_marker_sha256="$(kubectl exec deployment/dockerhub-pull-cache \
  -n "${NAMESPACE}" -c registry -- \
  sha256sum /artifacts/cache/.glm52-migration-v1.json | cut -d' ' -f1)"
[[ "${marker_sha256}" =~ ^[0-9a-f]{64}$ ]]
[[ "${pvc_marker_sha256}" == "${marker_sha256}" ]]
[[ "${actual_marker_sha256}" == "${marker_sha256}" ]]
kubectl exec deployment/dockerhub-pull-cache -n "${NAMESPACE}" -c registry -- \
  cat /artifacts/cache/.glm52-migration-v1.json \
  | jq -e '
      .schema_version == 1
      and (.captured_at | type == "string")
      and (.source | IN("/source/cache/dockerhub-registry", "empty-initialization"))
      and .destination == "/destination/dockerhub-registry"
      and (.tree_sha256 | type == "string" and test("^[0-9a-f]{64}$"))
      and (.file_count | type == "number" and . >= 0 and floor == .)
      and (.bytes | type == "number" and . >= 0 and floor == .)
      and (if .source == "empty-initialization"
        then .file_count == 0 and .bytes == 0
        else .file_count > 0 and .bytes > 0
      end)
    ' >/dev/null
kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- \
  curl -fsS "${mirror}v2/" >/dev/null
if ((skip_runner_mirror == 0)); then
  mirrors="$(kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- \
    docker info --format '{{json .RegistryConfig.Mirrors}}')"
  jq -e --arg mirror "${mirror}" 'index($mirror) != null' \
    <<<"${mirrors}" >/dev/null
fi
cache_bytes="$(kubectl exec deployment/dockerhub-pull-cache \
  -n "${NAMESPACE}" -c registry -- \
  du -sb /artifacts/cache/dockerhub-registry | cut -f1)"
echo "Docker Hub pull-through cache PASS: mirror=${mirror} bytes=${cache_bytes}"
