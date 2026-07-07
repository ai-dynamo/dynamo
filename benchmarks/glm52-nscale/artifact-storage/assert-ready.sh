#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -gt 1 || ($# -eq 1 && "$1" != --storage-only) ]]; then
  echo "usage: $0 [--storage-only]" >&2
  exit 2
fi
storage_only=0
if [[ ${1:-} == --storage-only ]]; then
  storage_only=1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/campaign.env"
glm52_require_cluster_env KUBE_CONTEXT NAMESPACE
kubectl() { command kubectl --context "${KUBE_CONTEXT}" "$@"; }

pvc=glm52-benchmark-output
marker_config=glm52-benchmark-output-marker
marker_name=.glm52-artifact-migration-v1.json
runner=glm52-eval-runner

kubectl get pvc "${pvc}" -n "${NAMESPACE}" -o json | jq -e '
  .status.phase == "Bound"
  and .spec.storageClassName == "cinder"
  and .spec.accessModes == ["ReadWriteOnce"]
  and .spec.resources.requests.storage == "1Ti"
  and .metadata.annotations["benchmarks.nvidia.com/retention-policy"] == "retain"
' >/dev/null
volume_name="$(kubectl get pvc "${pvc}" -n "${NAMESPACE}" -o jsonpath='{.spec.volumeName}')"
[[ -n "${volume_name}" ]]
[[ "$(kubectl get pv "${volume_name}" -o jsonpath='{.spec.persistentVolumeReclaimPolicy}')" == Retain ]]

marker_sha256="$(kubectl get configmap "${marker_config}" -n "${NAMESPACE}" \
  -o jsonpath='{.data.marker\.sha256}')"
pvc_marker_sha256="$(kubectl get pvc "${pvc}" -n "${NAMESPACE}" \
  -o jsonpath='{.metadata.annotations.benchmarks\.nvidia\.com/migration-marker-sha256}')"
[[ "${marker_sha256}" =~ ^[0-9a-f]{64}$ ]]
[[ "${pvc_marker_sha256}" == "${marker_sha256}" ]]

if ((storage_only == 1)); then
  echo "Benchmark output storage PASS: pvc=${pvc} marker_sha256=${marker_sha256}"
  exit 0
fi

kubectl get pod "${runner}" -n "${NAMESPACE}" -o json | jq -e '
  ([.spec.volumes[]
    | select(.name == "campaign-output"
      and .persistentVolumeClaim.claimName == "glm52-benchmark-output")]
    | length) == 1
  and all(.spec.containers[] | select(.name == "runner" or .name == "dind");
    any(.volumeMounts[];
      .name == "campaign-output"
      and .mountPath == "/artifacts/glm52-nscale"
      and .subPath == "glm52-nscale"))
' >/dev/null

for container in runner dind; do
  parent_device="$(kubectl exec "${runner}" -n "${NAMESPACE}" -c "${container}" -- \
    stat -c %d /artifacts)"
  output_device="$(kubectl exec "${runner}" -n "${NAMESPACE}" -c "${container}" -- \
    stat -c %d /artifacts/glm52-nscale)"
  if [[ "${parent_device}" == "${output_device}" ]]; then
    echo "Nested output mount is absent in ${container}" >&2
    exit 1
  fi
  mounted_marker_sha256="$(kubectl exec "${runner}" -n "${NAMESPACE}" -c "${container}" -- \
    sha256sum "/artifacts/glm52-nscale/${marker_name}" | awk '{print $1}')"
  [[ "${mounted_marker_sha256}" == "${marker_sha256}" ]]
done

echo "Benchmark output mount PASS: pvc=${pvc} marker_sha256=${marker_sha256}"
