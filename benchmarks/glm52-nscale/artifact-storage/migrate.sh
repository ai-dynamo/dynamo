#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "usage: $0" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/campaign.env"
glm52_require_cluster_env KUBE_CONTEXT NAMESPACE
kubectl() { command kubectl --context "${KUBE_CONTEXT}" "$@"; }

runner=glm52-eval-runner
source_pvc=glm52-benchmark-artifacts
destination_pvc=glm52-benchmark-output
migration_pod=glm52-benchmark-output-migrate-v1
migration_config=glm52-benchmark-output-migration-v1
marker_config=glm52-benchmark-output-marker
python_image=python:3.12-bookworm@sha256:c36262cd12ed3eb4c32146f5268ea5037e04c688ccf32cdb04b6084671845541
timeout_seconds="${ARTIFACT_MIGRATION_TIMEOUT_SECONDS:-1800}"
complete=0

if [[ ! "${timeout_seconds}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ARTIFACT_MIGRATION_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 2
fi

cleanup() {
  local status=$?
  trap - EXIT
  if ((status != 0 && complete == 0)); then
    kubectl logs "${migration_pod}" -n "${NAMESPACE}" 2>/dev/null || true
    echo "Migration failed closed; source VAST data and the dedicated PVC were retained." >&2
  fi
  kubectl delete pod "${migration_pod}" -n "${NAMESPACE}" \
    --ignore-not-found --wait=true >/dev/null 2>&1 || true
  exit "${status}"
}
trap cleanup EXIT

"${ROOT_DIR}/eval/assert-pinned-source.sh" >/dev/null
"${ROOT_DIR}/eval/assert-runner-idle.sh"
"${ROOT_DIR}/deploy/assert-clean-slate.sh" >/dev/null
kubectl get pod "${runner}" -n "${NAMESPACE}" >/dev/null

if ! kubectl get pvc "${destination_pvc}" -n "${NAMESPACE}" >/dev/null 2>&1; then
  kubectl apply --server-side -n "${NAMESPACE}" -f "${SCRIPT_DIR}/data-pvc.yaml"
fi
kubectl get pvc "${destination_pvc}" -n "${NAMESPACE}" -o json | jq -e '
  .spec.storageClassName == "cinder"
  and .spec.accessModes == ["ReadWriteOnce"]
  and .spec.resources.requests.storage == "1Ti"
' >/dev/null

kubectl create configmap "${migration_config}" -n "${NAMESPACE}" \
  --from-file=migrate_artifacts.py="${SCRIPT_DIR}/migrate_artifacts.py" \
  --dry-run=client -o yaml \
  | kubectl apply --server-side -f - >/dev/null

runner_node="$(kubectl get pod "${runner}" -n "${NAMESPACE}" -o jsonpath='{.spec.nodeName}')"
source_pvc_uid="$(kubectl get pvc "${source_pvc}" -n "${NAMESPACE}" -o jsonpath='{.metadata.uid}')"
[[ -n "${runner_node}" && -n "${source_pvc_uid}" ]]
kubectl delete pod "${migration_pod}" -n "${NAMESPACE}" \
  --ignore-not-found --wait=true >/dev/null

jq -n \
  --arg node "${runner_node}" \
  --arg image "${python_image}" \
  --arg source_pvc "${source_pvc}" \
  --arg destination_pvc "${destination_pvc}" \
  --arg source_pvc_uid "${source_pvc_uid}" '
  {
    apiVersion: "v1",
    kind: "Pod",
    metadata: {
      name: "glm52-benchmark-output-migrate-v1",
      labels: {
        "app.kubernetes.io/name": "glm52-benchmark-output-migration",
        "app.kubernetes.io/part-of": "glm52-nscale-benchmark"
      }
    },
    spec: {
      automountServiceAccountToken: false,
      nodeSelector: {"kubernetes.io/hostname": $node},
      restartPolicy: "Never",
      containers: [{
        name: "migrate",
        image: $image,
        imagePullPolicy: "IfNotPresent",
        command: [
          "python3", "/migration/migrate_artifacts.py",
          "--source", "/source/glm52-nscale",
          "--destination-parent", "/destination",
          "--source-pvc-uid", $source_pvc_uid,
          "--require-absent", "swebench/results/vllm-serve-ab-r2/verified",
          "--require-file-sha256",
          "swebench/results/dynamo-vllm-ab/verified/task-images.json=39c96a5fdec81852fb0342556ba8c0922f1bb54854e79ed2661b6e6981bb5a21"
        ],
        resources: {
          requests: {cpu: "1", memory: "1Gi"},
          limits: {cpu: "4", memory: "8Gi"}
        },
        volumeMounts: [
          {name: "source", mountPath: "/source", readOnly: true},
          {name: "destination", mountPath: "/destination"},
          {name: "migration", mountPath: "/migration", readOnly: true}
        ]
      }],
      volumes: [
        {name: "source", persistentVolumeClaim: {claimName: $source_pvc, readOnly: true}},
        {name: "destination", persistentVolumeClaim: {claimName: $destination_pvc}},
        {name: "migration", configMap: {name: "glm52-benchmark-output-migration-v1", defaultMode: 365}}
      ]
    }
  }' | kubectl apply -n "${NAMESPACE}" -f - >/dev/null

kubectl wait --for=jsonpath='{.status.phase}'=Bound \
  "pvc/${destination_pvc}" -n "${NAMESPACE}" --timeout=600s
volume_name="$(kubectl get pvc "${destination_pvc}" -n "${NAMESPACE}" \
  -o jsonpath='{.spec.volumeName}')"
[[ -n "${volume_name}" ]]
kubectl patch pv "${volume_name}" --type=merge \
  -p '{"spec":{"persistentVolumeReclaimPolicy":"Retain"}}' >/dev/null
[[ "$(kubectl get pv "${volume_name}" -o jsonpath='{.spec.persistentVolumeReclaimPolicy}')" == Retain ]]

deadline=$((SECONDS + timeout_seconds))
while true; do
  phase="$(kubectl get pod "${migration_pod}" -n "${NAMESPACE}" \
    -o jsonpath='{.status.phase}')"
  case "${phase}" in
    Succeeded) break ;;
    Failed)
      echo "Artifact migration pod failed" >&2
      exit 1
      ;;
  esac
  if ((SECONDS >= deadline)); then
    echo "Artifact migration did not finish within ${timeout_seconds} seconds" >&2
    exit 1
  fi
  sleep 5
done

migration_output="$(kubectl logs "${migration_pod}" -n "${NAMESPACE}")"
printf '%s\n' "${migration_output}"
jq -e --arg source_pvc_uid "${source_pvc_uid}" '
  (.state == "complete" or .state == "already_complete")
  and .marker.schema_version == 1
  and .marker.source_pvc_uid == $source_pvc_uid
  and .marker.source_identity == .marker.destination_identity
  and .marker.requirements.required_absent == ["swebench/results/vllm-serve-ab-r2/verified"]
  and .marker.requirements.required_files["swebench/results/dynamo-vllm-ab/verified/task-images.json"] == "39c96a5fdec81852fb0342556ba8c0922f1bb54854e79ed2661b6e6981bb5a21"
  and (.marker.source_identity.tree_sha256 | test("^[0-9a-f]{64}$"))
  and (.marker.source_identity.regular_files | type == "number" and . > 0)
  and (.marker.source_identity.bytes | type == "number" and . > 0)
  and (.marker_sha256 | test("^[0-9a-f]{64}$"))
' <<<"${migration_output}" >/dev/null

marker_sha256="$(jq -r '.marker_sha256' <<<"${migration_output}")"
source_tree_sha256="$(jq -r '.marker.source_identity.tree_sha256' <<<"${migration_output}")"
[[ "$(kubectl get pv "${volume_name}" -o jsonpath='{.spec.persistentVolumeReclaimPolicy}')" == Retain ]]

kubectl create configmap "${marker_config}" -n "${NAMESPACE}" \
  --from-literal=marker.sha256="${marker_sha256}" \
  --from-literal=source.pvc.uid="${source_pvc_uid}" \
  --from-literal=source.tree.sha256="${source_tree_sha256}" \
  --dry-run=client -o yaml \
  | kubectl apply --server-side -f - >/dev/null
kubectl annotate pvc "${destination_pvc}" -n "${NAMESPACE}" \
  benchmarks.nvidia.com/migration-marker-sha256="${marker_sha256}" \
  benchmarks.nvidia.com/source-pvc-uid="${source_pvc_uid}" \
  benchmarks.nvidia.com/source-tree-sha256="${source_tree_sha256}" \
  --overwrite >/dev/null

kubectl delete pod "${migration_pod}" -n "${NAMESPACE}" --wait=true >/dev/null
complete=1
"${SCRIPT_DIR}/assert-ready.sh" --storage-only
echo "Benchmark output migration PASS: tree_sha256=${source_tree_sha256} marker_sha256=${marker_sha256}"
