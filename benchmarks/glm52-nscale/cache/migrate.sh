#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -gt 1 || ($# -eq 1 && "$1" != --initialize-empty) ]]; then
  echo "usage: $0 [--initialize-empty]" >&2
  exit 2
fi
initialize_empty=0
if [[ ${1:-} == --initialize-empty ]]; then
  initialize_empty=1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/campaign.env"
glm52_require_cluster_env KUBE_CONTEXT NAMESPACE
kubectl() { command kubectl --context "${KUBE_CONTEXT}" "$@"; }

runner=glm52-eval-runner
migration_pod=dockerhub-pull-cache-migrate-v1
migration_config=dockerhub-pull-cache-migration-v1
marker_config=dockerhub-pull-cache-marker
python_image=python:3.12-bookworm@sha256:c36262cd12ed3eb4c32146f5268ea5037e04c688ccf32cdb04b6084671845541
marker_bound=0
legacy_rollback_armed=0
complete=0
migration_timeout_seconds="${CACHE_MIGRATION_TIMEOUT_SECONDS:-1800}"
if [[ ! "${migration_timeout_seconds}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CACHE_MIGRATION_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 2
fi

rollback() {
  local status=$?
  local restored=0
  trap - EXIT
  if ((status != 0 && complete == 0)); then
    kubectl logs "${migration_pod}" -n "${NAMESPACE}" 2>/dev/null || true
    if ((initialize_empty == 1)); then
      echo "Empty cache initialization has no legacy rollback; retaining dedicated resources for deploy.sh retry." >&2
    elif ((legacy_rollback_armed == 1)); then
      if kubectl apply --server-side -n "${NAMESPACE}" \
          -f "${SCRIPT_DIR}/registry-mirror-vast-rollback.yaml" \
        && kubectl rollout status deployment/dockerhub-pull-cache \
          -n "${NAMESPACE}" --timeout=600s \
        && kubectl get deployment dockerhub-pull-cache -n "${NAMESPACE}" \
          -o json | jq -e '.status.readyReplicas == 1
            and ([.spec.template.spec.volumes[]?
              | select(.persistentVolumeClaim.claimName == "glm52-benchmark-artifacts")]
              | length) == 1' >/dev/null; then
        restored=1
      fi
    else
      echo "Migration failed before legacy rollback was armed; leaving cache deployment unchanged." >&2
    fi
    if ((legacy_rollback_armed == 1 && marker_bound == 1 && restored == 1)); then
      kubectl delete configmap "${marker_config}" -n "${NAMESPACE}" \
        --ignore-not-found || true
      kubectl annotate pvc dockerhub-pull-cache-data -n "${NAMESPACE}" \
        benchmarks.nvidia.com/migration-marker-sha256- || true
    elif ((legacy_rollback_armed == 1 && marker_bound == 1)); then
      echo "Rollback did not restore the legacy cache; retaining marker binding." >&2
    fi
  fi
  exit "${status}"
}
trap rollback EXIT

"${ROOT_DIR}/eval/assert-runner-idle.sh"
if ((initialize_empty == 1)); then
  if kubectl get deployment dockerhub-pull-cache -n "${NAMESPACE}" \
      >/dev/null 2>&1; then
    echo "--initialize-empty requires no existing cache Deployment" >&2
    exit 1
  fi
else
  kubectl get deployment dockerhub-pull-cache -n "${NAMESPACE}" -o json \
    | jq -e '.status.readyReplicas == 1
      and ([.spec.template.spec.volumes[]?
        | select(.persistentVolumeClaim.claimName == "glm52-benchmark-artifacts")]
        | length) == 1' >/dev/null
fi

runner_node="$(kubectl get pod "${runner}" -n "${NAMESPACE}" \
  -o jsonpath='{.spec.nodeName}')"
runner_uid_before="$(kubectl get pod "${runner}" -n "${NAMESPACE}" \
  -o jsonpath='{.metadata.uid}')"
restart_counts_before="$(kubectl get pod "${runner}" -n "${NAMESPACE}" \
  -o json | jq -c '[.status.containerStatuses[]
    | {name, restart_count: .restartCount}] | sort_by(.name)')"
mirrors_before="$(kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- \
  docker info --format '{{json .RegistryConfig.Mirrors}}')"

if ! kubectl get pvc dockerhub-pull-cache-data -n "${NAMESPACE}" \
    >/dev/null 2>&1; then
  kubectl apply --server-side -n "${NAMESPACE}" -f "${SCRIPT_DIR}/data-pvc.yaml"
fi
kubectl create configmap "${migration_config}" -n "${NAMESPACE}" \
  --from-file=migrate_cache.py="${SCRIPT_DIR}/migrate_cache.py" \
  --dry-run=client -o yaml \
  | kubectl apply --server-side -f - >/dev/null

if ((initialize_empty == 0)); then
  legacy_rollback_armed=1
  kubectl scale deployment/dockerhub-pull-cache -n "${NAMESPACE}" --replicas=0
  kubectl wait --for=delete pod -n "${NAMESPACE}" \
    -l app.kubernetes.io/name=dockerhub-pull-cache --timeout=600s
fi

kubectl delete pod "${migration_pod}" -n "${NAMESPACE}" \
  --ignore-not-found --wait=true >/dev/null
jq -n \
  --arg node "${runner_node}" \
  --arg image "${python_image}" \
  --argjson initialize_empty "${initialize_empty}" '
  {
    apiVersion: "v1",
    kind: "Pod",
    metadata: {
      name: "dockerhub-pull-cache-migrate-v1",
      labels: {
        "app.kubernetes.io/name": "dockerhub-pull-cache-migration",
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
        command: ([
          "python3", "/migration/migrate_cache.py",
          "--destination", "/destination"
        ] + if $initialize_empty == 1
          then ["--initialize-empty"]
          else ["--source", "/source/cache/dockerhub-registry"]
        end),
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
        {name: "source", persistentVolumeClaim: {claimName: "glm52-benchmark-artifacts", readOnly: true}},
        {name: "destination", persistentVolumeClaim: {claimName: "dockerhub-pull-cache-data"}},
        {name: "migration", configMap: {name: "dockerhub-pull-cache-migration-v1", defaultMode: 365}}
      ]
    }
  }' | kubectl apply -n "${NAMESPACE}" -f - >/dev/null

deadline=$((SECONDS + migration_timeout_seconds))
while true; do
  phase="$(kubectl get pod "${migration_pod}" -n "${NAMESPACE}" \
    -o jsonpath='{.status.phase}')"
  case "${phase}" in
    Succeeded) break ;;
    Failed)
      echo "Cache migration pod failed" >&2
      exit 1
      ;;
  esac
  if ((SECONDS >= deadline)); then
    echo "Cache migration pod did not finish within ${migration_timeout_seconds} seconds" >&2
    exit 1
  fi
  sleep 5
done
migration_output="$(kubectl logs "${migration_pod}" -n "${NAMESPACE}")"
printf '%s\n' "${migration_output}"
marker_sha256="$(jq -er '
  .marker_sha256
  | select(type == "string" and test("^[0-9a-f]{64}$"))
' <<<"${migration_output}")"
volume_name="$(kubectl get pvc dockerhub-pull-cache-data -n "${NAMESPACE}" \
  -o jsonpath='{.spec.volumeName}')"
[[ -n "${volume_name}" ]]
kubectl patch pv "${volume_name}" --type=merge \
  -p '{"spec":{"persistentVolumeReclaimPolicy":"Retain"}}' >/dev/null
[[ "$(kubectl get pv "${volume_name}" \
  -o jsonpath='{.spec.persistentVolumeReclaimPolicy}')" == Retain ]]
jq -e '
  .marker.schema_version == 1
  and (.marker.captured_at | type == "string")
  and (.marker.source | IN("/source/cache/dockerhub-registry", "empty-initialization"))
  and .marker.destination == "/destination/dockerhub-registry"
  and (.marker.tree_sha256 | type == "string" and test("^[0-9a-f]{64}$"))
  and (.marker.file_count | type == "number" and . >= 0 and floor == .)
  and (.marker.bytes | type == "number" and . >= 0 and floor == .)
  and (if .marker.source == "empty-initialization"
    then .marker.file_count == 0 and .marker.bytes == 0
    else .marker.file_count > 0 and .marker.bytes > 0
  end)
' <<<"${migration_output}" >/dev/null
if ((initialize_empty == 1)); then
  jq -e '.marker.source == "empty-initialization"' \
    <<<"${migration_output}" >/dev/null
else
  jq -e '.marker.source == "/source/cache/dockerhub-registry"' \
    <<<"${migration_output}" >/dev/null
fi
kubectl create configmap "${marker_config}" -n "${NAMESPACE}" \
  --from-literal=marker.sha256="${marker_sha256}" \
  --dry-run=client -o yaml \
  | kubectl apply --server-side -f - >/dev/null
kubectl annotate pvc dockerhub-pull-cache-data -n "${NAMESPACE}" \
  benchmarks.nvidia.com/migration-marker-sha256="${marker_sha256}" \
  --overwrite >/dev/null
marker_bound=1
kubectl delete pod "${migration_pod}" -n "${NAMESPACE}" --wait=true >/dev/null

kubectl apply --server-side -n "${NAMESPACE}" -f "${SCRIPT_DIR}/registry-mirror.yaml"
kubectl rollout status deployment/dockerhub-pull-cache \
  -n "${NAMESPACE}" --timeout=600s
"${SCRIPT_DIR}/assert-ready.sh" --skip-runner-mirror

runner_uid_after="$(kubectl get pod "${runner}" -n "${NAMESPACE}" \
  -o jsonpath='{.metadata.uid}')"
restart_counts_after="$(kubectl get pod "${runner}" -n "${NAMESPACE}" \
  -o json | jq -c '[.status.containerStatuses[]
    | {name, restart_count: .restartCount}] | sort_by(.name)')"
mirrors_after="$(kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- \
  docker info --format '{{json .RegistryConfig.Mirrors}}')"
if [[ "${runner_uid_before}" != "${runner_uid_after}" \
  || "${restart_counts_before}" != "${restart_counts_after}" ]]; then
  echo "Runner identity or restart count changed during migration" >&2
  exit 1
fi
if ((initialize_empty == 1)); then
  "${SCRIPT_DIR}/configure-runner.sh"
  restart_counts_after="$(kubectl get pod "${runner}" -n "${NAMESPACE}" \
    -o json | jq -c '[.status.containerStatuses[]
      | {name, restart_count: .restartCount}] | sort_by(.name)')"
  [[ "${restart_counts_before}" == "${restart_counts_after}" ]]
elif [[ "${mirrors_before}" != "${mirrors_after}" ]]; then
  echo "Runner mirror list changed during migration" >&2
  exit 1
fi
"${SCRIPT_DIR}/assert-ready.sh"

"${SCRIPT_DIR}/verify.sh"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
remote=/artifacts/glm52-nscale/cache/migrations/${timestamp}.json
kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- \
  mkdir -p /artifacts/glm52-nscale/cache/migrations
kubectl exec deployment/dockerhub-pull-cache -n "${NAMESPACE}" -c registry -- \
  cat /artifacts/cache/.glm52-migration-v1.json \
  | kubectl exec -i "${runner}" -n "${NAMESPACE}" -c runner -- \
      bash -eu -c 'temporary="$1.tmp"; cat >"${temporary}"; mv "${temporary}" "$1"' \
      -- "${remote}"
remote_sha256="$(kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- \
  sha256sum "${remote}" | cut -d' ' -f1)"
[[ "${marker_sha256}" == "${remote_sha256}" ]]

complete=1
echo "Docker Hub cache migration PASS: ${remote} sha256=${remote_sha256}"
