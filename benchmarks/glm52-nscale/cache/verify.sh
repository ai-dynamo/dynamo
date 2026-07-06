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

"${SCRIPT_DIR}/assert-ready.sh" >/dev/null
"${ROOT_DIR}/eval/assert-runner-idle.sh"

runner=glm52-eval-runner
test_ref=docker.io/library/hello-world:latest
temporary="$(mktemp "${TMPDIR:-/tmp}/glm52-cache-proof.XXXXXX.json")"
cleanup() { rm -f "${temporary}"; }
trap cleanup EXIT

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

sha256_text() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum | cut -d' ' -f1
  else
    shasum -a 256 | cut -d' ' -f1
  fi
}

quota_remaining() {
  kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- bash -eu -c '
    repository=library/hello-world
    token=$(curl -fsSL "https://auth.docker.io/token?service=registry.docker.io&scope=repository:${repository}:pull" | jq -er .token)
    curl -sSI \
      -H "Authorization: Bearer ${token}" \
      -H "Accept: application/vnd.oci.image.index.v1+json" \
      "https://registry-1.docker.io/v2/${repository}/manifests/latest" \
      | tr -d "\r" \
      | awk -F": " "tolower(\$1) == \"ratelimit-remaining\" {print \$2}"
  '
}

captured_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- \
  docker image rm "${test_ref}" >/dev/null 2>&1 || true
kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- \
  docker pull --quiet --platform linux/amd64 "${test_ref}" >/dev/null
warm_image_id="$(kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- \
  docker image inspect --format '{{.Id}}' "${test_ref}")"
kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- \
  docker image rm "${test_ref}" >/dev/null
quota_before="$(quota_remaining)"
cache_bytes_before="$(kubectl exec deployment/dockerhub-pull-cache \
  -n "${NAMESPACE}" -c registry -- \
  du -sb /artifacts/cache/dockerhub-registry | cut -f1)"

image_ids=()
for _ in 1 2; do
  kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- \
    docker image rm "${test_ref}" >/dev/null 2>&1 || true
  kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- \
    docker pull --quiet --platform linux/amd64 "${test_ref}" >/dev/null
  image_ids+=("$(kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- \
    docker image inspect --format '{{.Id}}' "${test_ref}")")
done
kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- \
  docker image rm "${test_ref}" >/dev/null

quota_after="$(quota_remaining)"
cache_bytes_after="$(kubectl exec deployment/dockerhub-pull-cache \
  -n "${NAMESPACE}" -c registry -- \
  du -sb /artifacts/cache/dockerhub-registry | cut -f1)"
[[ "${image_ids[0]}" == "${image_ids[1]}" ]]
[[ "${warm_image_id}" == "${image_ids[0]}" ]]
[[ "${quota_before}" == "${quota_after}" ]]
[[ "${quota_before}" =~ ^[0-9]+\;w=[0-9]+$ ]]

mirror="$(kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- \
  docker info --format '{{json .RegistryConfig.Mirrors}}' | jq -er '.[0]')"
registry_image="$(kubectl get deployment dockerhub-pull-cache -n "${NAMESPACE}" \
  -o jsonpath='{.spec.template.spec.containers[0].image}')"
runner_uid="$(kubectl get pod "${runner}" -n "${NAMESPACE}" \
  -o jsonpath='{.metadata.uid}')"
runner_uid_sha256="$(printf '%s' "${runner_uid}" | sha256_text)"
cache_pvc_uid="$(kubectl get pvc dockerhub-pull-cache-data -n "${NAMESPACE}" \
  -o jsonpath='{.metadata.uid}')"
cache_pvc_uid_sha256="$(printf '%s' "${cache_pvc_uid}" | sha256_text)"
cache_storage_class="$(kubectl get pvc dockerhub-pull-cache-data \
  -n "${NAMESPACE}" -o jsonpath='{.spec.storageClassName}')"
cache_capacity="$(kubectl get pvc dockerhub-pull-cache-data \
  -n "${NAMESPACE}" -o jsonpath='{.status.capacity.storage}')"
migration_marker_sha256="$(kubectl exec deployment/dockerhub-pull-cache \
  -n "${NAMESPACE}" -c registry -- \
  sha256sum /artifacts/cache/.glm52-migration-v1.json | cut -d' ' -f1)"
restart_counts="$(kubectl get pod "${runner}" -n "${NAMESPACE}" -o json \
  | jq -c '[.status.containerStatuses[] | {name, restart_count: .restartCount}] | sort_by(.name)')"

jq -n \
  --arg captured_at "${captured_at}" \
  --arg mirror "${mirror}" \
  --arg registry_image "${registry_image}" \
  --arg runner_uid_sha256 "${runner_uid_sha256}" \
  --arg cache_pvc_uid_sha256 "${cache_pvc_uid_sha256}" \
  --arg cache_storage_class "${cache_storage_class}" \
  --arg cache_capacity "${cache_capacity}" \
  --arg migration_marker_sha256 "${migration_marker_sha256}" \
  --arg test_ref "${test_ref}" \
  --arg warm_image_id "${warm_image_id}" \
  --arg image_id "${image_ids[0]}" \
  --arg quota_before "${quota_before}" \
  --arg quota_after "${quota_after}" \
  --argjson restart_counts "${restart_counts}" \
  --argjson cache_bytes_before "${cache_bytes_before}" \
  --argjson cache_bytes_after "${cache_bytes_after}" \
  '{
    schema_version: 2,
    captured_at: $captured_at,
    mirror: $mirror,
    registry_image: $registry_image,
    runner_uid_sha256: $runner_uid_sha256,
    cache_pvc_uid_sha256: $cache_pvc_uid_sha256,
    cache_storage_class: $cache_storage_class,
    cache_capacity: $cache_capacity,
    migration_marker_sha256: $migration_marker_sha256,
    restart_counts: $restart_counts,
    cache_bytes_before: $cache_bytes_before,
    cache_bytes_after: $cache_bytes_after,
    test_ref: $test_ref,
    warm_image_id: $warm_image_id,
    image_id: $image_id,
    repeated_image_id_equal: true,
    quota_before: $quota_before,
    quota_after: $quota_after,
    quota_unchanged: true
  }' >"${temporary}"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
remote=/artifacts/glm52-nscale/cache/verifications/${timestamp}.json
kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- \
  mkdir -p /artifacts/glm52-nscale/cache/verifications
kubectl exec -i "${runner}" -n "${NAMESPACE}" -c runner -- \
  bash -eu -c 'temporary="${1}.new"; cat >"${temporary}"; mv "${temporary}" "$1"' \
  -- "${remote}" <"${temporary}"
local_sha256="$(sha256_file "${temporary}")"
remote_sha256="$(kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- \
  sha256sum "${remote}" | cut -d' ' -f1)"
[[ "${remote_sha256}" == "${local_sha256}" ]]

echo "Docker Hub cache verification PASS: ${remote} sha256=${remote_sha256}"
