#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -gt 1 || ($# -eq 1 && "$1" != --allow-active) ]]; then
  echo "usage: $0 [--allow-active]" >&2
  exit 2
fi
allow_active=0
if [[ ${1:-} == --allow-active ]]; then
  allow_active=1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/campaign.env"
glm52_require_cluster_env KUBE_CONTEXT NAMESPACE
kubectl() { command kubectl --context "${KUBE_CONTEXT}" "$@"; }

runner=glm52-eval-runner
mirror=http://dockerhub-pull-cache:5000
registry=dockerhub-pull-cache:5000

"${SCRIPT_DIR}/assert-ready.sh" --skip-runner-mirror >/dev/null
if ((allow_active == 0)); then
  "${ROOT_DIR}/eval/assert-runner-idle.sh"
else
  echo "Explicit active-work mirror reload enabled; immutable task-image guards remain authoritative."
fi

pod_uid_before="$(kubectl get pod "${runner}" -n "${NAMESPACE}" \
  -o jsonpath='{.metadata.uid}')"
restart_count_before="$(kubectl get pod "${runner}" -n "${NAMESPACE}" \
  -o jsonpath='{.status.containerStatuses[?(@.name=="dind")].restartCount}')"

jq -nc --arg mirror "${mirror}" --arg registry "${registry}" \
  '{"registry-mirrors": [$mirror], "insecure-registries": [$registry]}' \
  | kubectl exec -i "${runner}" -n "${NAMESPACE}" -c dind -- \
      sh -eu -c '
        umask 077
        mkdir -p /etc/docker
        temporary=/etc/docker/daemon.json.new
        cat >"${temporary}"
        mv "${temporary}" /etc/docker/daemon.json
        kill -HUP 1
      '

deadline=$((SECONDS + 60))
while true; do
  mirrors="$(kubectl exec "${runner}" -n "${NAMESPACE}" -c runner -- \
    docker info --format '{{json .RegistryConfig.Mirrors}}' 2>/dev/null || true)"
  if jq -e --arg mirror "${mirror}/" 'index($mirror) != null' \
    <<<"${mirrors}" >/dev/null 2>&1; then
    break
  fi
  if ((SECONDS >= deadline)); then
    echo "Docker daemon did not publish the registry mirror after SIGHUP" >&2
    exit 1
  fi
  sleep 2
done

pod_uid_after="$(kubectl get pod "${runner}" -n "${NAMESPACE}" \
  -o jsonpath='{.metadata.uid}')"
restart_count_after="$(kubectl get pod "${runner}" -n "${NAMESPACE}" \
  -o jsonpath='{.status.containerStatuses[?(@.name=="dind")].restartCount}')"
if [[ "${pod_uid_after}" != "${pod_uid_before}" \
  || "${restart_count_after}" != "${restart_count_before}" ]]; then
  echo "Runner identity or DinD restart count changed during mirror reload" >&2
  exit 1
fi

echo "Docker Hub mirror active without a runner restart: ${mirrors}"
