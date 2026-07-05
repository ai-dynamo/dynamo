#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMPAIGN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
# shellcheck disable=SC1091
source "${CAMPAIGN_DIR}/campaign.env"
glm52_require_cluster_env KUBE_CONTEXT NAMESPACE
NAMESPACE="${GLM52_NAMESPACE:-${NAMESPACE}}"
kubectl() { command kubectl --context "${KUBE_CONTEXT}" "$@"; }
POD="${EVAL_RUNNER_POD:-glm52-eval-runner}"
archive="$(mktemp "${TMPDIR:-/tmp}/glm52-eval.XXXXXX.tar")"
snapshot="$(mktemp -d "${TMPDIR:-/tmp}/glm52-source.XXXXXX")"
cleanup() {
  rm -f "${archive}"
  rm -rf "${snapshot}"
}
trap cleanup EXIT

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

ready="$(kubectl get pod "${POD}" -n "${NAMESPACE}" \
  -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' \
  2>/dev/null || true)"
if [[ "${ready}" != True ]]; then
  echo "Evaluation runner ${NAMESPACE}/${POD} is not ready" >&2
  exit 1
fi
kubectl exec -n "${NAMESPACE}" "${POD}" -c runner -- \
  docker info >/dev/null
kubectl exec -n "${NAMESPACE}" "${POD}" -c runner -- \
  docker compose version >/dev/null
"${SCRIPT_DIR}/assert-runner-idle.sh"

source_commit="$("${SCRIPT_DIR}/assert-pinned-source.sh")"
campaign_relative="benchmarks/glm52-nscale"

git -C "${REPO_ROOT}" archive --format=tar "${source_commit}" -- \
  "${campaign_relative}/campaign.env" "${campaign_relative}/eval" \
  | tar -xf - -C "${snapshot}"
COPYFILE_DISABLE=1 tar \
  --no-xattrs \
  -C "${snapshot}/${campaign_relative}" \
  -cf "${archive}" campaign.env eval

bundle_sha256="$(sha256_file "${archive}")"
source_branch="$(git -C "${REPO_ROOT}" branch --show-current)"
source_provenance="${snapshot}/source-provenance.json"
python3 "${snapshot}/${campaign_relative}/eval/source_provenance.py" build \
  --source-root "${snapshot}/${campaign_relative}" \
  --source-commit "${source_commit}" \
  --source-branch "${source_branch}" \
  --bundle-sha256 "${bundle_sha256}" \
  --output "${source_provenance}"
remote_archive="/workspace/.glm52-eval-${bundle_sha256}.tar"
remote_staging="/workspace/.glm52-eval-${bundle_sha256}.staging"

kubectl exec -i -n "${NAMESPACE}" "${POD}" -c runner -- \
  /bin/bash -eu -c 'cat >"$1"' -- "${remote_archive}" <"${archive}"

remote_sha256="$(kubectl exec -n "${NAMESPACE}" "${POD}" -c runner -- \
  sha256sum "${remote_archive}" | awk '{print $1}')"
if [[ "${remote_sha256}" != "${bundle_sha256}" ]]; then
  kubectl exec -n "${NAMESPACE}" "${POD}" -c runner -- rm -f "${remote_archive}"
  echo "Evaluation source transfer hash mismatch: local=${bundle_sha256}, remote=${remote_sha256}" >&2
  exit 1
fi

kubectl exec -n "${NAMESPACE}" "${POD}" -c runner -- \
  /bin/bash -eu -c '
    rm -rf "$2"
    mkdir -p "$2"
    tar -C "$2" -xf "$1"
    test -s "$2/campaign.env"
    test -d "$2/eval"
  ' -- "${remote_archive}" "${remote_staging}"

kubectl exec -n "${NAMESPACE}" "${POD}" -c runner -- \
  rm -f "${remote_staging}/source-provenance.json"
kubectl exec -i -n "${NAMESPACE}" "${POD}" -c runner -- \
  /bin/bash -eu -c 'cat >"$1/source-provenance.json"' -- "${remote_staging}" \
  < "${source_provenance}"

kubectl exec -n "${NAMESPACE}" "${POD}" -c runner -- \
  /bin/bash -eu -c '
    rm -rf /workspace/eval
    mv "$1/eval" /workspace/eval
    mv "$1/campaign.env" /workspace/campaign.env
    mv "$1/source-provenance.json" /workspace/source-provenance.json
    rmdir "$1"
    rm -f "$2"
  ' -- "${remote_staging}" "${remote_archive}"

echo "Evaluation sources synced to ${POD}:/workspace (${bundle_sha256})"
