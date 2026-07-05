#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
  echo "usage: $0 {bfcl|swebench|terminalbench} /artifacts/glm52-nscale/... LOCAL_DIR" >&2
  exit 2
}

[[ $# -eq 3 ]] || usage
suite="$1"
remote_dir="${2%/}"
local_dir="${3%/}"
case "${suite}" in
  bfcl)
    files=(
      summary.json complete-validation.json metadata.json endpoint-models.json
      runtime-continuity.json expected-ids.json failures.jsonl
      environment-lock.json environment.freeze.txt
    )
    ;;
  swebench)
    files=(
      score.json generation-summary.json run-metadata.json run-scope.json
      runtime-continuity.json environment.freeze.txt
      environment.normalized.freeze.txt
    )
    ;;
  terminalbench)
    files=(summary.json trials.csv task-images.json runtime-continuity.json)
    ;;
  *) usage ;;
esac

remote_prefix=/artifacts/glm52-nscale/
remote_relative="${remote_dir#"${remote_prefix}"}"
if [[ "${remote_dir}" == "${remote_relative}" \
  || -z "${remote_relative}" \
  || ! "${remote_relative}" =~ ^[A-Za-z0-9._/-]+$ \
  || "${remote_relative}" == *//* ]]; then
  echo "REMOTE_DIR must be a directory under /artifacts/glm52-nscale" >&2
  exit 2
fi
case "/${remote_relative}/" in
  */../*|*/./*)
    echo "REMOTE_DIR must not contain dot path components" >&2
    exit 2
    ;;
esac
if [[ -z "${local_dir}" || "${local_dir}" == / || "${local_dir}" == . || "${local_dir}" == .. ]]; then
  echo "LOCAL_DIR must name a new result directory" >&2
  exit 2
fi
if [[ -e "${local_dir}" || -L "${local_dir}" ]]; then
  echo "Refusing to overwrite local result directory: ${local_dir}" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/campaign.env"
glm52_require_cluster_env KUBE_CONTEXT NAMESPACE
kubectl() { command kubectl --context "${KUBE_CONTEXT}" "$@"; }
runner_pod="${EVAL_RUNNER_POD:-glm52-eval-runner}"

local_parent="$(dirname "${local_dir}")"
local_name="$(basename "${local_dir}")"
mkdir -p "${local_parent}"
staging="$(mktemp -d "${local_parent}/.${local_name}.staging.XXXXXX")"
cleanup() {
  rm -rf "${staging}"
}
trap cleanup EXIT

kubectl exec "${runner_pod}" -n "${NAMESPACE}" -c runner -- \
  /bin/bash -eu -c '
    directory="$1"
    shift
    test -d "${directory}"
    test ! -L "${directory}"
    for file in "$@"; do
      test -f "${directory}/${file}"
      test ! -L "${directory}/${file}"
    done
    exec tar -C "${directory}" -cf - -- "$@"
  ' -- "${remote_dir}" "${files[@]}" \
  | COPYFILE_DISABLE=1 tar -xf - -C "${staging}"

for file in "${files[@]}"; do
  if [[ ! -f "${staging}/${file}" || -L "${staging}/${file}" ]]; then
    echo "Fetched result is missing regular file: ${file}" >&2
    exit 1
  fi
done
entry_count="$(find "${staging}" -mindepth 1 -maxdepth 1 | wc -l | tr -d ' ')"
if ((entry_count != ${#files[@]})); then
  echo "Fetched result contains unexpected top-level entries" >&2
  exit 1
fi

mv "${staging}" "${local_dir}"
trap - EXIT
echo "Fetched ${suite} compact evidence to ${local_dir}"
