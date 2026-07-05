#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
  echo "usage: $0 VARIANT --phase {validation|ab|ba} --attestation /artifacts/.../runtime-continuity.json -- COMMAND [ARG ...]" >&2
  exit 2
}

[[ $# -ge 7 ]] || usage
variant="$1"
shift
[[ "$1" == --phase && $# -ge 5 ]] || usage
campaign_phase="$2"
shift 2
case "${campaign_phase}" in
  validation|ab|ba) ;;
  *) usage ;;
esac
[[ "$1" == --attestation && $# -ge 3 ]] || usage
attestation="$2"
shift 2
[[ "$1" == -- ]] || usage
shift
[[ $# -gt 0 ]] || usage

case "${variant}" in
  dynamo-vllm|dynamo-sglang)
    controller_kind=DynamoGraphDeployment
    controller_resource=dynamographdeployment
    ;;
  vllm-serve|sglang-serve)
    controller_kind=Deployment
    controller_resource=deployment
    ;;
  *) usage ;;
esac
controller_name="glm52-${variant}"
attestation_prefix=/artifacts/glm52-nscale/
attestation_relative="${attestation#"${attestation_prefix}"}"
if [[ "${attestation}" == "${attestation_relative}" \
  || ! "${attestation_relative}" =~ ^[A-Za-z0-9._/-]+/runtime-continuity\.json$ \
  || "${attestation_relative}" == *//* ]]; then
  echo "--attestation must be a runtime-continuity.json path under /artifacts/glm52-nscale" >&2
  exit 2
fi
case "/${attestation_relative}/" in
  */../*|*/./*)
    echo "--attestation must not contain dot path components" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/campaign.env"
glm52_require_cluster_env KUBE_CONTEXT NAMESPACE
kubectl() { command kubectl --context "${KUBE_CONTEXT}" "$@"; }
runner_pod="${EVAL_RUNNER_POD:-glm52-eval-runner}"
binding_remote="/artifacts/glm52-nscale/runtime-bindings/${variant}/active.json"

temporary="$(mktemp -d "${TMPDIR:-/tmp}/glm52-continuity.XXXXXX")"
lock_dir=/artifacts/glm52-nscale/.campaign-run.lock
lock_acquired=0
cleanup() {
  if ((lock_acquired == 1)); then
    kubectl exec "${runner_pod}" -n "${NAMESPACE}" -c runner -- \
      rm -rf "${lock_dir}" >/dev/null 2>&1 || true
  fi
  rm -rf "${temporary}"
}
trap cleanup EXIT

snapshot() {
  local output="$1"
  kubectl get "${controller_resource}" "${controller_name}" \
    -n "${NAMESPACE}" -o json > "${temporary}/controller.json"
  kubectl get pods -n "${NAMESPACE}" \
    -l "glm52.nvidia.com/variant=${variant}" -o json \
    > "${temporary}/pods.json"
  GLM52_CONTROLLER_KIND="${controller_kind}" python3 - \
    "${temporary}/controller.json" "${temporary}/pods.json" "${output}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


controller = json.loads(Path(sys.argv[1]).read_text())
pods_document = json.loads(Path(sys.argv[2]).read_text())
pods = {}
for pod in pods_document["items"]:
    role = pod["metadata"].get("labels", {}).get(
        "glm52.nvidia.com/role", "worker"
    )
    containers = pod["spec"]["containers"]
    if role in pods or len(containers) != 1:
        raise RuntimeError(f"invalid serving topology for role {role!r}")
    container = containers[0]
    statuses = {
        status["name"]: status for status in pod["status"].get("containerStatuses", [])
    }
    status = statuses.get(container["name"])
    if status is None or not status.get("containerID") or not status.get("imageID"):
        raise RuntimeError(f"missing container identity for role {role!r}")
    pods[role] = {
        "name_sha256": digest(pod["metadata"]["name"]),
        "uid_sha256": digest(pod["metadata"]["uid"]),
        "node_name_sha256": digest(pod["spec"]["nodeName"]),
        "image_id": status["imageID"],
        "container_id_sha256": digest(status["containerID"]),
        "restart_count": status["restartCount"],
    }

snapshot = {
    "controller": {
        "kind": os.environ["GLM52_CONTROLLER_KIND"],
        "name": controller["metadata"]["name"],
        "uid_sha256": digest(controller["metadata"]["uid"]),
        "generation": controller["metadata"]["generation"],
    },
    "pods": pods,
}
Path(sys.argv[3]).write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
PY
}

"${SCRIPT_DIR}/assert-runner-idle.sh"
if ! kubectl exec "${runner_pod}" -n "${NAMESPACE}" -c runner -- \
  mkdir "${lock_dir}" 2>/dev/null; then
  echo "Another guarded campaign run owns ${lock_dir}; refusing concurrent execution." >&2
  kubectl exec "${runner_pod}" -n "${NAMESPACE}" -c runner -- \
    cat "${lock_dir}/owner.json" >&2 2>/dev/null || true
  exit 1
fi
lock_acquired=1
jq -n --arg variant "${variant}" --arg acquired_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{schema_version: 1, variant: $variant, acquired_at: $acquired_at}' \
  | kubectl exec -i "${runner_pod}" -n "${NAMESPACE}" -c runner -- \
      /bin/bash -eu -c 'cat >"$1/owner.json"' -- "${lock_dir}"
kubectl exec "${runner_pod}" -n "${NAMESPACE}" -c runner -- \
  cat "${binding_remote}" > "${temporary}/binding.json"
python3 "${SCRIPT_DIR}/runtime_binding.py" \
  "${temporary}/binding.json" \
  --variant "${variant}" \
  --phase "${campaign_phase}" \
  --output "${temporary}/binding-wrapper.json"
deployment_sha256="$(jq -er '.deployment_sha256' \
  "${temporary}/binding-wrapper.json")"
if kubectl exec "${runner_pod}" -n "${NAMESPACE}" -c runner -- \
  test -e "${attestation}"; then
  existing_attestation="${temporary}/existing-runtime-continuity.json"
  kubectl exec "${runner_pod}" -n "${NAMESPACE}" -c runner -- \
    cat "${attestation}" > "${existing_attestation}"
  python3 "${SCRIPT_DIR}/runtime_binding.py" \
    "${temporary}/binding.json" \
    --variant "${variant}" \
    --phase "${campaign_phase}" \
    --continuity "${existing_attestation}" \
    --allow-command-failure >/dev/null
  existing_exit_code="$(jq -er '.command_exit_code' "${existing_attestation}")"
  if ((existing_exit_code == 0)); then
    echo "Refusing to overwrite successful runtime attestation: ${attestation}" >&2
    exit 1
  fi
  if command -v sha256sum >/dev/null 2>&1; then
    existing_sha256="$(sha256sum "${existing_attestation}" | awk '{print $1}')"
  else
    existing_sha256="$(shasum -a 256 "${existing_attestation}" | awk '{print $1}')"
  fi
  failure_dir="${attestation%.json}.failures"
  failure_attestation="${failure_dir}/${existing_sha256}.json"
  kubectl exec "${runner_pod}" -n "${NAMESPACE}" -c runner -- \
    mkdir -p "${failure_dir}"
  if kubectl exec "${runner_pod}" -n "${NAMESPACE}" -c runner -- \
    test -e "${failure_attestation}"; then
    archived_sha256="$(kubectl exec "${runner_pod}" -n "${NAMESPACE}" -c runner -- \
      sha256sum "${failure_attestation}" | awk '{print $1}')"
    if [[ "${archived_sha256}" != "${existing_sha256}" ]]; then
      echo "Archived failed attestation digest mismatch: ${failure_attestation}" >&2
      exit 1
    fi
    kubectl exec "${runner_pod}" -n "${NAMESPACE}" -c runner -- \
      rm -f "${attestation}"
  else
    kubectl exec "${runner_pod}" -n "${NAMESPACE}" -c runner -- \
      mv "${attestation}" "${failure_attestation}"
  fi
  echo "Archived failed runtime attestation: ${failure_attestation}"
fi
snapshot "${temporary}/pre.json"
if ! jq -e '[.pods[].restart_count] | length > 0 and all(.[]; . == 0)' \
  "${temporary}/pre.json" >/dev/null; then
  echo "Serving containers must have zero restarts before a guarded run" >&2
  exit 1
fi
if ! jq -e --slurpfile actual "${temporary}/pre.json" '
  .controller.kind == $actual[0].controller.kind
  and .controller.name == $actual[0].controller.name
  and .controller.uid_sha256 == $actual[0].controller.uid_sha256
  and .controller.generation == $actual[0].controller.generation
  and (.pods | with_entries(.value |= {
      name_sha256,
      uid_sha256,
      node_name_sha256,
      image_id
    })) == ($actual[0].pods | with_entries(.value |= {
      name_sha256,
      uid_sha256,
      node_name_sha256,
      image_id
    }))
' "${temporary}/binding.json" >/dev/null; then
  echo "Active runtime does not match the published ${variant} binding" >&2
  exit 1
fi
pre_captured_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

set +e
kubectl exec "${runner_pod}" -n "${NAMESPACE}" -c runner -- "$@"
command_exit_code=$?
set -e

if ! snapshot "${temporary}/post.json"; then
  echo "Failed to capture serving runtime identity after guarded command" >&2
  if ((command_exit_code != 0)); then
    exit "${command_exit_code}"
  fi
  exit 1
fi
post_captured_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if ! jq -e '[.pods[].restart_count] | length > 0 and all(.[]; . == 0)' \
  "${temporary}/post.json" >/dev/null; then
  echo "Serving containers must have zero restarts after a guarded run" >&2
  exit 1
fi
if ! jq -e -s '.[0] == .[1]' \
  "${temporary}/pre.json" "${temporary}/post.json" >/dev/null; then
  echo "Serving runtime identity changed during the guarded command (command exit ${command_exit_code})" >&2
  exit 1
fi

jq -n \
  --arg variant "${variant}" \
  --arg campaign_phase "${campaign_phase}" \
  --arg deployment_sha256 "${deployment_sha256}" \
  --arg pre_captured_at "${pre_captured_at}" \
  --arg post_captured_at "${post_captured_at}" \
  --argjson command_exit_code "${command_exit_code}" \
  --argjson stable true \
  --slurpfile pre "${temporary}/pre.json" \
  --slurpfile post "${temporary}/post.json" '
    {
      schema_version: 1,
      variant: $variant,
      campaign_phase: $campaign_phase,
      deployment_sha256: $deployment_sha256,
      command_exit_code: $command_exit_code,
      stable: $stable,
      pre_captured_at: $pre_captured_at,
      post_captured_at: $post_captured_at,
      pre: $pre[0],
      post: $post[0]
    }
  ' > "${temporary}/runtime-continuity.json"

if ((command_exit_code == 0)); then
  python3 "${SCRIPT_DIR}/runtime_binding.py" \
    "${temporary}/binding.json" \
    --variant "${variant}" \
    --phase "${campaign_phase}" \
    --continuity "${temporary}/runtime-continuity.json" >/dev/null
else
  python3 "${SCRIPT_DIR}/runtime_binding.py" \
    "${temporary}/binding.json" \
    --variant "${variant}" \
    --phase "${campaign_phase}" \
    --continuity "${temporary}/runtime-continuity.json" \
    --allow-command-failure >/dev/null
fi

attestation_dir="$(dirname "${attestation}")"
kubectl exec "${runner_pod}" -n "${NAMESPACE}" -c runner -- \
  mkdir -p "${attestation_dir}"
kubectl exec -i "${runner_pod}" -n "${NAMESPACE}" -c runner -- \
  /bin/bash -eu -c '
    temporary="$1.tmp"
    cat >"${temporary}"
    python3 -m json.tool "${temporary}" >/dev/null
    mv "${temporary}" "$1"
  ' -- "${attestation}" < "${temporary}/runtime-continuity.json"

if ((command_exit_code != 0)); then
  exit "${command_exit_code}"
fi
