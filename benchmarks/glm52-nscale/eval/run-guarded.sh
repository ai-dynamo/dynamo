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
kubectl() {
  command kubectl --context "${KUBE_CONTEXT}" \
    --request-timeout="${GLM52_GUARD_REQUEST_TIMEOUT:-60s}" "$@"
}
runner_pod="${EVAL_RUNNER_POD:-glm52-eval-runner}"
binding_remote="/artifacts/glm52-nscale/runtime-bindings/${variant}/active.json"

temporary="$(mktemp -d "${TMPDIR:-/tmp}/glm52-continuity.XXXXXX")"
lock_dir=/artifacts/glm52-nscale/.campaign-run.lock
driver_remote=/workspace/eval/remote-command-driver.py
driver_state="${lock_dir}/command"
invocation_id="$(python3 -c 'import uuid; print(uuid.uuid4().hex)')"
argv_sha256="$(python3 - "$@" <<'PY'
import hashlib
import json
import sys

payload = json.dumps(sys.argv[1:], ensure_ascii=False, separators=(",", ":"))
print(hashlib.sha256(payload.encode()).hexdigest())
PY
)"
poll_seconds="${GLM52_GUARD_POLL_SECONDS:-30}"
retry_seconds="${GLM52_GUARD_RETRY_SECONDS:-5}"
request_timeout="${GLM52_GUARD_REQUEST_TIMEOUT:-60s}"
if [[ ! "${poll_seconds}" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)$ \
  || ! "${retry_seconds}" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)$ \
  || ! "${request_timeout}" =~ ^[1-9][0-9]*(ms|s|m)$ ]]; then
  echo "Guard poll/retry intervals or request timeout are invalid" >&2
  exit 2
fi
lock_acquired=0
remote_command_attempted=0
remote_command_terminal=0
attestation_published=0
preserve_lock=0

remote_driver_request() {
  local max_attempts="$1"
  local operation="$2"
  shift 2
  local attempt=0
  local response
  local request_stderr="${temporary}/driver-request.stderr"
  local -a request=(
    python3 "${driver_remote}" "${operation}"
    --state-dir "${driver_state}"
    --invocation-id "${invocation_id}"
  )
  if [[ "${operation}" == acquire ]]; then
    request+=("$@")
  elif [[ "${operation}" == start ]]; then
    request+=(-- "$@")
  elif [[ "${operation}" == terminate ]]; then
    request+=(--timeout "${1:-10}")
  fi

  while true; do
    attempt=$((attempt + 1))
    : >"${request_stderr}"
    if response="$(kubectl exec "${runner_pod}" -n "${NAMESPACE}" -c runner -- \
      "${request[@]}" 2>"${request_stderr}")" \
      && jq -e --arg invocation_id "${invocation_id}" '
        type == "object"
        and .schema_version == 1
        and .invocation_id == $invocation_id
        and (.state | IN("acquired", "released", "starting", "running", "finished", "orphaned", "lost", "error"))
        and (if .state == "finished"
          then ((.exit_code | type) == "number"
            and (.exit_code | floor) == .exit_code
            and .exit_code >= 0
            and .exit_code <= 255)
          else true
        end)
        and (if .state == "error" then (.error | type == "string") else true end)
      ' <<<"${response}" >/dev/null 2>&1; then
      printf '%s\n' "${response}"
      return 0
    fi
    echo "Detached command ${operation} transport failed; retrying short kubectl exec (attempt ${attempt})." >&2
    if [[ -s "${request_stderr}" ]]; then
      tail -n 20 "${request_stderr}" >&2
    fi
    if ((max_attempts > 0 && attempt >= max_attempts)); then
      return 1
    fi
    sleep "${retry_seconds}"
  done
}

cleanup() {
  local response=""
  local state=""
  if ((remote_command_attempted == 1 && remote_command_terminal == 0 \
      && preserve_lock == 0)); then
    # A local HUP/TERM or tool disconnect must not cancel an expensive remote
    # evaluation. Leave both the supervisor and its durable lock untouched;
    # cancellation is an explicit recovery operation.
    preserve_lock=1
  fi
  if ((remote_command_attempted == 1 && remote_command_terminal == 1 \
      && attestation_published == 0)); then
    preserve_lock=1
  fi
  if ((lock_acquired == 1 && preserve_lock == 0)); then
    response="$(remote_driver_request 1 release 2>/dev/null || true)"
    state="$(jq -r '.state // empty' <<<"${response}" 2>/dev/null || true)"
    if [[ "${state}" != released ]]; then
      preserve_lock=1
    else
      lock_acquired=0
    fi
  fi
  if ((lock_acquired == 1 && preserve_lock == 1)); then
    echo "Retaining ${lock_dir}: detached evaluator state is not terminal." >&2
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

GLM52_REQUIRE_RUNNER=1 "${SCRIPT_DIR}/assert-runner-idle.sh"
kubectl exec "${runner_pod}" -n "${NAMESPACE}" -c runner -- \
  cat "${binding_remote}" > "${temporary}/binding.json"
python3 "${SCRIPT_DIR}/runtime_binding.py" \
  "${temporary}/binding.json" \
  --variant "${variant}" \
  --phase "${campaign_phase}" \
  --output "${temporary}/binding-wrapper.json"
deployment_sha256="$(jq -er '.deployment_sha256' \
  "${temporary}/binding-wrapper.json")"
acquired_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
lock_response="$(remote_driver_request 0 acquire \
  --variant "${variant}" \
  --campaign-phase "${campaign_phase}" \
  --attestation "${attestation_relative}" \
  --argv-sha256 "${argv_sha256}" \
  --deployment-sha256 "${deployment_sha256}" \
  --acquired-at "${acquired_at}")"
lock_state="$(jq -r '.state' <<<"${lock_response}")"
if [[ "${lock_state}" != acquired ]]; then
  jq -r '.error // "campaign lock acquisition failed"' <<<"${lock_response}" >&2
  exit 1
fi
lock_acquired=1
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
kubectl exec -i "${runner_pod}" -n "${NAMESPACE}" -c runner -- \
  /bin/bash -eu -c 'umask 077; cat >"$1/pre.json"' -- "${lock_dir}" \
  < "${temporary}/pre.json"

remote_command_attempted=1
driver_response="$(remote_driver_request 0 start "$@")"
driver_state_value="$(jq -r '.state' <<<"${driver_response}")"
started_epoch="$(date +%s)"
while true; do
  case "${driver_state_value}" in
    finished)
      command_exit_code="$(jq -er '.exit_code' <<<"${driver_response}")"
      remote_command_terminal=1
      break
      ;;
    running|starting)
      echo "Guarded evaluator command is ${driver_state_value}; elapsed $(( $(date +%s) - started_epoch ))s."
      sleep "${poll_seconds}"
      driver_response="$(remote_driver_request 0 status)"
      driver_state_value="$(jq -r '.state' <<<"${driver_response}")"
      ;;
    orphaned|lost)
      echo "Detached evaluator supervisor is ${driver_state_value}; refusing continuity attestation." >&2
      preserve_lock=1
      exit 1
      ;;
    error)
      jq -r '.error // "detached evaluator driver error"' \
        <<<"${driver_response}" >&2
      preserve_lock=1
      exit 1
      ;;
    *)
      echo "Unknown detached evaluator state: ${driver_state_value}" >&2
      preserve_lock=1
      exit 1
      ;;
  esac
done
if ((command_exit_code != 0)); then
  echo "Detached evaluator command exited ${command_exit_code}; last output follows." >&2
  kubectl exec "${runner_pod}" -n "${NAMESPACE}" -c runner -- \
    tail -n 200 "${driver_state}/output.log" >&2 2>/dev/null || true
fi
if ! GLM52_REQUIRE_RUNNER=1 GLM52_ALLOW_LOCK_INVOCATION_ID="${invocation_id}" \
  "${SCRIPT_DIR}/assert-runner-idle.sh"; then
  echo "Evaluator processes or containers remained after the detached command; refusing continuity attestation." >&2
  exit 1
fi

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
attestation_published=1

if ((command_exit_code != 0)); then
  exit "${command_exit_code}"
fi
