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

POD="${EVAL_RUNNER_POD:-glm52-eval-runner}"
PVC=glm52-benchmark-artifacts
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_parent="${ROOT_DIR}/results/runtime/eval-runner"
output_dir="${output_parent}/${timestamp}"
staging_dir="${output_parent}/.${timestamp}.staging"

cleanup() {
  rm -rf "${staging_dir}"
}
trap cleanup EXIT

for command in kubectl jq; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "Required command not found: ${command}" >&2
    exit 1
  fi
done

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

mkdir -p "${staging_dir}"

kubectl get pod "${POD}" -n "${NAMESPACE}" -o json | jq '
  . as $pod
  | {
      schema_version: 1,
      captured_at: (now | todateiso8601),
      pod: {
        namespace: $pod.metadata.namespace,
        name: $pod.metadata.name,
        uid: $pod.metadata.uid,
        created_at: $pod.metadata.creationTimestamp,
        manifest_sha256: $pod.metadata.annotations["benchmarks.nvidia.com/manifest-sha256"],
        node: $pod.spec.nodeName,
        phase: $pod.status.phase,
        qos_class: $pod.status.qosClass,
        conditions: [
          $pod.status.conditions[]?
          | {type, status, reason: (.reason // null), last_transition_time: .lastTransitionTime}
        ],
        init_containers: [
          $pod.spec.initContainers[] as $container
          | {
              name: $container.name,
              image: $container.image,
              image_id: ([$pod.status.initContainerStatuses[]?
                | select(.name == $container.name) | .imageID][0] // null)
            }
        ],
        containers: [
          $pod.spec.containers[] as $container
          | {
              name: $container.name,
              image: $container.image,
              image_id: ([$pod.status.containerStatuses[]?
                | select(.name == $container.name) | .imageID][0] // null),
              ready: ([$pod.status.containerStatuses[]?
                | select(.name == $container.name) | .ready][0] // false),
              resources: $container.resources
            }
        ],
        volumes: [
          $pod.spec.volumes[]
          | if .persistentVolumeClaim then
              {name, type: "persistentVolumeClaim", claim_name: .persistentVolumeClaim.claimName}
            elif .emptyDir then
              {name, type: "emptyDir", size_limit: (.emptyDir.sizeLimit // null)}
            else
              {name, type: "other"}
            end
        ]
      }
    }
' >"${staging_dir}/runtime-identity.json"

node_name="$(jq -r '.pod.node' "${staging_dir}/runtime-identity.json")"
kubectl get node "${node_name}" -o json | jq '
  {
    name: .metadata.name,
    uid: .metadata.uid,
    labels: {
      architecture: .metadata.labels["kubernetes.io/arch"],
      operating_system: .metadata.labels["kubernetes.io/os"],
      instance_type: .metadata.labels["node.kubernetes.io/instance-type"]
    },
    node_info: .status.nodeInfo,
    capacity: {cpu: .status.capacity.cpu, memory: .status.capacity.memory},
    allocatable: {cpu: .status.allocatable.cpu, memory: .status.allocatable.memory}
  }
' >"${staging_dir}/node-identity.json"

kubectl get pvc "${PVC}" -n "${NAMESPACE}" -o json | jq '
  {
    name: .metadata.name,
    uid: .metadata.uid,
    phase: .status.phase,
    storage_class: .spec.storageClassName,
    access_modes: .spec.accessModes,
    volume_mode: .spec.volumeMode,
    requested_storage: .spec.resources.requests.storage,
    capacity: .status.capacity.storage
  }
' >"${staging_dir}/pvc-identity.json"

kubectl exec -i -n "${NAMESPACE}" "${POD}" -c runner -- python3 - <<'PY' \
  >"${staging_dir}/toolchain.json"
import importlib.metadata
import json
import os
import pathlib
import platform
import subprocess


def command_version(argv):
    try:
        return subprocess.run(
            argv,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"


os_release = {}
for line in pathlib.Path("/etc/os-release").read_text().splitlines():
    if "=" in line:
        key, value = line.split("=", 1)
        if key in {"ID", "VERSION_ID", "PRETTY_NAME"}:
            os_release[key.lower()] = value.strip('"')

try:
    uv_package = importlib.metadata.version("uv")
except importlib.metadata.PackageNotFoundError:
    uv_package = None

print(json.dumps({
    "schema_version": 1,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "python_implementation": platform.python_implementation(),
    "cpu_count": os.cpu_count(),
    "os_release": os_release,
    "uv_package": uv_package,
    "commands": {
        "python": command_version(["python3", "--version"]),
        "uv": command_version(["uv", "--version"]),
        "git": command_version(["git", "--version"]),
        "jq": command_version(["jq", "--version"]),
        "tmux": command_version(["tmux", "-V"]),
        "docker_cli": command_version(["docker", "--version"]),
        "docker_compose": command_version(["docker", "compose", "version"]),
    },
}, indent=2, sort_keys=True))
PY

kubectl exec -n "${NAMESPACE}" "${POD}" -c runner -- \
  /bin/bash -eu -c \
  'docker version --format "{{json .}}" | jq "{Client: .Client, Server: .Server}"' \
  >"${staging_dir}/docker-version.json"

kubectl exec -n "${NAMESPACE}" "${POD}" -c runner -- \
  /bin/bash -eu -c \
  'docker info --format "{{json .}}" | jq "{ServerVersion, Driver, DockerRootDir, OperatingSystem, OSType, Architecture, KernelVersion, CgroupDriver, CgroupVersion, NCPU, MemTotal, Containers, ContainersRunning, Images}"' \
  >"${staging_dir}/docker-info.json"

if kubectl exec -n "${NAMESPACE}" "${POD}" -c runner -- \
  test -s /workspace/source-provenance.json; then
  kubectl exec -n "${NAMESPACE}" "${POD}" -c runner -- \
    cat /workspace/source-provenance.json \
    | jq . >"${staging_dir}/source-provenance.json"
else
  printf '%s\n' '{"present": false}' >"${staging_dir}/source-provenance.json"
fi

kubectl version --client -o json | jq . >"${staging_dir}/kubectl-client.json"

mkdir -p "${output_parent}"
mv "${staging_dir}" "${output_dir}"
trap - EXIT
echo "Captured secret-free evaluation runner identity under ${output_dir}"
