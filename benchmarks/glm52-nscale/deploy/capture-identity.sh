#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <variant> {validation|ab|ba}" >&2
  exit 2
fi

variant="$1"
campaign_phase="$2"
case "${campaign_phase}" in
  validation|ab|ba) ;;
  *) echo "Unknown campaign phase: ${campaign_phase}" >&2; exit 2 ;;
esac
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/campaign.env"
glm52_require_cluster_env KUBE_CONTEXT NAMESPACE NODE_NAME EXPECTED_GPU_UUIDS
kubectl() { command kubectl --context "${KUBE_CONTEXT}" "$@"; }
source_commit="$("${SCRIPT_DIR}/assert-pinned-source.sh")"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_parent="${ROOT_DIR}/results/runtime/${variant}/${campaign_phase}"
final_dir="${output_parent}/${timestamp}"
out_dir="${output_parent}/.${timestamp}.staging"
selector="glm52.nvidia.com/variant=${variant}"
case "${variant}" in
  dynamo-vllm)
    expected_image="${VLLM_IMAGE}"
    expected_pods=2
    is_dynamo=true
    context_flag="--max-model-len=${MAX_MODEL_LEN}"
    service_name="glm52-dynamo-vllm-frontend"
    endpoint="http://${service_name}:8000/v1"
    controller_kind="DynamoGraphDeployment"
    controller_name="glm52-dynamo-vllm"
    ;;
  vllm-serve)
    expected_image="${VLLM_IMAGE}"
    expected_pods=1
    is_dynamo=false
    context_flag="--max-model-len=${MAX_MODEL_LEN}"
    service_name="glm52-vllm-serve"
    endpoint="http://${service_name}:8000/v1"
    controller_kind="Deployment"
    controller_name="glm52-vllm-serve"
    ;;
  dynamo-sglang)
    expected_image="${SGLANG_IMAGE}"
    expected_pods=2
    is_dynamo=true
    context_flag="--context-length=${MAX_MODEL_LEN}"
    service_name="glm52-dynamo-sglang-frontend"
    endpoint="http://${service_name}:8000/v1"
    controller_kind="DynamoGraphDeployment"
    controller_name="glm52-dynamo-sglang"
    ;;
  sglang-serve)
    expected_image="${SGLANG_IMAGE}"
    expected_pods=1
    is_dynamo=false
    context_flag="--context-length=${MAX_MODEL_LEN}"
    service_name="glm52-sglang-serve"
    endpoint="http://${service_name}:8000/v1"
    controller_kind="Deployment"
    controller_name="glm52-sglang-serve"
    ;;
  *)
    echo "Unknown variant: ${variant}" >&2
    exit 2
    ;;
esac
cleanup() {
  rm -rf "${out_dir}"
}
trap cleanup EXIT
if [[ -e "${final_dir}" ]]; then
  echo "Capture destination already exists: ${final_dir}" >&2
  exit 1
fi
mkdir -p "${out_dir}"

if [[ "${is_dynamo}" == true ]]; then
  kubectl get dynamographdeployment "glm52-${variant}" -n "${NAMESPACE}" -o json \
    > "${out_dir}/controller.json"
  jq -e '
    .status.observedGeneration == .metadata.generation
    and .status.state == "successful"
    and ([.status.conditions[]? | select(.type == "Ready" and .status == "True")]
      | length == 1)
    and (.status.components | length == 2)
    and ([.status.components[]
      | select(.replicas == 1 and .updatedReplicas == 1 and .readyReplicas == 1)]
      | length == 2)
  ' "${out_dir}/controller.json" >/dev/null
else
  kubectl get deployment "glm52-${variant}" -n "${NAMESPACE}" -o json \
    > "${out_dir}/controller.json"
  jq -e '
    .status.observedGeneration == .metadata.generation
    and .status.replicas == 1
    and .status.updatedReplicas == 1
    and .status.readyReplicas == 1
    and .status.availableReplicas == 1
  ' "${out_dir}/controller.json" >/dev/null
fi

kubectl get pods -n "${NAMESPACE}" -l "${selector}" -o json > "${out_dir}/pods.json"
kubectl get pods -n "${NAMESPACE}" -l "${selector}" -o wide > "${out_dir}/pods.txt"
if ! jq -e \
  --arg expected_image "${expected_image}" \
  --arg gpu_node "${NODE_NAME}" \
  --arg model_revision "${MODEL_REVISION}" \
  --arg context_flag "${context_flag}" \
  --argjson expected_pods "${expected_pods}" \
  --argjson is_dynamo "${is_dynamo}" \
  --argjson tp_size "${TP_SIZE}" '
  def gpu_count:
    ([.spec.containers[]?.resources.requests["nvidia.com/gpu"] // "0" | tonumber]
      | add // 0);
  def argv_text:
    ([.spec.containers[]? | ((.command // []) + (.args // []))[]] | join("\n"));

  (.items | length) == $expected_pods
  and (.items | all(.[];
    . as $pod
    | $pod.status.phase == "Running"
    and ([$pod.spec.containers[].image]
      | length > 0 and all(.[]; . == $expected_image))
    and ([$pod.status.containerStatuses[]?] | length)
      == ([$pod.spec.containers[]] | length)
    and ([$pod.status.containerStatuses[]?]
      | all(.[];
        .ready == true
        and .restartCount == 0
        and (.containerID | type == "string")
        and (.containerID | test("sha256:[0-9a-f]{64}$"))
        and (.imageID | type == "string")
        and (.imageID | test("@sha256:[0-9a-f]{64}$"))))
    and (($pod | gpu_count) == 0 or $pod.spec.nodeName == $gpu_node)))
  and (if $is_dynamo then
    ([.items[]
      | select(.metadata.labels["glm52.nvidia.com/role"] == "frontend")
      | select(gpu_count == 0)] | length) == 1
    and ([.items[]
      | select(.metadata.labels["glm52.nvidia.com/role"] == "worker")
      | select(gpu_count == $tp_size)
      | select((argv_text | contains($model_revision))
        and (argv_text | contains($context_flag)))] | length) == 1
  else
    (.items | all(.[];
      (gpu_count == $tp_size)
      and (argv_text | contains($model_revision))
      and (argv_text | contains($context_flag))))
  end)
' "${out_dir}/pods.json" >/dev/null; then
  echo "Runtime pods do not match the pinned topology, image, revision, context, readiness, or GPU identity" >&2
  exit 1
fi
kubectl get node "${NODE_NAME}" -o json | jq '
  {
    name: .metadata.name,
    uid: .metadata.uid,
    labels: {
      architecture: .metadata.labels["kubernetes.io/arch"],
      operating_system: .metadata.labels["kubernetes.io/os"],
      instance_type: .metadata.labels["node.kubernetes.io/instance-type"]
    },
    node_info: .status.nodeInfo,
    capacity: {
      cpu: .status.capacity.cpu,
      memory: .status.capacity.memory,
      gpu: .status.capacity["nvidia.com/gpu"]
    },
    allocatable: {
      cpu: .status.allocatable.cpu,
      memory: .status.allocatable.memory,
      gpu: .status.allocatable["nvidia.com/gpu"]
    }
  }
' > "${out_dir}/node-identity.json"
if [[ "${is_dynamo}" == true ]]; then
  {
    kubectl get pods -n dynamo-system -o json
    kubectl get pods -n grove -o json
  } | jq -s '
    [.[].items[]
      | select(
          (.metadata.namespace == "dynamo-system"
            and (.metadata.name | contains("operator-controller-manager")))
          or (.metadata.namespace == "grove"
            and (.metadata.name | startswith("grove-operator-"))))
      | {
          namespace: .metadata.namespace,
          name: .metadata.name,
          uid: .metadata.uid,
          phase: .status.phase,
          containers: [.spec.containers[] as $container | {
            name: $container.name,
            image: $container.image,
            image_id: ([.status.containerStatuses[]?
              | select(.name == $container.name) | .imageID][0] // null),
            ready: ([.status.containerStatuses[]?
              | select(.name == $container.name) | .ready][0] // false)
          }]
        }]
  ' > "${out_dir}/control-plane.json"
  jq -e '
    length == 2
    and ([.[].namespace] | sort == ["dynamo-system", "grove"])
    and all(.[];
      .phase == "Running"
      and (.containers | length > 0)
      and all(.containers[];
        .ready == true
        and (.image_id | type == "string")
        and (.image_id | test("sha256:[0-9a-f]{64}$"))))
  ' "${out_dir}/control-plane.json" >/dev/null
  kubectl get crd \
    dynamographdeployments.nvidia.com \
    podcliques.grove.io \
    podcliquesets.grove.io \
    podgangs.scheduler.grove.io \
    -o json > "${out_dir}/control-plane-crds.json"
fi
kubectl get all -n "${NAMESPACE}" -l "${selector}" -o yaml > "${out_dir}/child-resources.yaml"
mkdir -p "${out_dir}/logs" "${out_dir}/pods"

pod_names="$(kubectl get pods -n "${NAMESPACE}" \
  -l "${selector}" \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}')"

: > "${out_dir}/events.txt"
while IFS= read -r pod; do
  [[ -n "${pod}" ]] || continue
  printf 'pod=%s\n' "${pod}" >> "${out_dir}/events.txt"
  kubectl get events -n "${NAMESPACE}" \
    --field-selector "involvedObject.kind=Pod,involvedObject.name=${pod}" \
    --sort-by=.lastTimestamp >> "${out_dir}/events.txt" || true
done <<< "${pod_names}"

while IFS= read -r pod; do
  [[ -n "${pod}" ]] || continue
  kubectl logs "${pod}" -n "${NAMESPACE}" --all-containers \
    > "${out_dir}/logs/${pod}.log" 2>&1 || true
  kubectl exec "${pod}" -n "${NAMESPACE}" -c main -- python3 -c '
import importlib.metadata as metadata
import json

packages = ("ai-dynamo", "vllm", "sglang", "transformers", "torch")
versions = {}
for package in packages:
    try:
        versions[package] = metadata.version(package)
    except metadata.PackageNotFoundError:
        versions[package] = None
print(json.dumps(versions, sort_keys=True))
' > "${out_dir}/pods/${pod}-packages.json" 2>&1
  kubectl exec "${pod}" -n "${NAMESPACE}" -c main -- \
    env GLM52_MODEL_PATH="${MODEL_PATH}" GLM52_MODEL_ID="${MODEL_ID}" \
      GLM52_MODEL_REVISION="${MODEL_REVISION}" python3 -c '
import hashlib
import json
import os
import re
from pathlib import Path

root = Path(os.environ["GLM52_MODEL_PATH"])
metadata_root = root / ".cache" / "huggingface" / "download"
if not metadata_root.is_dir():
    raise RuntimeError(f"missing Hugging Face metadata directory: {metadata_root}")
files = {}
revisions = set()
for metadata_path in sorted(metadata_root.glob("*.metadata")):
    lines = metadata_path.read_text().splitlines()
    if len(lines) < 2:
        raise RuntimeError(f"malformed Hugging Face metadata: {metadata_path}")
    revision, etag = lines[:2]
    revisions.add(revision)
    name = metadata_path.name.removesuffix(".metadata")
    target = root / name
    if not target.is_file():
        raise RuntimeError(f"metadata target is missing: {target}")
    record = {
        "etag": etag,
        "size_bytes": target.stat().st_size,
    }
    if not name.endswith(".safetensors"):
        record["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    files[name] = record

expected_revision = os.environ["GLM52_MODEL_REVISION"]
if revisions != {expected_revision}:
    raise RuntimeError(
        f"checkpoint revision mismatch: expected {expected_revision}, found {sorted(revisions)}"
    )
required_files = {
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "hf_quant_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
}
missing_required = sorted(required_files - files.keys())
if missing_required:
    raise RuntimeError(f"checkpoint manifest is missing required files: {missing_required}")
weight_shards = {
    name: record for name, record in files.items() if name.endswith(".safetensors")
}
index = json.loads((root / "model.safetensors.index.json").read_text())
weight_map = index.get("weight_map")
if not isinstance(weight_map, dict) or not weight_map:
    raise RuntimeError("checkpoint index has no non-empty weight_map")
index_shards = set(weight_map.values())
metadata_shards = set(weight_shards)
invalid_etags = sorted(
    name for name, record in weight_shards.items()
    if re.fullmatch(r"[0-9a-f]{64}", record["etag"]) is None
)
empty_shards = sorted(
    name for name, record in weight_shards.items() if record["size_bytes"] <= 0
)
if (
    len(weight_shards) != 47
    or index_shards != metadata_shards
    or invalid_etags
    or empty_shards
):
    raise RuntimeError(
        "checkpoint shard mismatch: "
        f"metadata={len(metadata_shards)}, index={len(index_shards)}, "
        f"missing_metadata={sorted(index_shards - metadata_shards)}, "
        f"not_in_index={sorted(metadata_shards - index_shards)}, "
        f"invalid_etags={invalid_etags}, empty={empty_shards}"
    )

print(json.dumps({
    "schema_version": 1,
    "model_id": os.environ["GLM52_MODEL_ID"],
    "huggingface_revision": expected_revision,
    "metadata_revisions": sorted(revisions),
    "weight_shard_count": len(weight_shards),
    "index_weight_shards": sorted(index_shards),
    "files": files,
}, indent=2, sort_keys=True))
' > "${out_dir}/pods/${pod}-model-manifest.json"
  gpu_request="$(jq -r --arg pod "${pod}" '
    [.items[] | select(.metadata.name == $pod)
      | .spec.containers[]?.resources.requests["nvidia.com/gpu"] // "0"
      | tonumber] | add // 0
  ' "${out_dir}/pods.json")"
  if ((gpu_request > 0)); then
    kubectl exec "${pod}" -n "${NAMESPACE}" -c main -- \
      nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu,compute_mode \
        --format=csv,noheader,nounits \
      > "${out_dir}/pods/${pod}-gpu.csv" 2>&1
    kubectl exec "${pod}" -n "${NAMESPACE}" -c main -- \
      nvidia-smi -q > "${out_dir}/pods/${pod}-gpu-details.txt" 2>&1
  fi
done <<< "${pod_names}"

case "${variant}" in
  dynamo-vllm|dynamo-sglang)
    kubectl get dynamographdeployment "glm52-${variant}" -n "${NAMESPACE}" -o yaml \
      > "${out_dir}/applied.yaml"
    kubectl get service "glm52-${variant}-frontend" -n "${NAMESPACE}" -o yaml \
      > "${out_dir}/service.yaml"
    kubectl get endpointslice -n "${NAMESPACE}" \
      -l "kubernetes.io/service-name=glm52-${variant}-frontend" -o yaml \
      > "${out_dir}/endpoint-slices.yaml"
    ;;
  vllm-serve|sglang-serve)
    kubectl get deployment "glm52-${variant}" -n "${NAMESPACE}" -o yaml \
      > "${out_dir}/applied.yaml"
    kubectl get service "glm52-${variant}" -n "${NAMESPACE}" -o yaml \
      > "${out_dir}/service.yaml"
    kubectl get endpointslice -n "${NAMESPACE}" \
      -l "kubernetes.io/service-name=glm52-${variant}" -o yaml \
      > "${out_dir}/endpoint-slices.yaml"
    ;;
esac

kubectl get pods -n "${NAMESPACE}" -l "${selector}" -o json \
  | jq '{captured_at: now | todate,
         pods: [.items[] as $pod | {
           name: $pod.metadata.name,
           node: $pod.spec.nodeName,
           phase: $pod.status.phase,
           containers: [$pod.spec.containers[] as $container | {
             name: $container.name,
             image: $container.image,
             command: $container.command,
             args: $container.args,
             image_id: ([$pod.status.containerStatuses[]?
                         | select(.name == $container.name)
                         | .imageID][0] // null),
             container_id: ([$pod.status.containerStatuses[]?
                             | select(.name == $container.name)
                             | .containerID][0] // null),
             restart_count: ([$pod.status.containerStatuses[]?
                              | select(.name == $container.name)
                              | .restartCount][0] // null),
             gpu_request: ($container.resources.requests["nvidia.com/gpu"] // "0")
           }]
         }]}' > "${out_dir}/runtime-identity.json"

GLM52_CAPTURE_DIR="${out_dir}" \
GLM52_VARIANT="${variant}" \
GLM52_CAMPAIGN_PHASE="${campaign_phase}" \
GLM52_IMAGE="${expected_image}" \
GLM52_ENDPOINT="${endpoint}" \
GLM52_SERVICE_NAME="${service_name}" \
GLM52_CONTROLLER_KIND="${controller_kind}" \
GLM52_CONTROLLER_NAME="${controller_name}" \
GLM52_SOURCE_COMMIT="${source_commit}" \
GLM52_TEMPLATE_PATH="${SCRIPT_DIR}/templates/${variant}.yaml" \
GLM52_RENDERED_MANIFEST_PATH="${SCRIPT_DIR}/rendered/${variant}.yaml" \
GLM52_MODEL_ID="${MODEL_ID}" \
GLM52_MODEL_REVISION="${MODEL_REVISION}" \
GLM52_SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
GLM52_MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
GLM52_EXPECTED_GPU_UUIDS="${EXPECTED_GPU_UUIDS}" \
python3 - <<'PY'
import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


root = Path(os.environ["GLM52_CAPTURE_DIR"])
tree_digest = hashlib.sha256()
for path in sorted(item for item in root.rglob("*") if item.is_file()):
    relative = path.relative_to(root).as_posix()
    tree_digest.update(relative.encode())
    tree_digest.update(b"\0")
    tree_digest.update(sha256(path).encode())
    tree_digest.update(b"\n")

controller = json.loads((root / "controller.json").read_text())
pods_document = json.loads((root / "pods.json").read_text())
pods = {}
worker_pod_name = None
for pod in sorted(pods_document["items"], key=lambda item: item["metadata"]["name"]):
    containers = pod["spec"]["containers"]
    if len(containers) != 1:
        raise RuntimeError(f"expected one serving container in {pod['metadata']['name']}")
    container = containers[0]
    statuses = {
        status["name"]: status for status in pod["status"].get("containerStatuses", [])
    }
    status = statuses.get(container["name"])
    if status is None or not status.get("imageID") or not status.get("containerID"):
        raise RuntimeError(
            f"missing runtime image/container ID for {pod['metadata']['name']}"
        )
    argv = (container.get("command") or []) + (container.get("args") or [])
    manifest_path = root / "pods" / f"{pod['metadata']['name']}-model-manifest.json"
    role = pod["metadata"].get("labels", {}).get(
        "glm52.nvidia.com/role", "worker"
    )
    if role in pods:
        raise RuntimeError(f"duplicate serving role in runtime binding: {role}")
    if role == "worker":
        worker_pod_name = pod["metadata"]["name"]
    pods[role] = {
        "name_sha256": sha256_text(pod["metadata"]["name"]),
        "uid_sha256": sha256_text(pod["metadata"]["uid"]),
        "node_name_sha256": sha256_text(pod["spec"]["nodeName"]),
        "image_id": status["imageID"],
        "argv_sha256": hashlib.sha256("\0".join(argv).encode()).hexdigest(),
        "model_manifest_sha256": sha256(manifest_path),
    }

if worker_pod_name is None:
    raise RuntimeError("runtime binding has no worker pod")
gpu_rows = list(csv.reader((root / "pods" / f"{worker_pod_name}-gpu.csv").open()))
if not gpu_rows:
    raise RuntimeError("runtime binding has no GPU rows")
gpu_rows = [[field.strip() for field in row] for row in gpu_rows]
if any(len(row) != 8 for row in gpu_rows):
    raise RuntimeError("unexpected nvidia-smi CSV shape")
gpu_uuids = sorted(row[1] for row in gpu_rows)
expected_gpu_uuids = sorted(
    value.strip()
    for value in os.environ["GLM52_EXPECTED_GPU_UUIDS"].split(",")
    if value.strip()
)
gpu_models = sorted({row[2] for row in gpu_rows})
driver_versions = sorted({row[3] for row in gpu_rows})
memory_totals = sorted({int(row[4]) for row in gpu_rows})
if len(gpu_rows) != 4 or len(gpu_models) != 1 or len(driver_versions) != 1:
    raise RuntimeError("GPU hardware identity is not homogeneous TP4")
if (
    len(expected_gpu_uuids) != 4
    or len(set(expected_gpu_uuids)) != 4
    or gpu_uuids != expected_gpu_uuids
):
    raise RuntimeError("GPU UUID set differs from the pinned campaign allocation")
node_identity = json.loads((root / "node-identity.json").read_text())

control_plane = None
control_plane_path = root / "control-plane.json"
if control_plane_path.is_file():
    entries = json.loads(control_plane_path.read_text())
    control_plane = {}
    for entry in entries:
        key = {
            "dynamo-system": "dynamo_operator_image_digests",
            "grove": "grove_operator_image_digests",
        }[entry["namespace"]]
        digests = set()
        for container in entry["containers"]:
            match = re.search(r"sha256:[0-9a-f]{64}$", container["image_id"])
            if match is None:
                raise RuntimeError("control-plane image ID has no digest")
            digests.add(match.group())
        digests = sorted(digests)
        control_plane[key] = digests

binding = {
    "schema_version": 1,
    "variant": os.environ["GLM52_VARIANT"],
    "campaign_phase": os.environ["GLM52_CAMPAIGN_PHASE"],
    "served_model_name": os.environ["GLM52_SERVED_MODEL_NAME"],
    "model_id": os.environ["GLM52_MODEL_ID"],
    "model_revision": os.environ["GLM52_MODEL_REVISION"],
    "max_model_len": int(os.environ["GLM52_MAX_MODEL_LEN"]),
    "image": os.environ["GLM52_IMAGE"],
    "endpoint": os.environ["GLM52_ENDPOINT"],
    "service_name": os.environ["GLM52_SERVICE_NAME"],
    "controller": {
        "kind": os.environ["GLM52_CONTROLLER_KIND"],
        "name": os.environ["GLM52_CONTROLLER_NAME"],
        "uid_sha256": sha256_text(controller["metadata"]["uid"]),
        "generation": controller["metadata"]["generation"],
    },
    "pods": pods,
    "recipe": {
        "source_commit": os.environ["GLM52_SOURCE_COMMIT"],
        "template_sha256": sha256(Path(os.environ["GLM52_TEMPLATE_PATH"])),
        "rendered_manifest_sha256": sha256(
            Path(os.environ["GLM52_RENDERED_MANIFEST_PATH"])
        ),
    },
    "hardware": {
        "gpu_count": len(gpu_rows),
        "gpu_model": gpu_models[0],
        "gpu_uuid_set_sha256": sha256_text("\n".join(gpu_uuids)),
        "driver_version": driver_versions[0],
        "gpu_memory_total_mib": memory_totals,
        "kernel_version": node_identity["node_info"]["kernelVersion"],
        "kubelet_version": node_identity["node_info"]["kubeletVersion"],
        "container_runtime_version": node_identity["node_info"][
            "containerRuntimeVersion"
        ],
    },
    "control_plane": control_plane,
    "capture": {
        "sha256": tree_digest.hexdigest(),
        "captured_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    },
}
(root / "runtime-binding.json").write_text(
    json.dumps(binding, indent=2, sort_keys=True) + "\n"
)
PY
python3 "${ROOT_DIR}/eval/runtime_binding.py" \
  "${out_dir}/runtime-binding.json" \
  --variant "${variant}" \
  --phase "${campaign_phase}" \
  --endpoint "${endpoint}" >/dev/null

mv "${out_dir}" "${final_dir}"
trap - EXIT

runner_pod="${EVAL_RUNNER_POD:-glm52-eval-runner}"
runner_ready="$(kubectl get pod "${runner_pod}" -n "${NAMESPACE}" \
  -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || true)"
if [[ "${runner_ready}" != True ]]; then
  echo "Evaluation runner ${NAMESPACE}/${runner_pod} is not ready for binding publication" >&2
  exit 1
fi
remote_binding_dir="/artifacts/glm52-nscale/runtime-bindings/${variant}"
kubectl exec "${runner_pod}" -n "${NAMESPACE}" -c runner -- \
  mkdir -p "${remote_binding_dir}"
kubectl exec -i "${runner_pod}" -n "${NAMESPACE}" -c runner -- \
  /bin/bash -eu -c '
    temporary="$1/.active.json.tmp"
    cat >"${temporary}"
    python3 -m json.tool "${temporary}" >/dev/null
    mv "${temporary}" "$1/active.json"
  ' -- "${remote_binding_dir}" < "${final_dir}/runtime-binding.json"

echo "Captured runtime identity under ${final_dir} and published ${remote_binding_dir}/active.json"
