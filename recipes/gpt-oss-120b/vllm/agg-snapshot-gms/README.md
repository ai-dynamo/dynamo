<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GPT-OSS-120B vLLM Snapshot and GMS

This experimental recipe creates a checkpoint before starting a single-GPU
vLLM worker. Dynamo Snapshot restores the worker process while GPU Memory
Service (GMS) reloads model weights from shared storage.

The manifest uses an immutable Dynamo main-branch nightly. Build the Snapshot
agent and placeholder from the same checkout so their checkpoint tooling
matches the runtime.

## Prerequisites

- Dynamo Platform built from a commit that contains the manifest's nightly
  commit, with checkpointing and the `gmsSnapshot` feature gate enabled.
- Kubernetes 1.34 or later with stable Dynamic Resource Allocation (DRA), a
  healthy NVIDIA GPU DRA driver, and the `gpu.nvidia.com` `DeviceClass`. See
  the [NVIDIA GPU DRA driver installation guide](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/dra-intro-install.html).
- One x86_64 B200 node with the NVIDIA container runtime and an NVIDIA R610 or
  newer driver. See
  [Cold Start Optimizations and Resiliency Support Matrix](../../../../docs/kubernetes/cold-start-and-resiliency.md)
  and [Snapshotting GPU Workers](../../../../docs/kubernetes/snapshot.md).
- A filesystem-mode, ReadWriteMany (RWX) storage class for the model and
  checkpoint PVCs.
- A container registry available to the GPU nodes.
- Docker, Helm, `kubectl`, and `jq`.

> [!WARNING]
> Snapshot is experimental. Its privileged DaemonSet uses the host PID, IPC,
> and network namespaces and performs CRIU and CUDA checkpoint operations.
> Review the chart and security policy before installing it.

Verify DRA before continuing:

```bash
export KUBE_CONTEXT=your-kubernetes-context

kubectl --context "${KUBE_CONTEXT}" wait --for=condition=Ready pod --all \
  -n nvidia-dra-driver-gpu --timeout=300s
kubectl --context "${KUBE_CONTEXT}" get deviceclass gpu.nvidia.com
kubectl --context "${KUBE_CONTEXT}" get resourceslices -o wide
```

At least one `ResourceSlice` from the `gpu.nvidia.com` driver must advertise the
target B200 node.

## Configure Dynamo Platform

Configure Dynamo Platform with:

```yaml
dynamo-operator:
  featureGates:
    gmsSnapshot: true
  checkpoint:
    enabled: true
    storage:
      type: pvc
      pvc:
        pvcName: snapshot-pvc
        basePath: /checkpoints
```

This configuration requires an existing `snapshot-pvc` in the workload
namespace.

## Create Shared Storage

Create the workload namespace and checkpoint PVC. Select an RWX-capable storage
class available in your environment:

```bash
export NAMESPACE=dynamo-demo
export CHECKPOINT_STORAGE_CLASS=your-rwx-storage-class

kubectl --context "${KUBE_CONTEXT}" create namespace "${NAMESPACE}" \
  --dry-run=client -o yaml |
  kubectl --context "${KUBE_CONTEXT}" apply -f -
cat <<EOF | kubectl --context "${KUBE_CONTEXT}" apply -n "${NAMESPACE}" -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: snapshot-pvc
spec:
  accessModes:
    - ReadWriteMany
  volumeMode: Filesystem
  storageClassName: ${CHECKPOINT_STORAGE_CLASS}
  resources:
    requests:
      storage: 1Ti
EOF
```

The operator mounts this PVC as `checkpoint-storage` in the checkpoint Job. The
manifest declares the same volume on the restored worker so `gms-loader` can
read the saved weights.

## Build the Snapshot Images

The main nightly does not publish a matching Snapshot agent or placeholder
image. Build both from this checkout, push them to your registry, and update
`deploy.yaml` with `PLACEHOLDER_IMAGE`:

```bash
export RUNTIME_IMAGE=nvcr.io/nvidia/ai-dynamo/vllm-runtime-nightly:20260717-5930e58
export SNAPSHOT_AGENT_IMAGE=registry.example.com/dynamo/snapshot-agent:20260717
export PLACEHOLDER_IMAGE=registry.example.com/dynamo/vllm-placeholder:20260717-5930e58

make -C deploy/snapshot docker-build-agent \
  IMG="${SNAPSHOT_AGENT_IMAGE}"
make -C deploy/snapshot docker-push-agent \
  IMG="${SNAPSHOT_AGENT_IMAGE}"
make -C deploy/snapshot docker-build-placeholder \
  PLACEHOLDER_BASE_IMG="${RUNTIME_IMAGE}" \
  PLACEHOLDER_IMG="${PLACEHOLDER_IMAGE}"
make -C deploy/snapshot docker-push-placeholder \
  PLACEHOLDER_IMG="${PLACEHOLDER_IMAGE}"
```

The placeholder retains the matching Dynamo, vLLM, and GMS runtime and adds the
checkpoint-and-restore tools. Use immutable tags for both derived images.

## Install the Snapshot Agent

Install the local Snapshot chart once in an infrastructure namespace. This
`podMount` configuration lets the operator mount each workload namespace's PVC
into its checkpoint and restore pods:

```bash
export SNAPSHOT_NAMESPACE=dynamo-snapshot
export SNAPSHOT_AGENT_REPOSITORY="${SNAPSHOT_AGENT_IMAGE%:*}"
export SNAPSHOT_AGENT_TAG="${SNAPSHOT_AGENT_IMAGE##*:}"

helm --kube-context "${KUBE_CONTEXT}" upgrade --install snapshot \
  ./deploy/helm/charts/snapshot \
  --namespace "${SNAPSHOT_NAMESPACE}" \
  --create-namespace \
  --set storage.accessMode=podMount \
  --set storage.pvc.create=false \
  --set rbac.namespaceRestricted=false \
  --set daemonset.image.repository="${SNAPSHOT_AGENT_REPOSITORY}" \
  --set daemonset.image.tag="${SNAPSHOT_AGENT_TAG}" \
  --set-json 'daemonset.imagePullSecrets=[]'

kubectl --context "${KUBE_CONTEXT}" rollout status daemonset/snapshot-agent \
  -n "${SNAPSHOT_NAMESPACE}" --timeout=300s
kubectl --context "${KUBE_CONTEXT}" get pods \
  -n "${SNAPSHOT_NAMESPACE}" \
  -l app.kubernetes.io/component=snapshot-agent -o wide
```

If the registry needs authentication, create its pull secret in
`${SNAPSHOT_NAMESPACE}` and configure `daemonset.imagePullSecrets` instead of
clearing the chart default.

## Download the Model

From the repository root, set `storageClassName` in
`recipes/gpt-oss-120b/model-cache/model-cache.yaml` to an RWX class and increase
the requested size if your storage backend needs additional headroom. Then
create the token secret and download the model:

```bash
export HF_TOKEN=your-hugging-face-token

kubectl --context "${KUBE_CONTEXT}" create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="${HF_TOKEN}" \
  -n "${NAMESPACE}" \
  --dry-run=client -o yaml |
  kubectl --context "${KUBE_CONTEXT}" apply -f -
kubectl --context "${KUBE_CONTEXT}" apply \
  -f recipes/gpt-oss-120b/model-cache/ -n "${NAMESPACE}"
kubectl --context "${KUBE_CONTEXT}" wait --for=condition=Complete \
  job/model-download -n "${NAMESPACE}" --timeout=6000s
```

## Create the Checkpoint

Apply the manifest with the worker replica count unchanged at zero:

```bash
kubectl --context "${KUBE_CONTEXT}" apply \
  -f recipes/gpt-oss-120b/vllm/agg-snapshot-gms/deploy.yaml \
  -n "${NAMESPACE}"
```

The two-phase flow is intentional: first wait for the GMS saver Job and
checkpoint to succeed, then manually scale the worker to one replica.
`WaitForCheckpoint` gates an early scale-up request; it does not automatically
change the committed `replicas: 0`.

Capture `status.jobName` as soon as the operator publishes it. Then wait for the
Job, verify the saver log, and wait for the `DynamoCheckpoint` to become
`Ready`:

```bash
set -euo pipefail

DGD_NAME=gpt-oss-120b-vllm-snapshot-gms
WORKER_COMPONENT=VllmWorker

wait_for_field() {
  local resource="$1"
  local jsonpath="$2"
  local value
  local deadline=$((SECONDS + 600))

  while (( SECONDS < deadline )); do
    value="$(kubectl --context "${KUBE_CONTEXT}" get "${resource}" \
      -n "${NAMESPACE}" -o "jsonpath=${jsonpath}" 2>/dev/null || true)"
    if [[ -n "${value}" ]]; then
      printf '%s\n' "${value}"
      return 0
    fi
    sleep 2
  done

  printf 'timed out waiting for %s field %s\n' "${resource}" "${jsonpath}" >&2
  return 1
}

CHECKPOINT_NAME="$(wait_for_field "dgd/${DGD_NAME}" \
  "{.status.checkpoints.${WORKER_COMPONENT}.checkpointName}")"
CHECKPOINT_JOB="$(wait_for_field "dckpt/${CHECKPOINT_NAME}" \
  '{.status.jobName}')"

kubectl --context "${KUBE_CONTEXT}" wait \
  --for=condition=Complete "job/${CHECKPOINT_JOB}" \
  -n "${NAMESPACE}" --timeout=3600s

GMS_SAVER_LOG="$(mktemp)"
trap 'rm -f "${GMS_SAVER_LOG}"' EXIT
kubectl --context "${KUBE_CONTEXT}" logs "job/${CHECKPOINT_JOB}" \
  -c gms-saver -n "${NAMESPACE}" | tee "${GMS_SAVER_LOG}"
grep -Fq "Save complete; exiting" "${GMS_SAVER_LOG}"

kubectl --context "${KUBE_CONTEXT}" wait \
  --for=jsonpath='{.status.phase}'=Ready \
  "dckpt/${CHECKPOINT_NAME}" -n "${NAMESPACE}" --timeout=3600s
```

Do not scale the worker unless every command succeeds.

## Restore the Worker

Find the component index by name and construct the JSON patch with `jq`. This
avoids relying on component order:

```bash
set -euo pipefail

WORKER_INDEX="$(
  kubectl --context "${KUBE_CONTEXT}" get "dgd/${DGD_NAME}" \
    -n "${NAMESPACE}" -o json |
    jq -er --arg component "${WORKER_COMPONENT}" '
      .spec.components
      | to_entries
      | map(select(.value.name == $component))
      | if length == 1 then .[0].key
        else error("expected exactly one matching worker component")
        end
    '
)"
SCALE_PATCH="$(
  jq -cn --arg index "${WORKER_INDEX}" \
    '[{op: "replace",
       path: ("/spec/components/" + $index + "/replicas"),
       value: 1}]'
)"

kubectl --context "${KUBE_CONTEXT}" patch "dgd/${DGD_NAME}" \
  -n "${NAMESPACE}" --type=json -p "${SCALE_PATCH}"
kubectl --context "${KUBE_CONTEXT}" get pods -n "${NAMESPACE}" \
  -l nvidia.com/dynamo-graph-deployment-name="${DGD_NAME}" -w
```

The operator creates the `ResourceClaimTemplate` and pod-owned
`ResourceClaim`. Do not create either resource manually:

```bash
kubectl --context "${KUBE_CONTEXT}" get \
  resourceclaimtemplates,resourceclaims -n "${NAMESPACE}"
```

## Smoke Test

After the frontend and restored worker are ready, keep the port-forward running
in one terminal:

```bash
kubectl --context "${KUBE_CONTEXT}" port-forward \
  svc/gpt-oss-120b-vllm-snapshot-gms-frontend 8000:8000 \
  -n "${NAMESPACE}"
```

Send a request from a second terminal:

```bash
curl --fail-with-body http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128
  }'
```

## Validation Results

The recipe was validated on one B200 with the runtime image pinned in the
manifest and Snapshot images built from the same main-branch checkout:

- GMS saved 69.47 GiB across 19 shards in 84.31 seconds.
- The Snapshot agent completed process checkpointing in 43.17 seconds.
- Process restore completed in 4.15 seconds.
- GMS reloaded 69.47 GiB in 7.20 seconds.
- The worker became Ready 13 seconds after scaling from zero to one.
- A chat-completions request returned HTTP 200 after restore.

These values are a functional baseline, not a performance guarantee. Storage,
node, driver, and cluster conditions affect timings.

## Artifact Layout and Cleanup

The Snapshot artifact uses
`/checkpoints/${CHECKPOINT_ID}/versions/${ARTIFACT_VERSION}`. The saver and
loader use the sibling path
`/checkpoints/${CHECKPOINT_ID}/gms/versions/${ARTIFACT_VERSION}`. Keeping both
artifacts below `/checkpoints/${CHECKPOINT_ID}` avoids a collision with the
CRIU version directory and lets `deletionPolicy: Delete` remove both trees when
the deployment is deleted.

The GMS `nixl` transfer backend stages data through the POSIX filesystem mounted
at `/checkpoints`; `posix` is not a transfer-backend enum value.

Delete the deployment when the test is complete:

```bash
kubectl --context "${KUBE_CONTEXT}" delete \
  "dgd/${DGD_NAME}" -n "${NAMESPACE}"
```
