---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Snapshotting GPU Workers
subtitle: Checkpoints initialized GPU workers with CRIU and cuda-checkpoint so later pods warm-start in seconds instead of minutes.
---

> [!WARNING]
> Snapshot is experimental and might work only in some cluster configurations.
> The `snapshot-agent` DaemonSet runs in privileged mode to perform CRIU
> operations. See [Limitations](#limitations) for details.

**NVIDIA Dynamo Snapshot** is infrastructure for fast-starting GPU applications
in Kubernetes using Checkpoint/Restore in Userspace (CRIU) and NVIDIA's
`cuda-checkpoint` utility. The usual flow is:

1. start a worker once and checkpoint its initialized state
2. store that checkpoint on the snapshot agent's shared PVC
3. restore later workers from that checkpoint instead of cold-starting again

Every agent in a snapshot deployment mounts the same ReadWriteMany (RWX) claim.
Checkpoint and restore workload pods do not mount that claim. During restore,
the agent temporarily exposes only the selected artifact directory at
`/tmp/checkpoint` inside the target container's mount namespace.

| Startup Type | Time | What Happens |
|--------------|------|--------------|
| **Cold Start** | ~1 min | Download model, load to GPU, initialize engine |
| **Warm Start** (restore from checkpoint) | ~10 sec | Restore from a ready checkpoint directory |

> [!NOTE]
> Restore time depends on storage bandwidth, GPU model, and whether the restore
> stays on the same node.

For more background on the snapshot architecture and startup improvements, see
[NVIDIA Dynamo Snapshot: Fast Startup for Inference Workloads on Kubernetes](https://developer.nvidia.com/blog/nvidia-dynamo-snapshot-fast-startup-for-inference-workloads-on-kubernetes/).

## Prerequisites

- x86_64 (`amd64`) GPU nodes
- NVIDIA driver 580.xx or newer on the target GPU nodes (590.xx or newer if testing multi-GPU snapshots)
- vLLM or SGLang backend today; TensorRT-LLM is supported only for the
  experimental single-GPU aggregated text worker path.
- Checkpoint storage that supports RWX access and concurrent visibility from
  every node eligible to run the snapshot agent.
- **CRI-O / OpenShift:** set `runtime.type=crio` on the snapshot chart (and `openshift.enabled=true` on OpenShift). Defaults are for containerd; see the chart README for sockets and Helm flags.

## Backend and Topology Support

| Backend | Single GPU | Multi-GPU, Single Node | Multinode |
| :--- | :---: | :---: | :---: |
| **vLLM** | Supported | Highly experimental | Work in progress |
| **SGLang** | Supported | Work in progress | Work in progress |
| **TensorRT-LLM** | Experimental | Work in progress | Work in progress |

- TensorRT-LLM support is limited to the experimental single-GPU aggregated text-worker path.
- Snapshot with GMS is not a supported production path and is disabled in normal deployments.
  Experimental testing requires the internal GMS Snapshot feature gate and CUDA Driver r610 or
  later.
- Multi-GPU support has limited validation and currently uses legacy IPC only for peer-to-peer
  communication.

For the cross-feature backend overview, see [Compatibility](../../../../reference/general/compatibility.mdx).

## Quick Start via `DynamoCheckpoint` CR

1. Build a placeholder image
2. Enable checkpointing in the platform
3. Install the snapshot chart
4. Create a `DynamoCheckpoint`
5. Wait for the checkpoint to become ready
6. Deploy a `DynamoGraphDeployment` that restores from its `checkpointRef`

### 1. Build and push a placeholder image

Snapshot-enabled workers use a placeholder image based on the normal runtime
image. It enters restore standby while the agent mounts the required binaries
temporarily into the container. If you do not already have one, build it and
push it to a registry your cluster can pull from:

```bash
export RUNTIME_IMAGE=registry.example.com/dynamo/vllm-runtime:1.0.0
export PLACEHOLDER_IMAGE=registry.example.com/dynamo/vllm-placeholder:1.0.0

cd deploy/snapshot

make docker-build-placeholder \
  PLACEHOLDER_BASE_IMG="${RUNTIME_IMAGE}" \
  PLACEHOLDER_IMG="${PLACEHOLDER_IMAGE}"

make docker-push-placeholder \
  PLACEHOLDER_IMG="${PLACEHOLDER_IMAGE}"
```

The placeholder image preserves the normal runtime entrypoint and command
contract. It does not contain CRIU, `cuda-checkpoint`, `nsrestore`, or the
checkpoint PVC. The agent supplies the executable bundle only while restore is
running.

To build the snapshot agent against a custom CRIU fork or ref, pass `CRIU_REPO`
and `CRIU_REF` through `make`. If they are unset, the Dockerfile defaults are
used.

```bash
make docker-build-agent \
  IMG=registry.example.com/dynamo/snapshot-agent:1.0.0 \
  CRIU_REPO="${YOUR_CRIU_REPO}" \
  CRIU_REF="branch-or-sha"
```

### 2. Enable checkpointing in the platform and verify it

Whether you are installing or upgrading `dynamo-platform`, the operator only needs checkpointing enabled:

```yaml
dynamo-operator:
  checkpoint:
    enabled: true
```

If the platform is already installed, verify that the operator config contains the checkpoint block:

```bash
OPERATOR_CONFIG=$(kubectl get deploy -n "${PLATFORM_NAMESPACE}" \
  -l app.kubernetes.io/name=dynamo-operator,app.kubernetes.io/component=manager \
  -o jsonpath='{.items[0].spec.template.spec.volumes[?(@.name=="operator-config")].configMap.name}')

kubectl get configmap "${OPERATOR_CONFIG}" -n "${PLATFORM_NAMESPACE}" \
  -o jsonpath='{.data.config\.yaml}' | sed -n '/^checkpoint:/,/^[^[:space:]]/p'
```

Verify that the rendered config includes `enabled: true`.

### 3. Install the snapshot chart

For the default namespace-restricted mode, install the snapshot chart in each
workload namespace. The chart creates the shared RWX PVC and agent DaemonSet in
that namespace:

```bash
helm upgrade --install snapshot ./deploy/helm/charts/snapshot \
  --namespace ${NAMESPACE} \
  --create-namespace \
  --set storage.pvc.create=true
```

Every agent pod mounts the claim read-write at `storage.pvc.basePath`. If your
cluster does not have a default storage class, also set
`storage.pvc.storageClass`. The selected class must support RWX.

To reuse an existing claim, set `storage.pvc.create=false` and
`storage.pvc.name=<claim-name>`. The existing claim must be in the chart release
namespace, declare `ReadWriteMany`, and be mountable from every eligible node.

For CRI-O, set `runtime.type=crio`. On OpenShift, also set
`openshift.enabled=true`. See the
[snapshot chart README](https://github.com/ai-dynamo/dynamo/blob/main/deploy/helm/charts/snapshot/README.md).

To run one cluster-wide agent fleet, install the chart once in an infrastructure
namespace and disable namespace restriction:

```bash
helm upgrade --install snapshot ./deploy/helm/charts/snapshot \
  --namespace dynamo-system \
  --create-namespace \
  --set storage.pvc.create=true \
  --set rbac.namespaceRestricted=false
```

The cluster-wide release still owns one PVC in `dynamo-system`. Every agent in
the fleet mounts that claim; workload namespaces do not need checkpoint PVCs.
The operator has no checkpoint storage configuration in either topology.

Verify that the DaemonSet is ready and the claim is bound:

```bash
kubectl rollout status daemonset/snapshot-agent -n dynamo-system
kubectl get pods -n dynamo-system -l app.kubernetes.io/component=snapshot-agent -o wide
kubectl get pvc snapshot-pvc -n dynamo-system
```

### 4. Create a standalone `DynamoCheckpoint`

> [!WARNING]
> `checkpoint.mode` is deprecated. Use `checkpoint.enabled` and omit `mode`.
> `spec.identity` is legacy v1alpha1 compatibility only; use `checkpointRef`
> to restore an existing checkpoint.

The checkpoint Job pod template should match the worker container you want to
checkpoint. For a standalone checkpoint, the important parts are the deprecated
legacy `spec.identity` metadata, a container named `main`, and the placeholder
image; the rest of the pod template should mirror your normal worker config.
Extra containers are allowed, but only `main` is checkpointed unless
`spec.job.targetContainerName` selects another container.

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoCheckpoint
metadata:
  name: qwen3-06b-bf16
spec:
  identity:
    model: Qwen/Qwen3-0.6B
    backendFramework: vllm
    tensorParallelSize: 1
    dtype: bfloat16
    maxModelLen: 2048

  job:
    activeDeadlineSeconds: 3600
    podTemplateSpec:
      spec:
        ...
        containers:
          - name: main
            image: registry.example.com/dynamo/vllm-placeholder:1.0.0
            ...
```

GMS + Snapshot support is currently disabled.

For a full working example, see [deploy/operator/config/samples/nvidia.com_v1alpha1_dynamocheckpoint.yaml](https://github.com/ai-dynamo/dynamo/blob/main/deploy/operator/config/samples/nvidia.com_v1alpha1_dynamocheckpoint.yaml).

Apply it:

```bash
kubectl apply -f qwen3-checkpoint.yaml -n ${NAMESPACE}
```

### 5. Wait for the checkpoint to become ready

```bash
kubectl get dckpt -n ${NAMESPACE} \
  -o custom-columns=NAME:.metadata.name,CHECKPOINT_ID:.status.checkpointID,PHASE:.status.phase

kubectl wait \
  --for=jsonpath='{.status.phase}'=Ready \
  dynamocheckpoint/qwen3-06b-bf16 \
  -n ${NAMESPACE} \
  --timeout=30m
```

The useful status fields are:

- `status.phase`: high-level lifecycle (`Pending`, `Creating`, `Ready`, `Failed`)
- `status.checkpointID`: artifact ID used by the snapshot protocol
- `status.identityHash`: deprecated compatibility alias for `status.checkpointID`
- `status.jobName`: checkpoint Job name
- `status.createdAt`: timestamp recorded when the checkpoint became ready
- `status.message`: progress or failure detail when available

### 6. Deploy a `DynamoGraphDeployment` that restores from `checkpointRef`

Once the checkpoint is `Ready`, restore a worker from it explicitly:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: vllm-checkpointref-demo
spec:
  services:
    Frontend:
      componentType: frontend
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: registry.example.com/dynamo/vllm-runtime:1.0.0

    VllmDecodeWorker:
      componentType: worker
      replicas: 1
      checkpoint:
        enabled: true
        checkpointRef: qwen3-06b-bf16
      extraPodSpec:
        mainContainer:
          image: registry.example.com/dynamo/vllm-placeholder:1.0.0
          ...
        ...
```

Apply it:

```bash
kubectl apply -f vllm-checkpointref-demo.yaml -n ${NAMESPACE}
kubectl get pods -n ${NAMESPACE} -w
```

The `VllmDecodeWorker` pod should restore from the ready checkpoint instead of creating a new one.

## Choosing a DGD checkpoint flow

Enable checkpointing with `checkpoint.enabled: true`. The presence of
`checkpointRef` selects the restore flow:

| Config | What happens | Use when |
|--------|--------------|----------|
| `enabled: true` and no `checkpointRef` | The DGD creates and owns one automatic checkpoint per worker generation. | Normal DGD-managed checkpointing. |
| `enabled: true` and `checkpointRef: <name>` | The DGD restores from the named existing checkpoint and creates none. | Explicit restore from a retained or pre-warmed checkpoint. |

In v1beta1, replace the old `mode: Auto` form with:

```yaml
experimental:
  checkpoint:
    enabled: true
```

In v1alpha1, keep the existing enable flag and omit `mode`:

```yaml
checkpoint:
  enabled: true
```

The old `mode` field is deprecated. Omit it in new configs.

`startupPolicy` controls when normal workers start relative to checkpoint
readiness. `Immediate` starts workers cold while the checkpoint job runs;
`WaitForCheckpoint` keeps replicas at zero until the checkpoint is ready.

`deletionPolicy` applies only to DGD-owned automatic checkpoint custom
resources. `Delete` removes the checkpoint resource with the DGD. `Retain`
preserves the resource so it can be used later with `checkpointRef`. Both
policies leave artifact data on the snapshot PVC.

## DGD-managed automatic checkpoints

Without `checkpointRef`, the DGD-managed path is used. For each
checkpoint-enabled worker generation, the DGD controller creates a DGD-owned
`DynamoCheckpoint`, and the checkpoint controller starts a checkpoint Job.
Automatic checkpoints are not reused across DGDs, even when two manifests are
identical.

The automatic checkpoint ID is derived from the DGD namespace/name/UID,
component name, and active worker hash. The DGD UID prevents cross-DGD reuse;
the worker hash lets a scale down/up on the same worker generation use the same
DGD-scoped checkpoint while creating a new checkpoint for a new worker
generation.

Treat a restored pod template as a compatibility template for the same workload:
once the checkpoint is ready, restore admission replaces the target container's
command and args with the restore placeholder, and the restored process resumes
from the checkpointed state rather than newly supplied command-line or
environment settings.

With `startupPolicy: Immediate`, existing Pods are not mutated or restarted just
because the checkpoint became ready. Scale or roll the worker to create restored
Pods after the checkpoint is ready.

For v1beta1 components, the automatic checkpoint config is:

```yaml
experimental:
  checkpoint:
    enabled: true
    startupPolicy: Immediate # default; optional
    deletionPolicy: Delete  # default; use Retain to keep the CR after DGD deletion
```

For v1alpha1 services, the automatic checkpoint config is:

```yaml
checkpoint:
  enabled: true
  startupPolicy: Immediate # default; optional
  deletionPolicy: Delete  # default; use Retain to keep the CR after DGD deletion
```

Inside a `DynamoGraphDeployment`, it looks like this:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: vllm-auto-demo
spec:
  services:
    Frontend:
      componentType: frontend
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: registry.example.com/dynamo/vllm-runtime:1.0.0

    VllmDecodeWorker:
      componentType: worker
      replicas: 1
      checkpoint:
        enabled: true
        startupPolicy: Immediate
        deletionPolicy: Delete
      extraPodSpec:
        mainContainer:
          image: registry.example.com/dynamo/vllm-placeholder:1.0.0
          ...
        ...
```

The legacy `checkpoint.identity` field is ignored for DGD-managed automatic checkpoints. It is retained only for API compatibility and standalone `DynamoCheckpoint` workflows.

Useful inspection commands:

```bash
kubectl get dgd vllm-auto-demo -n ${NAMESPACE} \
  -o jsonpath='{.status.checkpoints.VllmDecodeWorker.checkpointName}{"\n"}{.status.checkpoints.VllmDecodeWorker.checkpointID}{"\n"}{.status.checkpoints.VllmDecodeWorker.ready}{"\n"}'

kubectl get dckpt -n ${NAMESPACE}
```

If you use the default `Immediate` policy and want to create restored pods after the checkpoint becomes ready, scale the worker:

```bash
kubectl patch dgd vllm-auto-demo -n ${NAMESPACE} --type=merge \
  -p '{"spec":{"services":{"VllmDecodeWorker":{"replicas":2}}}}'
```

## Failover Restore

Failover restore is not yet available. The current Snapshot flow does not support GMS + Snapshot, so do not use failover restore as a supported checkpoint/restore path. For current GMS and active/passive failover guidance, see [Shadow Engine Failover](shadow-engine-failover.md).

## Lower-Level Testing With `snapshotctl`

You can checkpoint and restore pods with the lower-level `snapshotctl` utility.
Install the snapshot Helm chart first and ensure an agent is running on the
workload node. A namespace-restricted agent runs in the workload namespace; a
cluster-wide agent can run in an infrastructure namespace.

`snapshotctl` is intended for lower-level debugging and validation workflows, not as the primary user-facing checkpoint interface. For command details and manifest requirements, see [deploy/snapshot/cmd/snapshotctl/README.md](https://github.com/ai-dynamo/dynamo/blob/main/deploy/snapshot/cmd/snapshotctl/README.md).

### Checkpoint from a worker pod manifest

```bash
snapshotctl checkpoint \
  --manifest ./worker-pod.yaml \
  --container main \
  --namespace ${NAMESPACE}
```

The checkpoint manifest must be for a pod and use a placeholder image. `--container` names the workload container to checkpoint.

If you do not pass `--checkpoint-id`, `snapshotctl` generates one. After capture
completes, it reads the artifact handle reported by the agent on the bound
`PodSnapshotContent`:

```text
status=completed
namespace=...
name=...
checkpoint_job=...
checkpoint_id=manual-snapshot-...
checkpoint_location=/checkpoints/...
pod_snapshot=...
bound_content=...
```

`checkpoint_location` is an observed result rather than a predicted path. If
the agent has not populated the handle, `snapshotctl` omits that line while
continuing to report the checkpoint ID and status.

### Restore from a worker pod manifest

```bash
snapshotctl restore \
  --manifest ./worker-pod.yaml \
  --namespace ${NAMESPACE} \
  --checkpoint-id manual-snapshot-... \
  --containers main
```

This creates a new restore pod and returns after the request is submitted.
Restore output does not include `checkpoint_location`; the agent resolves the
artifact from its own configuration. Observe progress through Kubernetes
readiness, events, and logs.

### Restore an existing pod in place

```bash
snapshotctl restore \
  --pod existing-restore-target \
  --namespace ${NAMESPACE} \
  --checkpoint-id manual-snapshot-... \
  --containers main
```

This patches restore metadata onto an existing pod that is already snapshot-compatible and returns after the patch is accepted.

## Checkpoint IDs and Legacy Identity

`status.checkpointID` is the artifact ID used by the snapshot protocol and the
directory name under checkpoint storage. For DGD-managed automatic checkpoints,
this ID is scoped to a single DGD/component worker generation. It is not a
compatibility claim across DGDs, and identical manifests are not treated as
proof that a checkpoint can be reused safely.

The deprecated legacy `spec.identity` shape is still required on standalone
v1alpha1 `DynamoCheckpoint` objects. When a standalone checkpoint does not
already have `status.checkpointID` or the checkpoint-ID label, the operator computes the
legacy **16-character SHA256 hash** (64 bits) from these fields:

| Legacy field | Required | Affects legacy hash | Example |
|--------------|----------|---------------------|---------|
| `model` | ✓ | ✓ | `meta-llama/Llama-3-8B` |
| `backendFramework` | ✓ | ✓ | `vllm` |
| `dynamoVersion` | | ✓ | `0.9.0`, `1.0.0` |
| `tensorParallelSize` | | ✓ | `1`, `2`, `4`, `8` |
| `pipelineParallelSize` | | ✓ | `1`, `2` |
| `dtype` | | ✓ | `float16`, `bfloat16`, `fp8` |
| `maxModelLen` | | ✓ | `4096`, `8192` |
| `extraParameters` | | ✓ | custom key-value pairs |

Fields that do **not** change the legacy hash include:

- replica count
- node placement (`nodeSelector`, `affinity`, `tolerations`)
- resource requests/limits
- logging or observability configuration

DGD-managed automatic checkpoints ignore legacy identity as a reuse boundary:
omit `identity`; use `checkpointRef` to restore an existing checkpoint. The DGD
controller creates a DGD-scoped checkpoint ID and only synthesizes legacy
identity because the v1alpha1 `DynamoCheckpoint` API still requires it.

Checkpoint job launch behavior is inferred from the checkpoint Job pod template
(GPU resources or DRA claims), not from legacy identity fields such as
`tensorParallelSize` or `pipelineParallelSize`.

## `DynamoCheckpoint` CRD

The `DynamoCheckpoint` (shortname: `dckpt`) is the operator-managed resource for checkpoint lifecycle.

Use it when you want:

- pre-warmed checkpoints before any `DynamoGraphDeployment` exists
- explicit lifecycle control independent from a DGD
- a stable human-readable name that services can reference with `checkpointRef`

The operator requires:

- `spec.identity`
- `spec.job.podTemplateSpec`

`spec.job.backoffLimit` is deprecated and ignored. Checkpoint Jobs are always single-attempt.

Check status with:

```bash
kubectl get dckpt -n ${NAMESPACE}
kubectl describe dckpt qwen3-06b-bf16 -n ${NAMESPACE}
kubectl get dckpt qwen3-06b-bf16 -n ${NAMESPACE} -o yaml
```

The `status` block looks like:

```yaml
status:
  phase: Ready
  checkpointID: 3bff874d069f0ed5
  identityHash: 3bff874d069f0ed5 # deprecated compatibility alias
  jobName: checkpoint-job-3bff874d069f0ed5-1
  createdAt: "2026-01-29T10:05:00Z"
  message: ""
```

## Limitations

- **Backend and topology support is limited**: single-GPU support is the most mature path; see [Backend and Topology Support](#backend-and-topology-support) for the current scope.
- **Worker coverage is narrow**: specialized workers such as multimodal, embedding, and diffusion are not supported.
- **Multi-GPU remains preview**: vLLM tensor-parallel configurations have limited validation and are not yet a broadly supported path across clusters.
- **GMS restore remains experimental**: GMS + Snapshot is currently disabled.
- **Admission is create-only**: with DGD `startupPolicy: Immediate`, only Pods created after a checkpoint is `Ready` are restore-shaped. Existing Pods cold-started before checkpoint readiness keep running as-is.
- **Restore admission must be installed**: DGD restores rely on the snapshot Pod mutating webhook, so upgrade the snapshot chart/webhook configuration along with the operator and CRDs when enabling these features.
- **Network state is sensitive**: restore is sensitive to live TCP socket state. Loopback bootstrap/control sockets are the most reliable path today.
- **Privileged DaemonSet required**: `snapshot-agent` must run privileged to execute CRIU and `cuda-checkpoint`. Workload pods do not need to be privileged.

## Troubleshooting

### Checkpoint Job finishes but the checkpoint never becomes `Ready`

Snapshot only becomes `Ready` after `snapshot-agent` confirms the checkpoint contents. A completed Job is not enough by itself.

```bash
kubectl get dckpt <checkpoint-name> -n ${NAMESPACE} \
  -o custom-columns=NAME:.metadata.name,PHASE:.status.phase,MESSAGE:.status.message,JOB:.status.jobName

JOB_NAME=$(kubectl get dckpt <checkpoint-name> -n ${NAMESPACE} -o jsonpath='{.status.jobName}')
if [ -n "${JOB_NAME}" ]; then
  kubectl logs job/"${JOB_NAME}" -n ${NAMESPACE}
fi

kubectl logs daemonset/snapshot-agent -n ${NAMESPACE} --all-containers
```

If the worker template is wrong, the most common causes are using the raw runtime image instead of the placeholder image, or leaving out normal mounts and secrets that the worker needs to start.

### Restore cannot find or mount checkpoint storage

The agent resolves every artifact from `storage.pvc.basePath` in its own
configuration. It does not read a storage path from the workload pod or
checkpoint resource. For a namespace-restricted installation, verify the agent
and PVC in the workload namespace:

```bash
kubectl rollout status daemonset/snapshot-agent -n ${NAMESPACE}
kubectl get daemonset -n ${NAMESPACE} -l app.kubernetes.io/component=snapshot-agent -o wide
kubectl get pvc -n ${NAMESPACE}
```

For a cluster-wide installation, verify the agent and its shared PVC in the
chart release namespace:

```bash
kubectl rollout status daemonset/snapshot-agent -n dynamo-system
kubectl get pods -n dynamo-system -l app.kubernetes.io/component=snapshot-agent -o wide
kubectl get pvc snapshot-pvc -n dynamo-system
```

Check the agent logs for the resolved artifact path and confirm that every agent
pod mounts the same claim at the configured base path. A workload pod should
not contain a checkpoint PVC volume or a mount at that path.

### `snapshotctl` manifest is rejected or the restore target is wrong

`snapshotctl` requires a `Pod` manifest and a target-container list. Multi-container manifests are supported as long as every name passed via `--container` or `--containers` exists in the pod spec.

```bash
snapshotctl checkpoint --manifest ./worker-pod.yaml --container main --namespace ${NAMESPACE}
snapshotctl restore  --manifest ./worker-pod.yaml --containers main --namespace ${NAMESPACE} --checkpoint-id <checkpoint-id>
```

If the manifest already carries snapshot target metadata, it must agree with the CLI flag; `snapshotctl` rejects mismatches instead of silently picking one.

## Planned Features

- Stable multi-GPU and multinode support
- Broader TensorRT-LLM coverage beyond the current single-GPU aggregated text path

## Related Documentation

- [Installation Guide](../../../../kubernetes/installation/install-dynamo.md)
- [Shadow Engine Failover](shadow-engine-failover.md)
- [API Reference](../../../../reference/kubernetes-api/full-api-reference.mdx)
