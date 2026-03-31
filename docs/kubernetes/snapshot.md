---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Snapshot
---

> ⚠️ **Experimental Feature**: Dynamo Snapshot is currently in preview and may only be functional in some cluster setups. The `snapshot-agent` DaemonSet runs in privileged mode to perform CRIU operations. See [Limitations](#limitations) for details.

**Dynamo Snapshot** is infrastructure for fast-starting GPU applications in Kubernetes using CRIU (Checkpoint/Restore in Userspace) and NVIDIA's `cuda-checkpoint` utility. The usual flow is:

1. start a worker once and checkpoint its initialized state
2. store that checkpoint on a namespace-local snapshot volume
3. restore later workers from that checkpoint instead of cold-starting again

| Startup Type | Time | What Happens |
|--------------|------|--------------|
| **Cold Start** | ~1 min | Download model, load to GPU, initialize engine |
| **Warm Start** (restore from checkpoint) | ~10 sec | Restore from a ready checkpoint directory |

> ⚠️ Restore time depends on storage bandwidth, GPU model, and whether the restore stays on the same node.

## Prerequisites

- x86_64 (`amd64`) GPU nodes
- NVIDIA driver 580.xx or newer on the target GPU nodes
- security clearance to run a privileged DaemonSet
- vLLM or SGLang backend today
- `ReadWriteMany` storage if you need cross-node restore
- for operator-managed flows: Dynamo Platform/Operator installed with checkpointing enabled
- for lower-level `snapshotctl` testing: the snapshot chart installed in the target namespace; the operator is not required

Install the snapshot chart in every namespace where you want checkpoint and restore. Snapshot storage is owned by that chart, not by the operator.

## Quick Start: Explicit `DynamoCheckpoint` + `checkpointRef`

This is the clearest operator-managed flow:

1. build a placeholder image
2. install the snapshot chart
3. create a `DynamoCheckpoint`
4. wait for it to become `Ready`
5. deploy a `DynamoGraphDeployment` that restores from `checkpointRef`

### 1. Build and push a placeholder image

Snapshot-enabled workers must use a placeholder image that wraps the normal runtime image with restore tooling. If you do not already have one, build it and push it to a registry your cluster can pull from:

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

The placeholder image preserves the normal runtime entrypoint/command contract and adds the `criu`, `cuda-checkpoint`, and `nsrestore` tooling needed for checkpoint and restore.

### 2. Enable operator checkpointing

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

### 3. Install the snapshot chart in the workload namespace

```bash
helm upgrade --install snapshot ./deploy/helm/charts/snapshot \
  --namespace ${NAMESPACE} \
  --create-namespace \
  --set storage.pvc.create=true
```

Cross-node restore requires shared `ReadWriteMany` storage. The chart defaults to that mode. If your cluster does not have a default storage class, also set `storage.pvc.storageClass`.

If you are reusing an existing checkpoint PVC, do not set `storage.pvc.create=true`; install the chart with `storage.pvc.create=false` and set `storage.pvc.name` instead.

Only `storage.type=pvc` is implemented today. The chart keeps `storage.type` for future snapshot-owned backends, but `s3` and `oci` are not supported yet.

Verify that the PVC and DaemonSet are ready:

```bash
kubectl get pvc snapshot-pvc -n ${NAMESPACE}
kubectl rollout status daemonset/snapshot-agent -n ${NAMESPACE}
kubectl get pods -n ${NAMESPACE} -l app.kubernetes.io/component=snapshot-agent -o wide
```

### 4. Create a `DynamoCheckpoint`

The checkpoint Job pod template should match the worker container you want to checkpoint. For the snapshot flow, the important parts are the checkpoint identity and the placeholder image; the rest of the pod template should mirror your normal worker config.

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

For a full working example, see [deploy/operator/config/samples/nvidia.com_v1alpha1_dynamocheckpoint.yaml](https://github.com/ai-dynamo/dynamo/blob/main/deploy/operator/config/samples/nvidia.com_v1alpha1_dynamocheckpoint.yaml).

Apply it:

```bash
kubectl apply -f qwen3-checkpoint.yaml -n ${NAMESPACE}
```

### 5. Wait for the checkpoint to become ready

```bash
kubectl get dckpt -n ${NAMESPACE} \
  -o custom-columns=NAME:.metadata.name,HASH:.status.identityHash,PHASE:.status.phase

kubectl wait \
  --for=jsonpath='{.status.phase}'=Ready \
  dynamocheckpoint/qwen3-06b-bf16 \
  -n ${NAMESPACE} \
  --timeout=30m
```

The useful status fields are:

- `status.phase`: high-level lifecycle (`Pending`, `Creating`, `Ready`, `Failed`)
- `status.identityHash`: deterministic hash of `spec.identity`
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

## DGD Auto Flow

`checkpointRef` is the most explicit path. `mode: Auto` is the higher-level path: the operator computes the checkpoint identity hash, looks for an equivalent `DynamoCheckpoint`, and creates one only when no matching checkpoint exists.

If you already created an explicit `DynamoCheckpoint` with the same identity, Auto mode reuses it. If no matching checkpoint exists yet, the first worker cold-starts and the operator creates the checkpoint in the background.

```yaml
checkpoint:
  enabled: true
  mode: Auto
  identity:
    model: Qwen/Qwen3-0.6B
    backendFramework: vllm
    tensorParallelSize: 1
    dtype: bfloat16
    maxModelLen: 2048
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
        mode: Auto
        identity:
          model: Qwen/Qwen3-0.6B
          backendFramework: vllm
          tensorParallelSize: 1
          dtype: bfloat16
          maxModelLen: 2048
      extraPodSpec:
        mainContainer:
          image: registry.example.com/dynamo/vllm-placeholder:1.0.0
          ...
        ...
```

Useful inspection commands:

```bash
kubectl get dgd vllm-auto-demo -n ${NAMESPACE} \
  -o jsonpath='{.status.checkpoints.VllmDecodeWorker.checkpointName}{"\n"}{.status.checkpoints.VllmDecodeWorker.identityHash}{"\n"}{.status.checkpoints.VllmDecodeWorker.ready}{"\n"}'

kubectl get dckpt -n ${NAMESPACE}
```

If you want to force a new restore after the checkpoint becomes ready, scale the worker:

```bash
kubectl patch dgd vllm-auto-demo -n ${NAMESPACE} --type=merge \
  -p '{"spec":{"services":{"VllmDecodeWorker":{"replicas":2}}}}'
```

## Lower-Level Testing With `snapshotctl`

Use `snapshotctl` when you want to validate snapshot infrastructure without the operator. This is the fastest way to answer questions like:

- can this cluster run `snapshot-agent` correctly?
- can this worker image be checkpointed?
- can a restore pod come back ready from a known checkpoint?

This path still requires the snapshot chart in the target namespace, but it does **not** require Dynamo Platform/Operator.

### `snapshotctl` requirements

- the manifest must be a `Pod`
- `metadata.name` must be set
- the manifest must contain exactly one worker container
- that worker image should be a placeholder image if you want to restore it
- the target namespace must already have a ready `snapshot-agent` DaemonSet and mounted checkpoint PVC

The `snapshotctl` commands discover checkpoint storage by inspecting the `snapshot-agent` DaemonSet in the target namespace.

### Checkpoint from a worker pod manifest

```bash
snapshotctl checkpoint \
  --manifest ./worker-pod.yaml \
  --namespace ${NAMESPACE}
```

If you do not pass `--checkpoint-id`, `snapshotctl` generates one and prints it:

```text
status=completed
namespace=...
name=...
checkpoint_job=...
checkpoint_id=manual-snapshot-...
checkpoint_location=/checkpoints/...
```

### Restore from a worker pod manifest

```bash
snapshotctl restore \
  --manifest ./worker-pod.yaml \
  --namespace ${NAMESPACE} \
  --checkpoint-id manual-snapshot-...
```

This creates a new restore pod from the manifest and waits for the restore annotation to reach `completed`.

### Restore an existing pod in place

```bash
snapshotctl restore \
  --pod existing-restore-target \
  --namespace ${NAMESPACE} \
  --checkpoint-id manual-snapshot-...
```

This patches restore metadata onto an existing pod that is already snapshot-compatible.

## Checkpoint Identity

Checkpoints are uniquely identified by a **16-character SHA256 hash** (64 bits) of configuration that affects runtime state:

| Field | Required | Affects Hash | Example |
|-------|----------|-------------|---------|
| `model` | ✓ | ✓ | `meta-llama/Llama-3-8B` |
| `backendFramework` | ✓ | ✓ | `sglang`, `vllm` |
| `dynamoVersion` | | ✓ | `0.9.0`, `1.0.0` |
| `tensorParallelSize` | | ✓ | `1`, `2`, `4`, `8` |
| `pipelineParallelSize` | | ✓ | `1`, `2` |
| `dtype` | | ✓ | `float16`, `bfloat16`, `fp8` |
| `maxModelLen` | | ✓ | `4096`, `8192` |
| `extraParameters` | | ✓ | custom key-value pairs |

Fields that do **not** change the checkpoint hash include:

- replica count
- node placement (`nodeSelector`, `affinity`, `tolerations`)
- resource requests/limits
- logging or observability configuration

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
  identityHash: 3bff874d069f0ed5
  jobName: checkpoint-job-3bff874d069f0ed5-1
  createdAt: "2026-01-29T10:05:00Z"
  message: ""
```

## Limitations

- **LLM workers only**: checkpoint/restore supports LLM decode and prefill workers. Specialized workers such as multimodal, embedding, and diffusion are not supported.
- **Multi-GPU remains preview**: tensor-parallel configurations are exercised in internal testing, but they are not yet a broadly supported production path across clusters.
- **Network state is sensitive**: restore is sensitive to live TCP socket state. Loopback bootstrap/control sockets are the most reliable path today.
- **Privileged DaemonSet required**: `snapshot-agent` must run privileged to execute CRIU and `cuda-checkpoint`. Workload pods do not need to be privileged.

## Troubleshooting

### Checkpoint never becomes `Ready`

1. Check the checkpoint resource and Job:

   ```bash
   kubectl get dckpt -n ${NAMESPACE}
   kubectl describe dckpt <checkpoint-name> -n ${NAMESPACE}
   JOB_NAME=$(kubectl get dckpt <checkpoint-name> -n ${NAMESPACE} -o jsonpath='{.status.jobName}')
   if [ -n "${JOB_NAME}" ]; then
     kubectl logs job/"${JOB_NAME}" -n ${NAMESPACE}
   fi
   ```

2. Check the snapshot agent:

   ```bash
   kubectl rollout status daemonset/snapshot-agent -n ${NAMESPACE}
   kubectl logs daemonset/snapshot-agent -n ${NAMESPACE} --all-containers
   ```

3. Verify the same namespace has a ready snapshot chart install:

   ```bash
   kubectl get pvc -n ${NAMESPACE}
   kubectl get daemonset -n ${NAMESPACE} -l app.kubernetes.io/component=snapshot-agent -o wide
   ```

4. Verify the workload pod template matches the worker you intend to checkpoint:

   - placeholder image, not the raw runtime image
   - same command and args you expect to reuse later
   - required model/cache secrets and PVC mounts
   - loopback socket envs for the tested vLLM/SGLang flows

### Restore fails

1. Check the worker pod:

   ```bash
   kubectl logs <worker-pod> -n ${NAMESPACE}
   kubectl describe pod <worker-pod> -n ${NAMESPACE}
   ```

2. Confirm the referenced checkpoint is ready:

   ```bash
   kubectl get dckpt <checkpoint-name> -n ${NAMESPACE}
   ```

3. If you are using `snapshotctl`, verify that the manifest is a single-container `Pod` and that the target namespace has a ready `snapshot-agent` DaemonSet.

## Planned Features

- TensorRT-LLM backend support
- snapshot-owned S3/MinIO storage backend
- snapshot-owned OCI registry storage backend
- broader multi-GPU support

## Related Documentation

- [Installation Guide](installation-guide.md)
- [API Reference](api-reference.md)
