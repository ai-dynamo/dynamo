---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Snapshot
---

> ⚠️ **Experimental Feature**: Dynamo Snapshot is currently in **preview** and may only be functional in some k8s cluster setups. The Dynamo Snapshot DaemonSet runs in privileged mode to perform CRIU operations. See [Limitations](#limitations) for details.

**Dynamo Snapshot** is an experimental infrastructure for fast-starting GPU applications in Kubernetes using CRIU (Checkpoint/Restore in User-space) and NVIDIA's cuda-checkpoint utility. Dynamo Snapshot dramatically reduces cold-start times for large models from minutes to seconds by capturing initialized application state and restoring it on-demand.

| Startup Type | Time | What Happens |
|--------------|------|--------------|
| **Cold Start** | ~1 min | Download model, load to GPU, initialize engine |
| **Warm Start** (restore from checkpoint) | ~ 10 sec | Restore from checkpoint tar |

> ⚠️ Restore time may vary depending on cluster configuration (storage bandwidth, GPU model, etc.)

## Prerequisites

- Dynamo Platform/Operator installed on a k8s cluster with **x86_64 (amd64)** GPU nodes
- NVIDIA driver 580.xx or newer on the target GPU nodes
- `ReadWriteMany` storage if you need cross-node restore
- vLLM or SGLang backend (TensorRT-LLM is not supported yet)
- Security clearance to run a privileged DaemonSet

## Quick Start

This guide assumes a normal Dynamo deployment workflow is already present on your Kubernetes cluster.

### 1. Build and push a placeholder image

Snapshot-enabled workers must use a placeholder image that wraps the normal runtime image with the restore tooling. If you do not already have one, build it with the snapshot placeholder target and push it to a registry your cluster can pull from:

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

This flow is defined in [deploy/snapshot/Makefile](https://github.com/ai-dynamo/dynamo/blob/main/deploy/snapshot/Makefile) and [deploy/snapshot/Dockerfile](https://github.com/ai-dynamo/dynamo/blob/main/deploy/snapshot/Dockerfile). The placeholder image preserves the base runtime entrypoint and command contract, and adds the CRIU, `cuda-checkpoint`, and `nsrestore` tooling needed for restore.

### 2. Enable checkpointing in the platform and verify it

Whether you are installing or upgrading `dynamo-platform`, the operator must have checkpointing enabled and must point at the same storage that the snapshot chart will use:

```yaml
dynamo-operator:
  checkpoint:
    enabled: true
    storage:
      type: pvc
      pvc:
        pvcName: snapshot-pvc
        basePath: /checkpoints
```

If the platform is already installed, verify that the operator config contains the checkpoint block:

```bash
OPERATOR_CONFIG=$(kubectl get deploy -n "${PLATFORM_NAMESPACE}" \
  -l app.kubernetes.io/name=dynamo-operator,app.kubernetes.io/component=manager \
  -o jsonpath='{.items[0].spec.template.spec.volumes[?(@.name=="operator-config")].configMap.name}')

kubectl get configmap "${OPERATOR_CONFIG}" -n "${PLATFORM_NAMESPACE}" \
  -o jsonpath='{.data.config\.yaml}' | sed -n '/^checkpoint:/,/^[^[:space:]]/p'
```

Verify that the rendered config includes `enabled: true` and the same PVC name and base path you plan to use for the snapshot chart.

For the full platform/operator configuration surface, see [deploy/helm/charts/platform/README.md](https://github.com/ai-dynamo/dynamo/blob/main/deploy/helm/charts/platform/README.md) and [deploy/helm/charts/platform/components/operator/values.yaml](https://github.com/ai-dynamo/dynamo/blob/main/deploy/helm/charts/platform/components/operator/values.yaml).

### 3. Install the snapshot chart

```bash
helm upgrade --install snapshot ./deploy/helm/charts/snapshot \
  --namespace ${NAMESPACE} \
  --create-namespace \
  --set storage.pvc.create=true
```

Cross-node restore requires `ReadWriteMany` storage. The chart defaults to that mode.

For better restore times, use a fast `ReadWriteMany` StorageClass for the checkpoint PVC. If you are reusing an existing checkpoint PVC, do not set `storage.pvc.create=true`; install the chart with `storage.pvc.create=false` and point `storage.pvc.name` at the existing PVC instead.

Verify that the PVC and DaemonSet are ready:

```bash
kubectl get pvc snapshot-pvc -n ${NAMESPACE}
kubectl rollout status daemonset/snapshot-agent -n ${NAMESPACE}
```

For the full snapshot chart configuration surface, see [deploy/helm/charts/snapshot/README.md](https://github.com/ai-dynamo/dynamo/blob/main/deploy/helm/charts/snapshot/README.md) and [deploy/helm/charts/snapshot/values.yaml](https://github.com/ai-dynamo/dynamo/blob/main/deploy/helm/charts/snapshot/values.yaml).

### 4. Apply a snapshot-compatible `DynamoGraphDeployment`

This example is adapted from [examples/backends/vllm/deploy/agg.yaml](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/agg.yaml). The worker must use the placeholder image from step 1, and the checkpoint identity must describe the runtime state you want to reuse.

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: vllm-snapshot-demo
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
      resources:
        limits:
          gpu: "1"
      readinessProbe:
        httpGet:
          path: /live
          port: system
        periodSeconds: 1
        timeoutSeconds: 4
        failureThreshold: 3
      checkpoint:
        enabled: true
        mode: Auto
        identity:
          model: Qwen/Qwen3-0.6B
          backendFramework: vllm
      extraPodSpec:
        mainContainer:
          image: registry.example.com/dynamo/vllm-placeholder:1.0.0
          command:
            - python3
            - -m
            - dynamo.vllm
          args:
            - --model
            - Qwen/Qwen3-0.6B
            - --disable-custom-all-reduce
          env:
            - name: GLOO_SOCKET_IFNAME
              value: lo
            - name: NCCL_SOCKET_IFNAME
              value: lo
            - name: NCCL_DEBUG
              value: ERROR
            - name: TORCH_CPP_LOG_LEVEL
              value: ERROR
            - name: TORCH_DISTRIBUTED_DEBUG
              value: "OFF"
            - name: CUDA_ERROR_LEVEL
              value: "10"
            - name: NCCL_CUMEM_ENABLE
              value: "0"
            - name: NCCL_CUMEM_HOST_ENABLE
              value: "0"
            - name: NCCL_NVLS_ENABLE
              value: "0"
            - name: NCCL_P2P_DISABLE
              value: "0"
            - name: NCCL_SHM_DISABLE
              value: "1"
            - name: NCCL_IB_DISABLE
              value: "1"
            - name: TORCH_NCCL_ENABLE_MONITORING
              value: "0"
```

For SGLang, use `dynamo.sglang`, an SGLang placeholder image, `backendFramework: sglang`, and the matching CLI flags.

Apply the manifest:

```bash
kubectl apply -f vllm-snapshot-demo.yaml -n ${NAMESPACE}
```

On the first rollout, the worker cold-starts, the operator creates a `DynamoCheckpoint`, and the checkpoint Job writes data into `snapshot-pvc`.

### 5. Wait for the checkpoint to become ready

Capture the checkpoint name from DGD status, then wait for the `DynamoCheckpoint` phase to become `Ready`:

```bash
CHECKPOINT_NAME=$(kubectl get dgd vllm-snapshot-demo -n ${NAMESPACE} \
  -o jsonpath='{.status.checkpoints.VllmDecodeWorker.checkpointName}')

kubectl wait \
  --for=jsonpath='{.status.phase}'=Ready \
  "dynamocheckpoint/${CHECKPOINT_NAME}" \
  -n ${NAMESPACE} \
  --timeout=30m
```

The DGD status also reports the computed checkpoint hash at `.status.checkpoints.VllmDecodeWorker.identityHash`.

### 6. Trigger restore

Once the checkpoint is ready, scale the worker replicas from `1` to `2`:

```bash
kubectl patch dgd vllm-snapshot-demo -n ${NAMESPACE} --type=merge \
  -p '{"spec":{"services":{"VllmDecodeWorker":{"replicas":2}}}}'
```

New worker pods for `VllmDecodeWorker` will restore from the ready checkpoint automatically.

## Checkpoint Configuration

### Auto Mode (Recommended)

The operator computes the checkpoint identity hash, looks for an existing `DynamoCheckpoint` with a matching `nvidia.com/snapshot-checkpoint-hash` label, and creates one if it does not find one:

```yaml
checkpoint:
  enabled: true
  mode: Auto
  identity:
    model: "meta-llama/Llama-3-8B"
    backendFramework: "vllm"  # or "sglang"
    tensorParallelSize: 1
    dtype: "bfloat16"
    maxModelLen: 4096
```

When a service uses checkpointing, DGD status reports the resolved `checkpointName`, `identityHash`, and `ready` fields under `.status.checkpoints.<service-name>`.

### Manual Management and `checkpointRef`

Use `checkpointRef` when you want a service to restore from a specific `DynamoCheckpoint` CR:

```yaml
checkpoint:
  enabled: true
  checkpointRef: "qwen3-06b-vllm-prewarm"
```

This is useful when:
- You want to **pre-warm checkpoints** before creating DGDs
- You want **explicit control** over which checkpoint to use

`checkpointRef` resolves by `DynamoCheckpoint.metadata.name`, not by `status.identityHash`. A manual checkpoint can use any valid Kubernetes resource name.

If you are managing checkpoint CRs yourself, set `mode: Manual` on the service to prevent the operator from creating a new `DynamoCheckpoint` when identity-based lookup does not find one.

```bash
# Check checkpoint status by CR name
kubectl get dynamocheckpoint qwen3-06b-vllm-prewarm -n ${NAMESPACE}

# Now create DGD referencing it
kubectl apply -f my-dgd.yaml -n ${NAMESPACE}
```

If you want `mode: Auto` DGDs to discover a manually created checkpoint by identity, add the label `nvidia.com/snapshot-checkpoint-hash=<identity-hash>` to that `DynamoCheckpoint`. Auto-created checkpoints already use that label, and currently use the same hash as the CR name.

## Checkpoint Identity

Checkpoints are uniquely identified by a **16-character SHA256 hash** (64 bits) of configuration that affects runtime state:

| Field | Required | Affects Hash | Example |
|-------|----------|-------------|---------|
| `model` | ✓ | ✓ | `meta-llama/Llama-3-8B` |
| `backendFramework` | ✓ | ✓ | `sglang`, `vllm` |
| `dynamoVersion` | | ✓ | `0.9.0`, `1.0.0` |
| `tensorParallelSize` | | ✓ | `1`, `2`, `4`, `8` (default: 1) |
| `pipelineParallelSize` | | ✓ | `1`, `2` (default: 1) |
| `dtype` | | ✓ | `float16`, `bfloat16`, `fp8` |
| `maxModelLen` | | ✓ | `4096`, `8192` |
| `extraParameters` | | ✓ | Custom key-value pairs |

**Not included in hash** (don't invalidate checkpoint):
- `replicas`
- `nodeSelector`, `affinity`, `tolerations`
- `resources` (requests/limits)
- Logging/observability config

**Example with all fields:**
```yaml
checkpoint:
  enabled: true
  mode: Auto
  identity:
    model: "meta-llama/Llama-3-8B"
    backendFramework: "vllm"
    dynamoVersion: "1.0.0"
    tensorParallelSize: 1
    pipelineParallelSize: 1
    dtype: "bfloat16"
    maxModelLen: 8192
    extraParameters:
      enableChunkedPrefill: "true"
      quantization: "awq"
```

## DynamoCheckpoint CRD

The `DynamoCheckpoint` (shortname: `dckpt`) is a Kubernetes Custom Resource that manages checkpoint lifecycle.

**When to create a DynamoCheckpoint directly:**
- **Pre-warming:** Create checkpoints before deploying DGDs for instant startup
- **Explicit control:** Manage checkpoint lifecycle independently from DGDs

The operator requires `spec.identity` and `spec.job.podTemplateSpec`. The pod template should match the worker container you want checkpointed, including image, command, args, secrets, volumes, and resource limits. You do not need to set the checkpoint environment variables manually; the operator injects them for checkpoint jobs and restored pods.

**Create a checkpoint:**

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoCheckpoint
metadata:
  name: qwen3-06b-vllm-prewarm
  labels:
    nvidia.com/snapshot-checkpoint-hash: "e5962d34ba272638"  # Add this if Auto-mode identity lookup should find the CR
spec:
  identity:
    model: Qwen/Qwen3-0.6B
    backendFramework: vllm
    tensorParallelSize: 1
    dtype: bfloat16
    maxModelLen: 4096

  job:
    activeDeadlineSeconds: 3600
    backoffLimit: 3
    ttlSecondsAfterFinished: 300
    podTemplateSpec:
      spec:
        restartPolicy: Never
        containers:
          - name: main
            image: registry.example.com/dynamo/vllm-placeholder:1.0.0
            command:
              - python3
              - -m
              - dynamo.vllm
            args:
              - --model
              - Qwen/Qwen3-0.6B
              - --disable-custom-all-reduce
            env:
              - name: GLOO_SOCKET_IFNAME
                value: lo
              - name: NCCL_SOCKET_IFNAME
                value: lo
            resources:
              limits:
                nvidia.com/gpu: "1"
```

You can name the CR however you want if you plan to use `checkpointRef`. If you want `mode: Auto` identity lookup to find a manual CR, set the `nvidia.com/snapshot-checkpoint-hash` label to the computed 16-character identity hash. Using the hash as the CR name is a convenient convention, but it is not required.

**Check status:**

```bash
# List all checkpoints
kubectl get dynamocheckpoint -n ${NAMESPACE}
# Or use shortname
kubectl get dckpt -n ${NAMESPACE}

NAME                MODEL                          BACKEND  PHASE    HASH              AGE
qwen3-06b-vllm-prewarm Qwen/Qwen3-0.6B            vllm     Ready    e5962d34ba272638  5m
llama3-8b-vllm-prewarm meta-llama/Llama-3-8B      vllm     Creating 7ab4f89c12de3456  2m
```

**Phases:**

| Phase | Description |
|-------|-------------|
| `Pending` | CR created, waiting for job to start |
| `Creating` | Checkpoint job is running |
| `Ready` | Checkpoint available for use |
| `Failed` | Checkpoint creation failed |

`Ready` is a value in `status.phase`, not a Kubernetes condition. The `conditions` array tracks job lifecycle events:

| Condition Type | Meaning |
|----------------|---------|
| `JobCreated` | The checkpoint Job has been created |
| `JobCompleted` | The checkpoint Job has completed successfully or failed |

Other useful status fields are:

| Field | Meaning |
|-------|---------|
| `status.jobName` | Name of the checkpoint Job |
| `status.identityHash` | Computed 16-character hash for the checkpoint identity |
| `status.location` | Checkpoint location in the configured storage backend |
| `status.storageType` | Storage backend type (`pvc`, `s3`, or `oci`) |
| `status.createdAt` | Timestamp recorded when the checkpoint becomes ready |
| `status.message` | Failure or progress message when available |

**Detailed status:**

```bash
kubectl describe dckpt qwen3-06b-vllm-prewarm -n ${NAMESPACE}
```

```yaml
Status:
  Phase: Ready
  IdentityHash: e5962d34ba272638
  JobName: checkpoint-qwen3-06b-vllm-prewarm
  Location: /checkpoints/e5962d34ba272638.tar
  StorageType: pvc
  CreatedAt: 2026-01-29T10:05:00Z
  Conditions:
    - Type: JobCreated
      Status: "True"
      Reason: JobCreated
    - Type: JobCompleted
      Status: "True"
      Reason: JobSucceeded
```

**Reference from DGD:**

Once the checkpoint is `Ready`, you can reference it by CR name:

```yaml
spec:
  services:
    VllmDecodeWorker:
      checkpoint:
        enabled: true
        checkpointRef: "qwen3-06b-vllm-prewarm"
```

Or use `mode: Auto` with the same identity and snapshot-hash label, and the operator will reuse it automatically.

## Limitations

- **LLM workers only**: Checkpoint/restore supports LLM decode and prefill workers. Specialized workers (multimodal, embedding, diffusion) are not supported.
- **Single-GPU only**: Multi-GPU configurations may work in very basic hardware configurations, but are not officially supported yet.
- **Network state**: No active TCP connections can be checkpointed
- **Security**: Dynamo Snapshot runs as a **privileged DaemonSet** which is required to run CRIU and cuda-checkpoint. However, workload pods do not need to be privileged.

## Troubleshooting

### Checkpoint Not Ready

1. Check the checkpoint job:
   ```bash
   kubectl get dckpt -n ${NAMESPACE}
   kubectl describe dckpt <checkpoint-name> -n ${NAMESPACE}
   kubectl logs job/$(kubectl get dckpt <checkpoint-name> -n ${NAMESPACE} -o jsonpath='{.status.jobName}') -n ${NAMESPACE}
   ```

2. Check the DaemonSet:
   ```bash
   kubectl logs daemonset/snapshot-agent -n ${NAMESPACE} --all-containers
   ```

3. Verify that platform and chart storage settings match:
   ```bash
   kubectl get dckpt <checkpoint-name> -n ${NAMESPACE} -o yaml
   ```

### Restore Failing

1. Check pod logs:
   ```bash
   kubectl logs <worker-pod> -n ${NAMESPACE}
   ```

2. Describe the restore target pod:
   ```bash
   kubectl describe pod <worker-pod> -n ${NAMESPACE}
   ```

3. Confirm the referenced checkpoint is still `Ready`:
   ```bash
   kubectl get dckpt <checkpoint-name> -n ${NAMESPACE}
   ```

## Planned Features

- TensorRT-LLM backend support
- S3/MinIO storage backend
- OCI registry storage backend
- Multi-GPU checkpoints

## Related Documentation

- [Dynamo Snapshot Helm Chart README](https://github.com/ai-dynamo/dynamo/blob/main/deploy/helm/charts/snapshot/README.md) - Chart configuration
- [Installation Guide](installation-guide.md) - Platform installation
- [API Reference](api-reference.md) - Complete CRD specifications
