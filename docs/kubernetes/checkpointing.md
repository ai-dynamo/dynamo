<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Checkpoint/Restore for Fast Pod Startup

Reduce cold start times for LLM inference workers from ~3 minutes to ~30 seconds using container checkpointing.

## Overview

Checkpointing captures the complete state of a running worker pod (including GPU memory) and saves it to storage. New pods can restore from this checkpoint instead of performing a full cold start.

| Startup Type | Time | What Happens |
|--------------|------|--------------|
| **Cold Start** | ~3 min | Download model, load to GPU, initialize engine |
| **Warm Start** (checkpoint) | ~30 sec | Restore from checkpoint tar |

## Prerequisites

- Dynamo Platform installed (v0.4.0+)
- GPU nodes with CRIU support
- Storage backend configured (PVC, S3, or OCI registry)

## Quick Start

### 1. Enable Checkpointing

Update your Helm values:

```yaml
# values.yaml
dynamo-operator:
  checkpoint:
    enabled: true
    storage:
      type: pvc  # or s3, oci
      pvc:
        pvcName: "checkpoint-storage"
        basePath: "/checkpoints"
```

Create the PVC (for PVC storage type):

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: checkpoint-storage
spec:
  accessModes:
    - ReadWriteMany  # Required for multi-node access
  resources:
    requests:
      storage: 100Gi
```

### 2. Configure Your DGD

Add checkpoint configuration to your service:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: my-llm
spec:
  services:
    VllmWorker:
      replicas: 2
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/dynamo-vllm:latest
          args:
            - python3 -m dynamo.vllm --model meta-llama/Llama-3-8B
      resources:
        limits:
          nvidia.com/gpu: "1"

      # Checkpoint configuration
      checkpoint:
        enabled: true
        mode: auto  # Automatically create checkpoint if not found
        identity:
          model: "meta-llama/Llama-3-8B"
          framework: "vllm"
          tensorParallelSize: 1
          dtype: "bfloat16"
```

### 3. Deploy

```bash
kubectl apply -f my-llm.yaml -n dynamo-system
```

On first deployment:
1. A checkpoint job runs to create the checkpoint
2. Worker pods start with cold start (checkpoint not ready yet)
3. Once checkpoint is ready, new pods (scale-up, restarts) restore from checkpoint

## Storage Backends

### PVC (Default)

Use when you have RWX storage available (e.g., NFS, EFS, Filestore).

```yaml
checkpoint:
  storage:
    type: pvc
    pvc:
      pvcName: "checkpoint-storage"
      basePath: "/checkpoints"
```

**Requirements:**
- RWX (ReadWriteMany) PVC for multi-node access
- Sufficient storage (checkpoints are ~10-50GB per model)

### S3 / MinIO

Use when RWX storage is not available or you prefer object storage.

```yaml
checkpoint:
  storage:
    type: s3
    s3:
      # AWS S3
      uri: "s3://my-bucket/checkpoints"

      # Or MinIO / custom S3
      uri: "s3://minio.example.com/my-bucket/checkpoints"

      # Optional: credentials secret
      credentialsSecretRef: "s3-creds"
```

**Credentials Secret (if not using IRSA/Workload Identity):**

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: s3-creds
stringData:
  AWS_ACCESS_KEY_ID: "AKIA..."
  AWS_SECRET_ACCESS_KEY: "..."
  AWS_REGION: "us-west-2"  # Optional
```

### OCI Registry

Use when you want to leverage existing container registry infrastructure.

```yaml
checkpoint:
  storage:
    type: oci
    oci:
      uri: "oci://myregistry.io/checkpoints"
      credentialsSecretRef: "registry-creds"  # Docker config secret
```

## Checkpoint Modes

### Auto Mode (Recommended)

The operator automatically creates a `DynamoCheckpoint` CR if one doesn't exist:

```yaml
checkpoint:
  enabled: true
  mode: auto
  identity:
    model: "meta-llama/Llama-3-8B"
    framework: "vllm"
    tensorParallelSize: 1
```

### Reference Mode

Reference an existing `DynamoCheckpoint` CR by name using `checkpointRef`:

```yaml
checkpoint:
  enabled: true
  checkpointRef: "llama3-8b-vllm-tp1"  # Name of DynamoCheckpoint CR
```

This is useful when:
- You want to **pre-warm checkpoints** before creating DGDs
- You want to **share checkpoints** across multiple DGDs explicitly
- You created a checkpoint manually using the DynamoCheckpoint CRD

**Flow:**
1. Create a `DynamoCheckpoint` CR (see [DynamoCheckpoint CRD](#dynamocheckpoint-crd) section)
2. Wait for it to become `Ready`
3. Reference it in your DGD using `checkpointRef`

```bash
# Check checkpoint status
kubectl get dynamocheckpoint llama3-8b-vllm-tp1 -n dynamo-system
NAME                  PHASE   HASH           AGE
llama3-8b-vllm-tp1    Ready   abc123def456   5m

# Now create DGD referencing it
kubectl apply -f my-dgd.yaml
```

## Checkpoint Identity

Checkpoints are uniquely identified by a hash of configuration that affects runtime state:

| Field | Required | Affects Hash | Example |
|-------|----------|-------------|---------|
| `model` | ✓ | ✓ | `meta-llama/Llama-3-8B` |
| `framework` | ✓ | ✓ | `vllm`, `sglang`, `trtllm` |
| `frameworkVersion` | | ✓ | `0.4.0`, `0.5.1` |
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
  mode: auto
  identity:
    model: "meta-llama/Llama-3-8B"
    framework: "vllm"
    frameworkVersion: "0.4.0"
    tensorParallelSize: 1
    pipelineParallelSize: 1
    dtype: "bfloat16"
    maxModelLen: 8192
    extraParameters:
      enableChunkedPrefill: "true"
      quantization: "awq"
```

**Checkpoint Sharing:** Multiple DGDs with the same identity automatically share the same checkpoint.

## DynamoCheckpoint CRD

The `DynamoCheckpoint` (shortname: `dckpt`) is a Kubernetes Custom Resource that manages checkpoint lifecycle.

**When to create a DynamoCheckpoint directly:**
- **Pre-warming:** Create checkpoints before deploying DGDs for instant startup
- **Explicit control:** Manage checkpoint lifecycle independently from DGDs
- **Sharing:** Create a single checkpoint and reference it from multiple DGDs

**Create a checkpoint:**

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoCheckpoint
metadata:
  name: llama3-8b-vllm-tp1
spec:
  identity:
    model: "meta-llama/Llama-3-8B"
    framework: "vllm"
    tensorParallelSize: 1
    dtype: "bfloat16"

  job:
    activeDeadlineSeconds: 3600
    podTemplateSpec:
      spec:
        containers:
          - name: main
            image: nvcr.io/nvidia/ai-dynamo/dynamo-vllm:latest
            command: ["python3", "-m", "dynamo.vllm"]
            args: ["--model", "meta-llama/Llama-3-8B"]
            resources:
              limits:
                nvidia.com/gpu: "1"
            env:
              - name: HF_TOKEN
                valueFrom:
                  secretKeyRef:
                    name: hf-token-secret
                    key: HF_TOKEN
```

**Check status:**

```bash
# List all checkpoints
kubectl get dynamocheckpoint -n dynamo-system
# Or use shortname
kubectl get dckpt -n dynamo-system

NAME                  PHASE      HASH           AGE
llama3-8b-vllm-tp1    Ready      abc123def456   5m
llama3-70b-vllm-tp4   Creating   xyz789abc012   2m
```

**Phases:**
| Phase | Description |
|-------|-------------|
| `Pending` | CR created, waiting for job to start |
| `Creating` | Checkpoint job is running |
| `Ready` | Checkpoint available for use |
| `Failed` | Checkpoint creation failed |

**Detailed status:**

```bash
kubectl describe dckpt llama3-8b-vllm-tp1 -n dynamo-system
```

```yaml
Status:
  Phase: Ready
  IdentityHash: abc123def456
  Location: /checkpoints/abc123def456.tar  # or s3://bucket/...
  StorageType: pvc
  CreatedAt: 2024-12-11T10:05:00Z
```

**Reference from DGD:**

Once the checkpoint is `Ready`, reference it in your DGD:

```yaml
spec:
  services:
    VllmWorker:
      checkpoint:
        enabled: true
        checkpointRef: "llama3-8b-vllm-tp1"
```

## Troubleshooting

### Checkpoint Not Creating

1. Check the checkpoint job:
   ```bash
   kubectl get jobs -l nvidia.com/checkpoint-source=true -n dynamo-system
   kubectl logs job/checkpoint-<name> -n dynamo-system
   ```

2. Check the DaemonSet:
   ```bash
   kubectl logs daemonset/dynamo-checkpoint-agent -n dynamo-system
   ```

3. Verify storage access:
   ```bash
   kubectl exec -it <checkpoint-agent-pod> -- ls -la /checkpoints
   ```

### Restore Failing

1. Check pod logs:
   ```bash
   kubectl logs <worker-pod> -n dynamo-system
   ```

2. Verify checkpoint file exists:
   ```bash
   # For PVC
   kubectl exec -it <any-pod-with-pvc> -- ls -la /checkpoints/

   # For S3
   aws s3 ls s3://my-bucket/checkpoints/
   ```

3. Check environment variables:
   ```bash
   kubectl exec <worker-pod> -- env | grep DYNAMO_CHECKPOINT
   ```

### Cold Start Despite Checkpoint

Pods fall back to cold start if:
- Checkpoint file doesn't exist yet (still being created)
- Checkpoint file is corrupted
- CRIU restore fails

Check logs for "Falling back to cold start" message.

## Best Practices

1. **Use RWX PVCs** for multi-node deployments
2. **Pre-warm checkpoints** before scaling up
3. **Monitor checkpoint size** - large models create large checkpoints
4. **Use S3/OCI** when RWX storage is not available
5. **Clean up old checkpoints** to save storage

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DYNAMO_CHECKPOINT_STORAGE_TYPE` | Backend: `pvc`, `s3`, `oci` |
| `DYNAMO_CHECKPOINT_LOCATION` | Source location (URI) |
| `DYNAMO_CHECKPOINT_PATH` | Local path to tar file |
| `DYNAMO_CHECKPOINT_HASH` | Identity hash (debugging) |
| `DYNAMO_CHECKPOINT_SIGNAL_FILE` | Signal file (creation mode only) |

## Complete Example

Create a checkpoint and use it in a DGD:

```yaml
# 1. Create the DynamoCheckpoint CR
apiVersion: nvidia.com/v1alpha1
kind: DynamoCheckpoint
metadata:
  name: vllm-llama3-8b
  namespace: dynamo-system
spec:
  identity:
    model: "meta-llama/Meta-Llama-3-8B-Instruct"
    framework: "vllm"
    tensorParallelSize: 1
    dtype: "bfloat16"
  job:
    activeDeadlineSeconds: 3600
    backoffLimit: 3
    podTemplateSpec:
      spec:
        containers:
          - name: main
            image: nvcr.io/nvidia/ai-dynamo/dynamo-vllm:latest
            command: ["python3", "-m", "dynamo.vllm"]
            args:
              - "--model"
              - "meta-llama/Meta-Llama-3-8B-Instruct"
              - "--tensor-parallel-size"
              - "1"
              - "--dtype"
              - "bfloat16"
            env:
              - name: HF_TOKEN
                valueFrom:
                  secretKeyRef:
                    name: hf-token-secret
                    key: HF_TOKEN
            resources:
              limits:
                nvidia.com/gpu: "1"
        restartPolicy: Never
---
# 2. Wait for Ready: kubectl get dckpt vllm-llama3-8b -n dynamo-system -w
---
# 3. Reference the checkpoint in your DGD
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: my-llm
  namespace: dynamo-system
spec:
  services:
    VllmWorker:
      replicas: 2
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/dynamo-vllm:latest
      resources:
        limits:
          nvidia.com/gpu: "1"
      checkpoint:
        enabled: true
        checkpointRef: "vllm-llama3-8b"  # Reference the checkpoint above
```

## Related Documentation

- [Installation Guide](./installation_guide.md) - Platform installation
- [API Reference](./api_reference.md) - Complete CRD specifications

