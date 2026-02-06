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

> ⚠️ **Experimental Feature**: ChReK is currently in **beta/preview**. It requires privileged mode for restore operations. See [Limitations](#limitations) for details.

Reduce cold start times for LLM inference workers from ~3 minutes to ~30 seconds using container checkpointing.

## Overview

Checkpointing captures the complete state of a running worker pod (including GPU memory) and saves it to storage. New pods can restore from this checkpoint instead of performing a full cold start.

| Startup Type | Time | What Happens |
|--------------|------|--------------|
| **Cold Start** | ~3 min | Download model, load to GPU, initialize engine |
| **Warm Start** (checkpoint) | ~30 sec | Restore from checkpoint tar |

## Prerequisites

- Dynamo Platform installed (v0.4.0+)
- ChReK Helm chart installed (separate from platform)
- GPU nodes with CRIU support
- RWX PVC storage (PVC is currently the only supported backend)

## Quick Start

### 1. Install ChReK Infrastructure

First, install the ChReK Helm chart in each namespace where you need checkpointing:

```bash
# Install ChReK infrastructure
helm install chrek nvidia/chrek \
  --namespace my-team \
  --create-namespace \
  --set storage.pvc.size=100Gi
```

This creates:
- A PVC for checkpoint storage (`chrek-pvc`)
- A DaemonSet for CRIU operations (`chrek-agent`)

### 2. Configure Operator Values

Update your Helm values to point to the ChReK infrastructure:

```yaml
# values.yaml
dynamo-operator:
  checkpoint:
    enabled: true
    storage:
      type: pvc  # Only PVC is currently supported (S3/OCI planned)
      pvc:
        pvcName: "chrek-pvc"  # Must match ChReK chart
        basePath: "/checkpoints"
      signalHostPath: "/var/lib/chrek/signals"  # Must match ChReK chart
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
      replicas: 1
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
          backendFramework: "vllm"
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

### PVC (Currently Supported)

Use when you have RWX storage available (e.g., NFS, EFS, Filestore).

```yaml
checkpoint:
  storage:
    type: pvc
    pvc:
      pvcName: "chrek-pvc"
      basePath: "/checkpoints"
```

**Requirements:**
- RWX (ReadWriteMany) PVC for multi-node access
- Sufficient storage (checkpoints are ~10-50GB per model)

### S3 / MinIO (Planned - Not Yet Implemented)

> ⚠️ **Note:** S3 storage backend is defined in the API but not yet fully implemented.

Object storage support is planned for a future release. The configuration will look like:

```yaml
checkpoint:
  storage:
    type: s3  # Not yet supported
    s3:
      # AWS S3
      uri: "s3://my-bucket/checkpoints"

      # Or MinIO / custom S3
      uri: "s3://minio.example.com/my-bucket/checkpoints"

      # Optional: credentials secret
      credentialsSecretRef: "s3-creds"
```

### OCI Registry (Planned - Not Yet Implemented)

> ⚠️ **Note:** OCI registry storage backend is defined in the API but not yet fully implemented.

Container registry storage support is planned for a future release. The configuration will look like:

```yaml
checkpoint:
  storage:
    type: oci  # Not yet supported
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
    backendFramework: "vllm"
    tensorParallelSize: 1
```

### Reference Mode

Reference an existing `DynamoCheckpoint` CR by its 16-character hash using `checkpointRef`:

```yaml
checkpoint:
  enabled: true
  checkpointRef: "e5962d34ba272638"  # 16-char hash of DynamoCheckpoint CR
```

This is useful when:
- You want to **pre-warm checkpoints** before creating DGDs
- You want to **explicit control** over which checkpoint to use

**Flow:**
1. Create a `DynamoCheckpoint` CR (see [DynamoCheckpoint CRD](#dynamocheckpoint-crd) section)
2. Wait for it to become `Ready`
3. Reference it in your DGD using `checkpointRef` with the hash

```bash
# Check checkpoint status (using 16-char hash name)
kubectl get dynamocheckpoint e5962d34ba272638 -n dynamo-system
NAME                MODEL                   BACKEND  PHASE  HASH              AGE
e5962d34ba272638    meta-llama/Llama-3-8B  vllm     Ready  e5962d34ba272638  5m

# Now create DGD referencing it
kubectl apply -f my-dgd.yaml
```

## Checkpoint Identity

Checkpoints are uniquely identified by a **16-character SHA256 hash** (64 bits) of configuration that affects runtime state:

| Field | Required | Affects Hash | Example |
|-------|----------|-------------|---------|
| `model` | ✓ | ✓ | `meta-llama/Llama-3-8B` |
| `framework` | ✓ | ✓ | `vllm`, `sglang`, `trtllm` |
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
  mode: auto
  identity:
    model: "meta-llama/Llama-3-8B"
    backendFramework: "vllm"
    dynamoVersion: "0.9.0"
    tensorParallelSize: 1
    pipelineParallelSize: 1
    dtype: "bfloat16"
    maxModelLen: 8192
    extraParameters:
      enableChunkedPrefill: "true"
      quantization: "awq"
```

**Checkpoint Naming:** The `DynamoCheckpoint` CR is automatically named using the 16-character identity hash (e.g., `e5962d34ba272638`).

**Checkpoint Sharing:** Multiple DGDs with the same identity automatically share the same checkpoint.

## DynamoCheckpoint CRD

The `DynamoCheckpoint` (shortname: `dckpt`) is a Kubernetes Custom Resource that manages checkpoint lifecycle.

**When to create a DynamoCheckpoint directly:**
- **Pre-warming:** Create checkpoints before deploying DGDs for instant startup
- **Explicit control:** Manage checkpoint lifecycle independently from DGDs

**Note:** With the new hash-based naming, checkpoint names are automatically generated (16-character hash). The operator handles checkpoint discovery and reuse automatically in `auto` mode.

**Create a checkpoint:**

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoCheckpoint
metadata:
  name: e5962d34ba272638  # Use the computed 16-char hash
spec:
  identity:
    model: "meta-llama/Llama-3-8B"
    backendFramework: "vllm"
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

**Note:** You can compute the hash yourself, or use `auto` mode to let the operator create it.

**Check status:**

```bash
# List all checkpoints
kubectl get dynamocheckpoint -n dynamo-system
# Or use shortname
kubectl get dckpt -n dynamo-system

NAME                MODEL                          BACKEND  PHASE    HASH              AGE
e5962d34ba272638    meta-llama/Llama-3-8B         vllm     Ready    e5962d34ba272638  5m
a7b4f89c12de3456    meta-llama/Llama-3-70B        vllm     Creating a7b4f89c12de3456  2m
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
kubectl describe dckpt e5962d34ba272638 -n dynamo-system
```

```yaml
Status:
  Phase: Ready
  IdentityHash: e5962d34ba272638
  Location: /checkpoints/e5962d34ba272638
  StorageType: pvc
  CreatedAt: 2026-01-29T10:05:00Z
```

**Reference from DGD:**

Once the checkpoint is `Ready`, you can reference it by hash:

```yaml
spec:
  services:
    VllmWorker:
      checkpoint:
        enabled: true
        checkpointRef: "e5962d34ba272638"  # 16-char hash
```

Or use `auto` mode and the operator will find/create it automatically.

## Limitations

⚠️ **Important**: ChReK has significant limitations that impact production readiness:

### Security Considerations
- **🔴 Privileged mode required**: Restore pods **must run in privileged mode** for CRIU to function
- Privileged containers have elevated host access, which may violate security policies in many production environments
- This requirement applies to all worker pods that restore from checkpoints

### Technical Limitations
- **vLLM backend only**: Currently only the vLLM backend supports checkpoint/restore. SGLang and TensorRT-LLM support is planned.
- **Single-node only**: Checkpoints must be created and restored on the same node
- **Single-GPU only**: Multi-GPU configurations are not yet supported
- **Network state**: Active TCP connections are closed during restore (handled with `tcp-close` CRIU option)
- **Storage**: Only PVC backend currently implemented (S3/OCI planned)

### Recommendation
ChReK is **experimental/beta** and best suited for:
- ✅ Development and testing environments
- ✅ Research and experimentation
- ✅ Controlled production environments with appropriate security controls
- ❌ Security-sensitive production workloads without proper risk assessment

## Troubleshooting

### Checkpoint Not Creating

1. Check the checkpoint job:
   ```bash
   kubectl get jobs -l nvidia.com/checkpoint-source=true -n dynamo-system
   kubectl logs job/checkpoint-<name> -n dynamo-system
   ```

2. Check the DaemonSet:
   ```bash
   kubectl logs daemonset/chrek-agent -n dynamo-system
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
   kubectl exec <worker-pod> -- env | grep DYN_CHECKPOINT
   ```

### Cold Start Despite Checkpoint

Pods fall back to cold start if:
- Checkpoint file doesn't exist yet (still being created)
- Checkpoint file is corrupted
- CRIU restore fails

Check logs for "Falling back to cold start" message.

## Best Practices

1. **Use RWX PVCs** for multi-node deployments (currently the only supported backend)
2. **Pre-warm checkpoints** before scaling up
3. **Monitor checkpoint size** - large models create large checkpoints
4. **Clean up old checkpoints** to save storage

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DYN_CHECKPOINT_STORAGE_TYPE` | Backend: `pvc`, `s3`, `oci` |
| `DYN_CHECKPOINT_LOCATION` | Source location (URI) |
| `DYN_CHECKPOINT_PATH` | Local path to tar file |
| `DYN_CHECKPOINT_HASH` | Identity hash (debugging) |
| `DYN_CHECKPOINT_SIGNAL_FILE` | Signal file (creation mode only) |

## Complete Example

Create a checkpoint and use it in a DGD:

```yaml
# 1. Create the DynamoCheckpoint CR
apiVersion: nvidia.com/v1alpha1
kind: DynamoCheckpoint
metadata:
  name: e5962d34ba272638  # 16-char hash (computed from identity)
  namespace: dynamo-system
spec:
  identity:
    model: "meta-llama/Meta-Llama-3-8B-Instruct"
    backendFramework: "vllm"
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
# 2. Wait for Ready: kubectl get dckpt e5962d34ba272638 -n dynamo-system -w
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
        checkpointRef: "e5962d34ba272638"  # Reference by hash
```

## Related Documentation

- [ChReK Overview](README.md) - ChReK architecture and use cases
- [ChReK Standalone Usage Guide](standalone.md) - Use ChReK without Dynamo Platform
- ChReK Helm Chart README - See `deploy/helm/charts/chrek/README.md` in the repository for chart configuration
- [Installation Guide](../installation-guide.md) - Platform installation
- [API Reference](../api-reference.md) - Complete CRD specifications

