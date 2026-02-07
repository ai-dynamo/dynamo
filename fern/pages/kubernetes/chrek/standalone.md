---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

> ⚠️ **Experimental Feature**: ChReK is currently in **beta/preview**. It requires privileged mode for restore operations, which may not be suitable for all production environments. Review the [security implications](#security-considerations) before deploying.

This guide explains how to use **ChReK** (Checkpoint/Restore for Kubernetes) as a standalone component without deploying the full Dynamo platform. This is useful if you want to add checkpoint/restore capabilities to your own GPU workloads.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Step 1: Deploy ChReK](#step-1-deploy-chrek)
- [Step 2: Build Checkpoint-Enabled Images](#step-2-build-checkpoint-enabled-images)
- [Step 3: Create Checkpoint Jobs](#step-3-create-checkpoint-jobs)
- [Step 4: Restore from Checkpoints](#step-4-restore-from-checkpoints)
- [Environment Variables Reference](#environment-variables-reference)
- [Checkpoint Flow Explained](#checkpoint-flow-explained)
- [Troubleshooting](#troubleshooting)

---

## Overview

When using ChReK standalone, you are responsible for:

1. **Deploying the ChReK Helm chart** (DaemonSet + PVC)
2. **Building checkpoint-enabled container images** with the restore entrypoint
3. **Creating checkpoint jobs** with the correct environment variables
4. **Creating restore pods** that detect and use the checkpoints

The ChReK DaemonSet handles the actual CRIU checkpoint/restore operations automatically once your pods are configured correctly.

---

## Prerequisites

- Kubernetes cluster with:
  - NVIDIA GPUs with checkpoint support
  - **Privileged security context allowed** (⚠️ required for CRIU - see [Security Considerations](#security-considerations))
  - PVC storage (ReadWriteMany recommended for multi-node)
- Docker or compatible container runtime for building images
- Access to the ChReK source code: `deploy/chrek/`

### Security Considerations

⚠️ **Important**: ChReK restore operations **require privileged mode**, which has significant security implications:

- **Privileged containers** can access all host devices and bypass most security restrictions
- This may violate security policies in production environments
- Privileged containers, if compromised, can potentially compromise node security

**Recommended for:**
- ✅ Development and testing environments
- ✅ Research and experimentation
- ✅ Controlled production environments with appropriate security controls

**Not recommended for:**
- ❌ Multi-tenant clusters without proper isolation
- ❌ Security-sensitive production workloads without risk assessment
- ❌ Environments with strict security compliance requirements

### Technical Limitations

⚠️ **Current Restrictions:**
- **vLLM backend only**: Currently only the vLLM backend supports checkpoint/restore. SGLang and TensorRT-LLM support is planned.
- **Single-node only**: Checkpoints must be created and restored on the same node
- **Single-GPU only**: Multi-GPU configurations are not yet supported
- **Network state**: Active TCP connections are closed during restore
- **Storage**: Only PVC backend currently implemented (S3/OCI planned)

---

## Step 1: Deploy ChReK

### Install the Helm Chart

```bash
# Clone the repository
git clone https://github.com/ai-dynamo/dynamo.git
cd dynamo

# Install ChReK in your namespace
helm install chrek ./deploy/helm/charts/chrek \
  --namespace my-app \
  --create-namespace \
  --set storage.pvc.size=100Gi \
  --set storage.pvc.storageClass=your-storage-class
```

### Verify Installation

```bash
# Check the DaemonSet is running
kubectl get daemonset -n my-app
# NAME          DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE
# chrek-agent   3         3         3       3            3

# Check the PVC is bound
kubectl get pvc -n my-app
# NAME        STATUS   VOLUME     CAPACITY   ACCESS MODES   STORAGECLASS
# chrek-pvc   Bound    pvc-xyz    100Gi      RWX            your-storage-class
```

---

## Step 2: Build Checkpoint-Enabled Images

ChReK provides a convenient `placeholder` target in its Dockerfile that automatically injects checkpoint/restore capabilities into your existing container images.

### Quick Start: Using the Placeholder Target (Recommended)

```bash
cd deploy/chrek

# Define your images
export BASE_IMAGE="your-app:latest"           # Your existing application image
export RESTORE_IMAGE="your-app:checkpoint-enabled"  # Output checkpoint-enabled image

# Build using the placeholder target
docker build \
  --target placeholder \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  -t "$RESTORE_IMAGE" \
  .

# Push to your registry
docker push "$RESTORE_IMAGE"
```

**Example with a Dynamo vLLM image:**

```bash
cd deploy/chrek

export DYNAMO_IMAGE="nvidia/dynamo-vllm:v1.2.0"
export RESTORE_IMAGE="nvidia/dynamo-vllm:v1.2.0-checkpoint"

docker build \
  --target placeholder \
  --build-arg BASE_IMAGE="$DYNAMO_IMAGE" \
  -t "$RESTORE_IMAGE" \
  .
```

### What the Placeholder Target Does

The ChReK Dockerfile's `placeholder` stage automatically:

- ✅ Builds the restore-entrypoint binary
- ✅ Injects it into `/usr/local/bin/restore-entrypoint`
- ✅ Adds `smart-entrypoint.sh` to `/usr/local/bin/`
- ✅ Sets executable permissions
- ✅ Configures the entrypoint to detect and restore checkpoints
- ✅ Preserves your original application CMD

### Alternative: Manual Multi-Stage Build

If you need more control, you can create your own Dockerfile:

```dockerfile
# Stage 1: Build restore-entrypoint
FROM golang:1.23-alpine AS restore-builder
WORKDIR /build
COPY deploy/chrek/cmd/restore-entrypoint ./cmd/restore-entrypoint
COPY deploy/chrek/pkg ./pkg
COPY deploy/chrek/go.mod deploy/chrek/go.sum ./

RUN go build -o /restore-entrypoint ./cmd/restore-entrypoint

# Stage 2: Your application image
FROM your-base-image:latest

# Copy restore-entrypoint
COPY --from=restore-builder /restore-entrypoint /usr/local/bin/restore-entrypoint

# Copy smart-entrypoint.sh
COPY deploy/chrek/scripts/smart-entrypoint.sh /usr/local/bin/smart-entrypoint.sh
RUN chmod +x /usr/local/bin/smart-entrypoint.sh /usr/local/bin/restore-entrypoint

# Set smart-entrypoint as the default entrypoint
ENTRYPOINT ["/usr/local/bin/smart-entrypoint.sh"]

# Your application command (becomes CMD, can be overridden)
CMD ["python", "your_app.py"]
```

> **💡 Tip**: Using the `placeholder` target is the recommended approach as it's maintained with the ChReK codebase and ensures compatibility.

---

## Step 3: Create Checkpoint Jobs

A checkpoint job loads your application, waits for the ChReK DaemonSet to checkpoint it, and then exits.

### Required Environment Variables

Your checkpoint job MUST set these environment variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `DYN_CHECKPOINT_SIGNAL_FILE` | Path where DaemonSet writes completion signal | `/checkpoint-signal/my-checkpoint.done` |
| `DYN_CHECKPOINT_READY_FILE` | Path where your app signals it's ready | `/tmp/checkpoint-ready` |
| `DYN_CHECKPOINT_HASH` | Unique identifier for this checkpoint | `abc123def456` |
| `DYN_CHECKPOINT_LOCATION` | Directory where checkpoint is stored | `/checkpoints/abc123def456` |
| `DYN_CHECKPOINT_STORAGE_TYPE` | Storage backend type | `pvc` |

### Required Labels

Add this label to enable DaemonSet checkpoint detection:

```yaml
labels:
  nvidia.com/checkpoint-source: "true"
```

### Example Checkpoint Job

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: checkpoint-my-model
  namespace: my-app
spec:
  template:
    metadata:
      labels:
        nvidia.com/checkpoint-source: "true"  # Required for DaemonSet detection
    spec:
      restartPolicy: Never

      # Init container to clean up stale signal files
      initContainers:
      - name: cleanup-signal-file
        image: busybox:latest
        command:
        - sh
        - -c
        - |
          rm -f /checkpoint-signal/my-checkpoint.done || true
          echo "Signal file cleanup complete"
        volumeMounts:
        - name: checkpoint-signal
          mountPath: /checkpoint-signal

      containers:
      - name: main
        image: my-app:checkpoint-enabled

        # Security context required for CRIU
        securityContext:
          privileged: true
          capabilities:
            add: ["SYS_ADMIN", "SYS_PTRACE", "SYS_CHROOT"]

        # Readiness probe: Pod becomes Ready when model is loaded
        # This is what triggers the DaemonSet to start checkpointing
        readinessProbe:
          exec:
            command: ["sh", "-c", "cat ${DYN_CHECKPOINT_READY_FILE}"]
          initialDelaySeconds: 15
          periodSeconds: 2

        # Remove liveness/startup probes for checkpoint jobs
        # Model loading can take several minutes
        livenessProbe: null
        startupProbe: null

        # Checkpoint-related environment variables
        env:
        - name: DYN_CHECKPOINT_SIGNAL_FILE
          value: "/checkpoint-signal/my-checkpoint.done"
        - name: DYN_CHECKPOINT_READY_FILE
          value: "/tmp/checkpoint-ready"
        - name: DYN_CHECKPOINT_HASH
          value: "abc123def456"
        - name: DYN_CHECKPOINT_LOCATION
          value: "/checkpoints/abc123def456"
        - name: DYN_CHECKPOINT_STORAGE_TYPE
          value: "pvc"

        # GPU request
        resources:
          limits:
            nvidia.com/gpu: 1

        # Required volume mounts
        volumeMounts:
        - name: checkpoint-storage
          mountPath: /checkpoints
        - name: checkpoint-signal
          mountPath: /checkpoint-signal
        - name: tmp
          mountPath: /tmp

      volumes:
      - name: checkpoint-storage
        persistentVolumeClaim:
          claimName: chrek-pvc
      - name: checkpoint-signal
        hostPath:
          path: /var/lib/chrek/signals
          type: DirectoryOrCreate
      - name: tmp
        emptyDir: {}
```

### Application Code Requirements

Your application must implement the checkpoint flow. Here's the pattern used by Dynamo vLLM:

```python
import os
import time

def main():
    # 1. Check for checkpoint mode
    signal_file = os.environ.get("DYN_CHECKPOINT_SIGNAL_FILE")
    ready_file = os.environ.get("DYN_CHECKPOINT_READY_FILE")
    restore_marker = os.environ.get("DYN_RESTORE_MARKER_FILE", "/tmp/dynamo-restored")

    is_checkpoint_mode = signal_file is not None

    if is_checkpoint_mode:
        print("Checkpoint mode detected")

        # 2. Load your model/application
        model = load_model()

        # 3. Optional: Put model to sleep to reduce memory footprint
        # model.sleep()

        # 4. Write ready file (for application use, not DaemonSet)
        if ready_file:
            with open(ready_file, "w") as f:
                f.write("ready")
            print(f"Wrote checkpoint ready file: {ready_file}")

        # 5. Log readiness messages (helps debugging)
        print("CHECKPOINT_READY: Model loaded, ready for container checkpoint")
        print(f"CHECKPOINT_READY: Waiting for signal file: {signal_file}")
        print(f"CHECKPOINT_READY: Or restore marker file: {restore_marker}")

        # 6. Wait for checkpoint completion OR restore detection
        while True:
            # Check if we've been restored (marker file created by restore entrypoint)
            if os.path.exists(restore_marker):
                print(f"Detected restore from checkpoint (marker: {restore_marker})")
                # Continue with normal application flow
                break

            # Check if checkpoint is complete (signal file created by DaemonSet)
            if os.path.exists(signal_file):
                print(f"Checkpoint signal file detected: {signal_file}")
                print("Checkpoint complete, exiting")
                return  # Exit gracefully

            time.sleep(1)

    # Normal application flow (or post-restore flow)
    run_application()
```

**Important Notes:**

1. **Ready File & Readiness Probe**: The checkpoint job must have a readiness probe that checks for the ready file:
   ```yaml
   readinessProbe:
     exec:
       command: ["sh", "-c", "cat ${DYN_CHECKPOINT_READY_FILE}"]
     initialDelaySeconds: 15
     periodSeconds: 2
   ```
   The ChReK DaemonSet triggers checkpointing when:
   - Pod has `nvidia.com/checkpoint-source: "true"` label
   - Pod status is `Ready` (readiness probe passes = ready file exists)

2. **Restore Marker**: Created by `restore-entrypoint` before CRIU restore, allows the restored process to detect it was restored

3. **Two Exit Paths**:
   - **Signal file found**: Checkpoint complete, exit gracefully
   - **Restore marker found**: Process was restored, continue running


---

## Step 4: Restore from Checkpoints

Restore pods automatically detect and restore from checkpoints if they exist.

### Example Restore Pod

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-app-restored
  namespace: my-app
spec:
  restartPolicy: Never

  containers:
  - name: main
    image: my-app:checkpoint-enabled

    # Security context required for CRIU restore
    securityContext:
      privileged: true
      capabilities:
        add: ["SYS_ADMIN", "SYS_PTRACE", "SYS_CHROOT"]

    # Set checkpoint environment variables
    env:
    - name: DYN_CHECKPOINT_HASH
      value: "abc123def456"  # Must match checkpoint job
    - name: DYN_CHECKPOINT_PATH
      value: "/checkpoints"  # Base path (hash appended automatically)

    # Optional: Customize restore marker file path
    # - name: DYN_RESTORE_MARKER_FILE
    #   value: "/tmp/dynamo-restored"

    # GPU request
    resources:
      limits:
        nvidia.com/gpu: 1

    # Mount checkpoint storage (READ-ONLY is fine for restore)
    volumeMounts:
    - name: checkpoint-storage
      mountPath: /checkpoints
      readOnly: true
    - name: checkpoint-signal
      mountPath: /checkpoint-signal

  volumes:
  - name: checkpoint-storage
    persistentVolumeClaim:
      claimName: chrek-pvc
  - name: checkpoint-signal
    hostPath:
      path: /var/lib/chrek/signals
      type: DirectoryOrCreate
```

### How Restore Works

1. **Smart Entrypoint Detects Checkpoint**: The `smart-entrypoint.sh` checks if a checkpoint exists at `/checkpoints/${DYN_CHECKPOINT_HASH}/`
2. **Calls Restore Entrypoint**: If found, calls `/usr/local/bin/restore-entrypoint` which invokes CRIU
3. **CRIU Restores Process**: The entire process tree is restored from the checkpoint, including GPU state
4. **Application Continues**: Your application resumes exactly where it was checkpointed

---

## Environment Variables Reference

### Checkpoint Jobs

| Variable | Required | Description |
|----------|----------|-------------|
| `DYN_CHECKPOINT_SIGNAL_FILE` | Yes | Full path to signal file (e.g., `/checkpoint-signal/my-checkpoint.done`) |
| `DYN_CHECKPOINT_READY_FILE` | Yes | Full path where app signals readiness (e.g., `/tmp/checkpoint-ready`) |
| `DYN_CHECKPOINT_HASH` | Yes | Unique checkpoint identifier (alphanumeric string) |
| `DYN_CHECKPOINT_LOCATION` | Yes | Directory where checkpoint is stored (e.g., `/checkpoints/abc123`) |
| `DYN_CHECKPOINT_STORAGE_TYPE` | Yes | Storage backend: `pvc`, `s3`, or `oci` |

### Restore Pods

| Variable | Required | Description |
|----------|----------|-------------|
| `DYN_CHECKPOINT_HASH` | Yes | Checkpoint identifier (must match checkpoint job) |
| `DYN_CHECKPOINT_PATH` | Yes | Base checkpoint directory (hash appended automatically) |
| `DYN_RESTORE_MARKER_FILE` | No | Path for restore marker file (default: `/tmp/dynamo-restored`) |

### Optional CRIU Tuning (Advanced)

| Variable | Default | Description |
|----------|---------|-------------|
| `CRIU_TIMEOUT` | `0` (unlimited) | CRIU operation timeout in seconds |
| `CRIU_LOG_LEVEL` | `4` | CRIU log verbosity (0-4) |
| `CRIU_WORK_DIR` | `/tmp` | CRIU working directory |
| `CUDA_PLUGIN_DIR` | `/usr/local/lib/criu` | Path to CRIU CUDA plugin |
| `CRIU_SKIP_IN_FLIGHT` | `false` | Skip in-flight TCP connections |
| `CRIU_AUTO_DEDUP` | `false` | Enable auto-deduplication |
| `CRIU_LAZY_PAGES` | `false` | Enable lazy page migration (experimental) |
| `WAIT_FOR_CHECKPOINT` | `false` | Wait for checkpoint to appear before starting |
| `RESTORE_WAIT_TIMEOUT` | `300` | Max seconds to wait for checkpoint |
| `DEBUG` | `false` | Enable debug mode (sleeps 300s on error) |

---

## Checkpoint Flow Explained

### 1. Checkpoint Creation Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Pod starts with nvidia.com/checkpoint-source=true label  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Application loads model and creates ready file           │
│    /tmp/checkpoint-ready                                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Pod becomes Ready (kubelet readiness probe passes)       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. ChReK DaemonSet detects:                                 │
│    - Pod is Ready                                            │
│    - Has checkpoint-source label                             │
│    - Ready file exists: /tmp/checkpoint-ready               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. DaemonSet executes CRIU checkpoint via runc:             │
│    - Freezes container process                               │
│    - Dumps memory (CPU + GPU)                                │
│    - Saves to /checkpoints/${HASH}/                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. DaemonSet writes signal file:                            │
│    /checkpoint-signal/${HASH}.done                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. Application detects signal file and exits gracefully     │
└─────────────────────────────────────────────────────────────┘
```

### 2. Restore Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Pod starts with DYN_CHECKPOINT_HASH set                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. smart-entrypoint.sh checks for checkpoint:               │
│    /checkpoints/${DYN_CHECKPOINT_HASH}/checkpoint.done      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ├─ Not Found ─────────────────┐
                       │                              │
                       ▼                              ▼
           ┌───────────────────────┐    ┌──────────────────────┐
           │ Checkpoint exists     │    │ Cold start           │
           └──────────┬────────────┘    │ Run original CMD     │
                      │                 └──────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Call restore-entrypoint with checkpoint path             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. restore-entrypoint extracts checkpoint and calls CRIU:   │
│    criu restore --images-dir /checkpoints/${HASH}/images    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. CRIU restores process from checkpoint                    │
│    - Restores memory (CPU + GPU)                             │
│    - Restores file descriptors                               │
│    - Resumes process execution                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Application continues from checkpointed state            │
│    (Model already loaded, GPU memory initialized)           │
└─────────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

### Checkpoint Not Created

**Symptom**: Job runs but no checkpoint appears in `/checkpoints/`

**Checks**:
1. Verify the pod has the label:
   ```bash
   kubectl get pod <pod-name> -o jsonpath='{.metadata.labels.nvidia\.com/checkpoint-source}'
   ```

2. Check pod readiness:
   ```bash
   kubectl get pod <pod-name> -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}'
   ```

3. Check ready file was created:
   ```bash
   kubectl exec <pod-name> -- ls -la /tmp/checkpoint-ready
   ```

4. Check DaemonSet logs:
   ```bash
   kubectl logs -n my-app daemonset/chrek-agent --all-containers
   ```

### Restore Fails

**Symptom**: Pod fails to restore from checkpoint

**Checks**:
1. Verify checkpoint files exist:
   ```bash
   kubectl exec <pod-name> -- ls -la /checkpoints/${DYN_CHECKPOINT_HASH}/
   ```

2. Check privileged mode is enabled:
   ```bash
   kubectl get pod <pod-name> -o jsonpath='{.spec.containers[0].securityContext.privileged}'
   ```

3. Check CRIU logs in `/tmp/criu-restore.log`:
   ```bash
   kubectl exec <pod-name> -- cat /tmp/criu-restore.log
   ```

4. Ensure checkpoint and restore have same:
   - Container image
   - GPU count
   - Volume mounts
   - Environment variables (except POD_NAME, POD_IP, etc.)

### Permission Denied Errors

**Symptom**: `CRIU: Permission denied` or `Operation not permitted`

**Solution**: Ensure pod has:
```yaml
securityContext:
  privileged: true
  capabilities:
    add:
    - SYS_ADMIN
    - SYS_PTRACE
    - SYS_CHROOT
```

### Signal File Not Appearing

**Symptom**: Application waits forever for signal file

**Checks**:
1. Verify hostPath mount is correct:
   ```bash
   kubectl get pod <pod-name> -o jsonpath='{.spec.volumes[?(@.name=="checkpoint-signal")]}'
   ```

2. Check DaemonSet has access to the same path:
   ```bash
   kubectl get daemonset -n my-app chrek-agent -o jsonpath='{.spec.template.spec.volumes[?(@.name=="signal-dir")]}'
   ```

3. Verify paths match exactly:
   - Pod: `/var/lib/chrek/signals`
   - DaemonSet: `/var/lib/chrek/signals`

---

## Additional Resources

- [ChReK Helm Chart Values](https://github.com/ai-dynamo/dynamo/tree/main/deploy/helm/charts/chrek/values.yaml)
- [Smart Entrypoint Script](https://github.com/ai-dynamo/dynamo/tree/main/deploy/chrek/scripts/smart-entrypoint.sh)
- [CRIU Documentation](https://criu.org/Main_Page)
- [CUDA Checkpoint Plugin](https://docs.nvidia.com/cuda/cuda-checkpoint-plugin/)

---

## Getting Help

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review DaemonSet logs: `kubectl logs -n <namespace> daemonset/chrek-agent`
3. Open an issue on [GitHub](https://github.com/ai-dynamo/dynamo/issues)
