---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Standalone Usage
---

> ⚠️ **Experimental Feature**: ChReK is currently in **beta/preview**. The ChReK DaemonSet runs in privileged mode to perform CRIU operations. Review the [security implications](#security-considerations) before deploying.

This guide explains how to use **ChReK** (Checkpoint/Restore for Kubernetes) as a standalone component without deploying the full Dynamo platform. This is useful if you want to add checkpoint/restore capabilities to your own GPU workloads.

## Table of Contents

- [Overview](#overview)
- [Using ChReK Without the Dynamo Operator](#using-chrek-without-the-dynamo-operator)
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
2. **Building checkpoint-enabled container images** with the CRIU runtime dependencies
3. **Creating checkpoint jobs** with the correct environment variables
4. **Creating restore pods** that detect and use the checkpoints

The ChReK DaemonSet handles the actual CRIU checkpoint/restore operations automatically once your pods are configured correctly.

---

## Using ChReK Without the Dynamo Operator

When using ChReK with the Dynamo operator, the operator automatically configures workload pods for checkpoint/restore. Without the operator, you must handle this configuration manually. This section documents what the operator normally injects and how to replicate it.

### Container Naming

The ChReK DaemonSet needs to identify which container in your pod is the model-serving workload (as opposed to sidecars like istio-proxy or log collectors). It resolves the target container by name:

1. If a container is named `main`, it is selected
2. Otherwise, the first container in the pod spec is selected

When using the Dynamo operator, the model container is always named `main`. In standalone mode, you must either name your model container `main` or ensure it is the first container listed in your pod spec. All YAML examples in this guide use `name: main`.

### Seccomp Profile

The operator sets a seccomp profile on all checkpoint/restore workload pods to block `io_uring` syscalls. The chrek DaemonSet deploys the profile file (`profiles/block-iouring.json`) to each node, but you must reference it in your pod specs:

```yaml
spec:
  securityContext:
    seccompProfile:
      type: Localhost
      localhostProfile: profiles/block-iouring.json
```

Without this profile, `io_uring` syscalls during restore can cause CRIU failures.

### Sleep Infinity Command for Restore Pods

The operator overrides the container command to `["sleep", "infinity"]` on restore-target pods. This produces a Running-but-not-Ready placeholder pod that the chrek DaemonSet watcher detects and restores externally via `nsenter`. Without this override, the container runs its normal entrypoint (cold-starting instead of waiting for restore).

```yaml
containers:
- name: main
  image: my-app:checkpoint-enabled
  command: ["sleep", "infinity"]
```

### Recreate Deployment Strategy

The operator forces `Recreate` strategy when restore labels are present. This prevents the old and new pods from running simultaneously, which would cause failures — two pods competing for the same GPU checkpoint data. If you are using a Deployment, set this manually:

```yaml
apiVersion: apps/v1
kind: Deployment
spec:
  strategy:
    type: Recreate
```

### PVC Volume Mount Consistency

CRIU requires identical mount layouts between checkpoint and restore. The operator ensures the checkpoint PVC is mounted at the same path in both the checkpoint job and restore pod. When configuring manually, make sure your checkpoint job and restore pod use the exact same `mountPath` for the checkpoint PVC (e.g., `/checkpoints`).

### Downward API Volume (Currently Unused)

The operator injects a Downward API volume at `/etc/podinfo` for post-restore identity discovery (pod name, namespace, UID). This is not currently consumed by any component — you can skip it for now.

### Environment Variables

The following environment variables are normally injected by the operator. They are already documented in the [Environment Variables Reference](#environment-variables-reference) below, but note that without the operator you must set them manually:

- **Checkpoint jobs:** `DYN_READY_FOR_CHECKPOINT_FILE`, `DYN_CHECKPOINT_LOCATION`, `DYN_CHECKPOINT_STORAGE_TYPE`, `DYN_CHECKPOINT_HASH`
- **Restore pods:** `DYN_CHECKPOINT_PATH`, `DYN_CHECKPOINT_HASH`

---

## Prerequisites

- Kubernetes cluster with:
  - NVIDIA GPUs with checkpoint support
  - **Privileged DaemonSet allowed** (⚠️ the ChReK DaemonSet runs privileged - see [Security Considerations](#security-considerations))
  - PVC storage (ReadWriteMany recommended for multi-node)
- Docker or compatible container runtime for building images
- Access to the ChReK source code: `deploy/chrek/`

### Security Considerations

⚠️ **Important**: The ChReK **DaemonSet** runs in privileged mode to perform CRIU checkpoint/restore operations. Your workload pods (checkpoint jobs, restore pods) do **not** need privileged mode — all CRIU privilege lives in the DaemonSet, which performs external restore via `nsenter`.

- **The DaemonSet** has `privileged: true`, `hostPID`, `hostIPC`, and `hostNetwork`
- This may violate security policies in production environments
- If the DaemonSet is compromised, it could potentially compromise node security

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

ChReK provides a `placeholder` target in its Dockerfile that layers CRIU runtime dependencies onto your existing container images. The DaemonSet performs restore externally via `nsenter`, so these dependencies must be present in the image.

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

- ✅ Installs CRIU runtime libraries (required by `nsrestore` running inside the pod's namespaces)
- ✅ Copies the `criu` binary to `/usr/local/sbin/criu`
- ✅ Copies `cuda-checkpoint` to `/usr/local/sbin/cuda-checkpoint` (used for CUDA state checkpoint/restore)
- ✅ Copies `nsrestore` to `/usr/local/bin/nsrestore` (invoked by DaemonSet via `nsenter`)
- ✅ Creates checkpoint directories (`/checkpoints`, `/var/run/criu`, `/var/criu-work`)
- ✅ Preserves your original application image contents

The placeholder image does **not** override the entrypoint or CMD. For restore pods, the operator (or you, in standalone mode) overrides the command to `sleep infinity`.

> **💡 Tip**: Using the `placeholder` target is the recommended approach as it's maintained with the ChReK codebase and ensures compatibility.

---

## Step 3: Create Checkpoint Jobs

A checkpoint job loads your application, waits for the ChReK DaemonSet to checkpoint it, and then exits.

### Required Environment Variables

Your checkpoint job MUST set these environment variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `DYN_READY_FOR_CHECKPOINT_FILE` | Path where your app signals it's ready | `/tmp/ready-for-checkpoint` |
| `DYN_CHECKPOINT_HASH` | Unique identifier for this checkpoint | `abc123def456` |
| `DYN_CHECKPOINT_LOCATION` | Directory where checkpoint is stored | `/checkpoints/abc123def456` |
| `DYN_CHECKPOINT_STORAGE_TYPE` | Storage backend type | `pvc` |

### Required Labels

Add this label to enable DaemonSet checkpoint detection:

```yaml
labels:
  nvidia.com/chrek-is-checkpoint-source: "true"
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
        nvidia.com/chrek-is-checkpoint-source: "true"  # Required for DaemonSet detection
        nvidia.com/chrek-checkpoint-hash: "abc123def456"  # Must match DYN_CHECKPOINT_HASH
    spec:
      restartPolicy: Never

      # Seccomp profile to block io_uring syscalls (deployed by the chrek DaemonSet)
      securityContext:
        seccompProfile:
          type: Localhost
          localhostProfile: profiles/block-iouring.json

      containers:
      - name: main
        image: my-app:checkpoint-enabled

        # Readiness probe: Pod becomes Ready when model is loaded
        # This is what triggers the DaemonSet to start checkpointing
        readinessProbe:
          exec:
            command: ["cat", "/tmp/ready-for-checkpoint"]
          initialDelaySeconds: 15
          periodSeconds: 2

        # Remove liveness/startup probes for checkpoint jobs
        # Model loading can take several minutes
        livenessProbe: null
        startupProbe: null

        # Checkpoint-related environment variables
        env:
        - name: DYN_READY_FOR_CHECKPOINT_FILE
          value: "/tmp/ready-for-checkpoint"
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

      volumes:
      - name: checkpoint-storage
        persistentVolumeClaim:
          claimName: chrek-pvc
```

### Application Code Requirements

Your application must implement the checkpoint flow. The DaemonSet communicates with your application via Unix signals (not files):

- **`SIGUSR1`**: Checkpoint completed — your process should exit gracefully
- **`SIGCONT`**: Restore completed — your process should wake up and continue
- **`SIGKILL`**: Checkpoint failed — process is terminated immediately (unhandleable)

Here's the pattern used by Dynamo vLLM (see `components/src/dynamo/vllm/checkpoint_restore.py`):

```python
import asyncio
import os
import signal

async def main():
    ready_file = os.environ.get("DYN_READY_FOR_CHECKPOINT_FILE")
    if not ready_file:
        # Not in checkpoint mode, run normally
        await run_application()
        return

    print("Checkpoint mode detected")

    # 1. Load your model/application
    model = await load_model()

    # 2. Put model to sleep for CRIU-friendly GPU state
    await model.sleep()

    # 3. Install signal handlers BEFORE writing the ready file to avoid a race
    #    where the DaemonSet sends a signal while default disposition (terminate)
    #    is still in effect. No handler needed for checkpoint failure — the
    #    watcher sends SIGKILL which terminates the process immediately.
    checkpoint_done = asyncio.Event()
    restore_done = asyncio.Event()

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGUSR1, checkpoint_done.set)
    loop.add_signal_handler(signal.SIGCONT, restore_done.set)

    # 4. Write ready file — triggers DaemonSet checkpoint via readiness probe
    with open(ready_file, "w") as f:
        f.write("ready")

    print("Ready for checkpoint. Waiting for watcher signal...")

    # Wait for whichever signal comes first (SIGKILL on failure kills us
    # immediately, so only success/restore signals reach this point)
    done, pending = await asyncio.wait(
        [asyncio.create_task(checkpoint_done.wait()),
         asyncio.create_task(restore_done.wait())],
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()

    if restore_done.is_set():
        # SIGCONT: Process was restored from checkpoint
        print("Restore complete, waking model")
        await model.wake_up()
        await run_application()
    else:
        # SIGUSR1: Checkpoint complete, exit
        print("Checkpoint complete, exiting")
```

**Important Notes:**

1. **Ready File & Readiness Probe**: The checkpoint job must have a readiness probe that checks for the ready file. The ChReK DaemonSet triggers checkpointing when:
   - Pod has `nvidia.com/chrek-is-checkpoint-source: "true"` label
   - Pod status is `Ready` (readiness probe passes = ready file exists)

2. **Signal handler ordering**: Install signal handlers **before** writing the ready file. Otherwise there is a race window where the DaemonSet sends a signal while the default disposition (terminate) is still in effect.

3. **Signal-based coordination**: The DaemonSet sends `SIGUSR1` after checkpoint completes, `SIGCONT` after restore completes, and `SIGKILL` if checkpoint fails. Your application must handle `SIGUSR1` and `SIGCONT` (not poll for files). `SIGKILL` cannot be caught — the kernel terminates the process immediately.

4. **Three exit paths**:
   - **SIGUSR1 received**: Checkpoint complete, exit gracefully
   - **SIGCONT received**: Process was restored, wake model and continue
   - **SIGKILL received**: Checkpoint failed, process terminated immediately (no handler needed)


---

## Step 4: Restore from Checkpoints

The DaemonSet performs restore externally — your restore pod just needs to be a placeholder that sleeps until the DaemonSet restores the checkpointed process into it.

### Example Restore Pod

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-app-restored
  namespace: my-app
  labels:
    nvidia.com/chrek-is-restore-target: "true"  # Required: watcher detects restore pods by this label
    nvidia.com/chrek-checkpoint-hash: "abc123def456"  # Required: watcher uses this to locate the checkpoint
spec:
  restartPolicy: Never

  # Seccomp profile to block io_uring syscalls (deployed by the chrek DaemonSet)
  # Without this, io_uring syscalls may cause CRIU restore failures
  securityContext:
    seccompProfile:
      type: Localhost
      localhostProfile: profiles/block-iouring.json

  containers:
  - name: main
    image: my-app:checkpoint-enabled

    # Override command to sleep — the chrek DaemonSet performs external restore
    # on Running-but-not-Ready pods. Without this, the container would cold-start.
    command: ["sleep", "infinity"]

    # Set checkpoint environment variables
    env:
    - name: DYN_CHECKPOINT_HASH
      value: "abc123def456"  # Must match checkpoint job
    - name: DYN_CHECKPOINT_PATH
      value: "/checkpoints"  # Base path (hash appended automatically)

    # GPU request
    resources:
      limits:
        nvidia.com/gpu: 1

    # CRIU needs write access for restore.log — do NOT set readOnly
    volumeMounts:
    - name: checkpoint-storage
      mountPath: /checkpoints

  volumes:
  - name: checkpoint-storage
    persistentVolumeClaim:
      claimName: chrek-pvc
```

### How Restore Works

1. **Pod starts as placeholder**: The `sleep infinity` command keeps the pod Running but not Ready
2. **DaemonSet detects restore pod**: The watcher finds pods with `nvidia.com/chrek-is-restore-target=true` that are Running but not Ready
3. **External restore via nsenter**: The DaemonSet enters the pod's namespaces and performs CRIU restore, including GPU state
4. **Application continues**: Your application resumes exactly where it was checkpointed

---

## Environment Variables Reference

### Checkpoint Jobs

| Variable | Required | Description |
|----------|----------|-------------|
| `DYN_READY_FOR_CHECKPOINT_FILE` | Yes | Full path where app signals readiness (e.g., `/tmp/ready-for-checkpoint`) |
| `DYN_CHECKPOINT_HASH` | Yes | Unique checkpoint identifier (16-char hex string) |
| `DYN_CHECKPOINT_LOCATION` | Yes | Directory where checkpoint is stored (e.g., `/checkpoints/abc123def456`) |
| `DYN_CHECKPOINT_STORAGE_TYPE` | Yes | Storage backend: `pvc`, `s3`, or `oci` |

### Restore Pods

| Variable | Required | Description |
|----------|----------|-------------|
| `DYN_CHECKPOINT_HASH` | Yes | Checkpoint identifier (must match checkpoint job) |
| `DYN_CHECKPOINT_PATH` | Yes | Base checkpoint directory (hash appended automatically) |

### Signals (DaemonSet → Application)

The DaemonSet communicates checkpoint/restore completion via Unix signals, not files:

| Signal | Direction | Meaning |
|--------|-----------|---------|
| `SIGUSR1` | DaemonSet → checkpoint pod | Checkpoint completed, process should exit |
| `SIGCONT` | DaemonSet → restored pod | Restore completed, process should wake up |
| `SIGKILL` | DaemonSet → checkpoint pod | Checkpoint failed — process terminated immediately |

CRIU tuning options are configured via the ChReK Helm chart's `config.checkpoint.criu` values, not environment variables. See the [Helm Chart Values](https://github.com/ai-dynamo/dynamo/tree/main/deploy/helm/charts/chrek/values.yaml) for available options.

---

## Checkpoint Flow Explained

### 1. Checkpoint Creation Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Pod starts with nvidia.com/chrek-is-checkpoint-source=true label  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Application loads model and creates ready file           │
│    /tmp/ready-for-checkpoint                                 │
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
│    - Has chrek-is-checkpoint-source label                     │
│    - Has chrek-checkpoint-hash label                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. DaemonSet executes CRIU checkpoint:                      │
│    - Freezes container process                               │
│    - Dumps memory (CPU + GPU)                                │
│    - Saves to /checkpoints/${HASH}/                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. DaemonSet sends SIGUSR1 to the application process       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. Application receives SIGUSR1 and exits gracefully        │
└─────────────────────────────────────────────────────────────┘
```

### 2. Restore Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Pod starts with restore labels and sleep infinity        │
│    (Running but not Ready)                                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. ChReK DaemonSet detects:                                 │
│    - Pod is Running but not Ready                            │
│    - Has chrek-is-restore-target label                       │
│    - Has chrek-checkpoint-hash label                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. DaemonSet performs external restore via nsenter:          │
│    - Enters pod's namespaces (mount, net, pid, ipc)         │
│    - Runs nsrestore with CRIU inside the pod's context      │
│    - Restores memory (CPU + GPU via cuda-checkpoint)        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. DaemonSet sends SIGCONT to the restored process           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Application receives SIGCONT, wakes model, continues      │
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
   kubectl get pod <pod-name> -o jsonpath='{.metadata.labels.nvidia\.com/chrek-is-checkpoint-source}'
   ```

2. Check pod readiness:
   ```bash
   kubectl get pod <pod-name> -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}'
   ```

3. Check ready file was created:
   ```bash
   kubectl exec <pod-name> -- ls -la /tmp/ready-for-checkpoint
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

2. Check DaemonSet logs for restore errors:
   ```bash
   kubectl logs -n my-app daemonset/chrek-agent --all-containers
   ```

3. Check pod events for restore status annotations:
   ```bash
   kubectl describe pod <pod-name>
   ```

4. Ensure checkpoint and restore have same:
   - Container image (built with `placeholder` target)
   - GPU count
   - Volume mounts (same `mountPath` for checkpoint PVC)

### Restore Pod Not Detected

**Symptom**: Pod runs `sleep infinity` but DaemonSet never restores it

**Checks**:
1. Verify the pod has the required labels:
   ```bash
   kubectl get pod <pod-name> -o jsonpath='{.metadata.labels}'
   ```
   Must have both `nvidia.com/chrek-is-restore-target: "true"` and `nvidia.com/chrek-checkpoint-hash: "<hash>"`.

2. Verify the pod is Running but not Ready (this is the trigger):
   ```bash
   kubectl get pod <pod-name> -o jsonpath='{.status.phase}'
   kubectl get pod <pod-name> -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}'
   ```

3. Verify the DaemonSet is running on the same node:
   ```bash
   kubectl get pods -n my-app -l app.kubernetes.io/name=chrek -o wide
   ```

---

## Additional Resources

- [ChReK Helm Chart Values](https://github.com/ai-dynamo/dynamo/tree/main/deploy/helm/charts/chrek/values.yaml)
- [Dynamo vLLM ChReK Integration](https://github.com/ai-dynamo/dynamo/tree/main/components/src/dynamo/vllm/checkpoint_restore.py) - Reference signal handler implementation
- [ChReK Dockerfile](https://github.com/ai-dynamo/dynamo/tree/main/deploy/chrek/Dockerfile)
- [CRIU Documentation](https://criu.org/Main_Page)
- [CUDA Checkpoint Utility](https://github.com/NVIDIA/cuda-checkpoint)

---

## Getting Help

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review DaemonSet logs: `kubectl logs -n <namespace> daemonset/chrek-agent`
3. Open an issue on [GitHub](https://github.com/ai-dynamo/dynamo/issues)
