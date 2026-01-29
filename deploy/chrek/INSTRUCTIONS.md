# Checkpoint/Restore Instructions for Dynamo vLLM Workers

This document describes the steps to set up and run CRIU-based checkpoint/restore for Dynamo vLLM workers in a local kind cluster with GPU support.

## Prerequisites

- Docker with NVIDIA Container Toolkit
- kind (Kubernetes in Docker)
- kubectl
- Go 1.21+
- NVIDIA GPU with compatible drivers (tested with 590 drivers on RTX 5880 Ada)

## 1. Create Kind Cluster with GPU Support

Create a kind cluster configuration that exposes the GPU:

```bash
# Create kind cluster with GPU support
kind create cluster --name dynamo-gpu --config deploy/helm/kind-gpu-config.yaml
```

The cluster needs the NVIDIA device plugin and runtime class configured.

## 2. Deploy Dynamo Platform

Deploy the Dynamo platform components using Helm:

```bash
cd deploy/helm
helm install dynamo . -n dynamo-system --create-namespace -f minikube-values.yaml
```

This deploys:
- DynamoGraphDeployment (DGD) controller
- DynamoCheckpoint controller
- Other platform components

## 3. Build the Checkpoint Agent (chrek)

Build the checkpoint agent binary:

```bash
cd deploy/chrek
go build -o checkpoint-agent ./cmd/checkpoint-agent
```

## 4. Build the Restore Entrypoint

Build the restore-entrypoint binary that handles CRIU restore operations:

```bash
cd deploy/chrek
go build -o restore-entrypoint-new ./cmd/restore-entrypoint
```

## 5. Create Patched Placeholder Image

The placeholder image needs updated code with `DYN_` prefix environment variables (matching the operator). Create a patch Dockerfile:

```dockerfile
# Dockerfile.patch
ARG BASE_IMAGE=nvcr.io/nvidian/dynamo-dev/schwinns-dynamo-vllm-placeholder:latest

FROM ${BASE_IMAGE}

# Copy updated main.py with DYN_ prefix env vars
COPY main.py /opt/dynamo/venv/lib/python3.12/site-packages/dynamo/vllm/main.py

# Copy updated smart-entrypoint.sh with DYN_ prefix env vars
COPY scripts/smart-entrypoint.sh /smart-entrypoint.sh
RUN chmod +x /smart-entrypoint.sh

# Copy updated restore-entrypoint binary
COPY restore-entrypoint-new /restore-entrypoint
RUN chmod +x /restore-entrypoint
```

Build and load into kind:

```bash
docker build -f Dockerfile.patch -t schwinns-dynamo-vllm-placeholder:patched .
kind load docker-image schwinns-dynamo-vllm-placeholder:patched --name dynamo-gpu
```

## 6. Create DynamoGraphDeployment

Create a DGD manifest with checkpoint enabled:

```yaml
# vllm-test-dgd.yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: vllm-test
  namespace: schwinns
spec:
  pvcs:
    - name: hf-cache-pvc
      create: false
  services:
    frontend:
      componentType: frontend
      replicas: 1
      volumeMounts:
        - name: hf-cache-pvc
          mountPoint: /home/dynamo/.cache/huggingface
      extraPodSpec:
        imagePullSecrets:
          - name: ngc-secret
        mainContainer:
          image: nvcr.io/nvidian/dynamo-dev/schwinns-dynamo-vllm-runtime:latest
          env:
            - name: HF_HOME
              value: /home/dynamo/.cache/huggingface
            - name: DYN_STORE_KV
              value: "file"
            - name: DYN_REQUEST_PLANE
              value: "tcp"
    worker:
      componentType: worker
      replicas: 1
      envFromSecret: hf-token-secret
      resources:
        limits:
          gpu: "1"
      volumeMounts:
        - name: hf-cache-pvc
          mountPoint: /home/dynamo/.cache/huggingface
      checkpoint:
        enabled: true
        mode: Auto
        identity:
          model: "Qwen/Qwen3-0.6B"
          backendFramework: vllm
          frameworkVersion: "0.14.0"
          tensorParallelSize: 1
          pipelineParallelSize: 1
          dtype: bfloat16
          maxModelLen: 8192
          extraParameters:
            enableChunkedPrefill: "true"
      extraPodSpec:
        runtimeClassName: nvidia
        imagePullSecrets:
          - name: ngc-secret
        mainContainer:
          image: schwinns-dynamo-vllm-placeholder:patched
          imagePullPolicy: Never  # Use local image from kind load
          workingDir: /workspace/examples/backends/vllm
          command:
            - "/smart-entrypoint.sh"
          args:
            - "python3"
            - "-m"
            - "dynamo.vllm"
            - "--model"
            - "Qwen/Qwen3-0.6B"
            - "--tensor-parallel-size"
            - "1"
            - "--connector"
            - "none"
            - "--enable-sleep-mode"
            - "--no-enable-prefix-caching"
          env:
            - name: HF_HOME
              value: /home/dynamo/.cache/huggingface
            - name: DYN_STORE_KV
              value: "file"
            - name: DYN_REQUEST_PLANE
              value: "tcp"
```

Apply the DGD:

```bash
kubectl apply -f vllm-test-dgd.yaml
```

## 7. Wait for Worker to Start and Create Checkpoint

The checkpoint controller will automatically:
1. Detect the worker pod is ready
2. Check if a matching checkpoint exists
3. If not, trigger checkpoint creation via the checkpoint agent

Monitor the checkpoint agent logs:

```bash
kubectl logs -n schwinns -l app=dynamo-checkpoint-agent -f
```

Monitor the worker pod for checkpoint signal:

```bash
kubectl logs -n schwinns -l app=vllm-test-worker -f
```

The checkpoint process:
1. Agent sends `DYN_CHECKPOINT_SIGNAL_FILE` to worker
2. Worker's vLLM main.py detects the signal and calls CRIU dump
3. CRIU creates checkpoint images in `/checkpoints/<hash>/`
4. Worker writes `checkpoint.done` marker when complete
5. Worker exits after checkpoint

## 8. Restore from Checkpoint

When the worker pod restarts (or is deleted and recreated), the restore flow:

1. `smart-entrypoint.sh` checks for checkpoint via `DYN_CHECKPOINT_HASH` env var
2. If checkpoint exists, invokes `/restore-entrypoint`
3. `restore-entrypoint`:
   - Applies rootfs diff (filesystem changes from checkpoint)
   - Creates nvidia-ctk-hook symlink (see below)
   - Calls CRIU restore with proper mount mappings
   - Forwards output from restored process

To trigger a restore, delete the worker pod:

```bash
kubectl delete pod -n schwinns -l app=vllm-test-worker
```

The new pod will detect the existing checkpoint and restore from it.

## 9. Verify Restore Success

Check worker logs for successful restore:

```bash
kubectl logs -n schwinns -l app=vllm-test-worker
```

Expected output:
```
[smart-entrypoint] Checkpoint found: <hash> (checkpoint.done marker present)
[smart-entrypoint] CHECKPOINT RESTORE MODE
...
time="..." level=info msg="Created nvidia-ctk-hook symlink for CRIU mount remapping"
time="..." level=info msg="CRIU c.Restore completed successfully" duration=5.xxxs
time="..." level=info msg="=== Restore operation completed ==="
```

Test inference:

```bash
kubectl port-forward -n schwinns svc/vllm-test-frontend 8000:8000 &
curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'
```

## Key Technical Details

### Environment Variable Prefix

The operator uses `DYN_` prefix for environment variables:
- `DYN_CHECKPOINT_PATH` - Base directory for checkpoints (default: `/checkpoints`)
- `DYN_CHECKPOINT_HASH` - Hash/ID of checkpoint to restore
- `DYN_CHECKPOINT_SIGNAL_FILE` - Signal file to trigger checkpoint

### NVIDIA Container Toolkit Mount Handling

NVIDIA Container Toolkit creates tmpfs mounts with random UUIDs:
```
/run/nvidia-ctk-hook<random-uuid>
```

The checkpoint container and restore container have different UUIDs. The restore-entrypoint handles this by:

1. Parsing the checkpoint's `dump.log` to find the original nvidia-ctk-hook path
2. Parsing `/proc/1/mountinfo` to find the restore container's nvidia-ctk-hook path
3. Creating a symlink from the checkpoint path to the restore path
4. Adding an external mount mapping for CRIU

This is implemented in `pkg/restore/mounts.go`:
- `findNvidiaCtkhookFromDumpLog()` - Extracts path from dump.log
- `PrepareNvidiaCtkhookSymlink()` - Creates the symlink
- `GenerateExtMountMaps()` - Generates CRIU mount mappings

### Checkpoint Contents

A checkpoint directory contains:
```
/checkpoints/<hash>/
├── checkpoint.done      # Marker indicating checkpoint is complete
├── metadata.json        # Checkpoint metadata (mounts, paths, etc.)
├── dump.log            # CRIU dump log
├── restore.log         # CRIU restore log (after restore attempt)
├── rootfs-diff.tar     # Filesystem changes since container start
├── deleted.files       # List of deleted files
├── inventory.img       # CRIU inventory
├── core-*.img          # CRIU process state images
├── mm-*.img            # CRIU memory images
├── pagemap-*.img       # CRIU page maps
├── pages-*.img         # CRIU memory pages
└── ...                 # Other CRIU image files
```

### CUDA Plugin

For GPU checkpoints, CRIU uses the CUDA plugin:
- Plugin directory: `/usr/local/lib/criu`
- Requires `CRIU_TIMEOUT` environment variable (e.g., 21600 seconds)
- Plugin handles CUDA context save/restore automatically

## Troubleshooting

### Checkpoint not triggering

1. Check checkpoint agent logs for errors
2. Verify `DYN_CHECKPOINT_SIGNAL_FILE` path is accessible
3. Check if checkpoint identity matches existing checkpoints

### Restore fails with mount errors

1. Check for nvidia-ctk-hook path mismatch in logs
2. Verify symlink creation succeeded
3. Check CRIU restore.log for specific mount errors

### CUDA restore fails

1. Ensure CRIU CUDA plugin is installed at `/usr/local/lib/criu`
2. Set `CRIU_TIMEOUT` environment variable
3. Check for CUDA driver version compatibility

### Worker not becoming ready after restore

1. Check if restored process is running: `kubectl exec ... -- ps aux`
2. Check for network binding issues (port conflicts)
3. Verify Kubernetes discovery is working

## Performance Benchmarks

Timing results from testing on RTX 5880 Ada (48GB) with Qwen/Qwen3-0.6B model:

### Checkpoint Creation (CRIU Dump)

| Operation | Duration |
|-----------|----------|
| CRIU dump (GPU + process state) | ~12.3 seconds |
| Rootfs diff capture | ~0.7 seconds |
| **Total checkpoint** | **~13.1 seconds** |

### Restore (CRIU Restore)

| Operation | Run 1 | Run 2 | Run 3 |
|-----------|-------|-------|-------|
| Apply rootfs diff | 142ms | 142ms | 147ms |
| **CRIU restore** | **4.59s** | **4.09s** | **4.09s** |
| GPU wake up | 5.3ms | 6.1ms | 4.8ms |
| **Total restore** | **4.73s** | **4.23s** | **4.24s** |

Average CRIU restore time: **~4.3 seconds**

### Cold Start (For Comparison)

| Operation | Duration |
|-----------|----------|
| Model loading | ~17 seconds |
| torch.compile | ~15 seconds |
| CUDA graph capture | ~4 seconds |
| KV cache initialization | ~25 seconds |
| **Total cold start** | **~91 seconds** |

### Summary

- **Restore is ~21x faster than cold start** (4.3s vs 91s)
- CRIU checkpoint captures full GPU state including CUDA contexts and memory
- The nvidia-ctk-hook UUID mismatch is handled via symlink creation
- Inference works correctly after restore

## Rebuilding After Code Changes

After modifying restore code:

```bash
cd deploy/chrek

# Rebuild binary
go build -o restore-entrypoint-new ./cmd/restore-entrypoint

# Rebuild Docker image
docker build -f Dockerfile.patch -t schwinns-dynamo-vllm-placeholder:patched .

# Load into kind
kind load docker-image schwinns-dynamo-vllm-placeholder:patched --name dynamo-gpu

# Delete worker pod to pick up new image
kubectl delete pod -n schwinns -l app=vllm-test-worker
```
