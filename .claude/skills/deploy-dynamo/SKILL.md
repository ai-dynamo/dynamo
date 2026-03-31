---
name: deploy-dynamo
description: Deploy a debug/dev pod to Kubernetes with Dynamo runtime image, model-cache PVC, shared memory, HF secret, and GPU resources. Use when creating or updating k8s pod deployments for testing.
---

# Deploy Dynamo Pod to Kubernetes

Claude Code skill for creating and deploying a Kubernetes pod with a Dynamo runtime image for debugging or testing.

## When Invoked

### 1. Gather Information

Ask for (use defaults if not provided):
- **Pod name** — name for the pod (default: `biswa-debug`)
- **Image** — full container image URI (required)
- **GPU count** — number of GPUs to request (default: `4`)
- **Model cache PVC** — PVC name for model storage (default: `model-cache`)
- **Model cache mount path** — where to mount the PVC (default: `/opt/models`)
- **Shared memory size** — size of `/dev/shm` (default: `80Gi`)
- **HF secret name** — Kubernetes secret with HuggingFace token (default: `hf-token-secret`)
- **Command** — container entrypoint (default: `sleep infinity`)

### 2. Generate Pod YAML

Create the pod YAML file at the repo root (e.g., `<pod-name>.yaml`) with:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: <pod-name>
spec:
  containers:
    - name: <pod-name>
      image: <image>
      command: ["sleep", "infinity"]
      envFrom:
        - secretRef:
            name: <hf-secret-name>
      env:
        - name: HF_HOME
          value: <model-cache-mount-path>
      volumeMounts:
        - name: model-cache
          mountPath: <model-cache-mount-path>
        - name: dshm
          mountPath: /dev/shm
      resources:
        limits:
          nvidia.com/gpu: "<gpu-count>"
        requests:
          nvidia.com/gpu: "<gpu-count>"
  volumes:
    - name: model-cache
      persistentVolumeClaim:
        claimName: <model-cache-pvc>
    - name: dshm
      emptyDir:
        medium: Memory
        sizeLimit: <shared-memory-size>
```

### 3. Deploy

```bash
kubectl apply -f <pod-name>.yaml
```

### 4. Verify

```bash
kubectl get pod <pod-name> -w
```

Wait until the pod status is `Running`.

### 5. Connect (Optional)

```bash
kubectl exec -it <pod-name> -- /bin/bash
```

## Reference Files

| File | Purpose |
|------|---------|
| `recipes/kimi-k2.5/trtllm/agg/nvidia/deploy.yaml` | Reference for volume mounts, shared memory, HF secret, GPU resources |
| `recipes/llama-3-70b/model-cache/model-cache.yaml` | Reference PVC definition for model-cache |

## Notes

- Ensure `tsh login` / `tsh kube login` is done before deploying — credentials expire.
- The `model-cache` PVC must already exist in the target namespace.
- The `hf-token-secret` secret must already exist with key `HF_TOKEN`.
- Do NOT switch k8s namespace without explicit user permission.
