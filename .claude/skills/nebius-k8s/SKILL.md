---
name: nebius-k8s
description: Deploy and manage DynamoGraphDeployments on the Nebius H200 cluster. Use when deploying models, creating DGDs, managing k8s resources, or working with the Nebius cluster.
---

# Nebius K8s Cluster - Dynamo Deployments

## Cluster Overview

| Property | Value |
|----------|-------|
| **Context** | `nebius-mk8s-dynamo-h200-01` |
| **GPU Nodes** | 16+ nodes with 8x H200 GPUs each |
| **Default Namespace** | `aflowers-exemplar` |
| **Storage Class** | `csi-mounted-fs-path-sc` (RWX) |
| **Dynamo Operator** | Cluster-wide, managed (do not install) |

### Available CRDs
- `dgd` - DynamoGraphDeployment
- `dcd` - DynamoComponentDeployment
- `dgdr` - DynamoGraphDeploymentRequest
- `dgdsa` - DynamoGraphDeploymentScalingAdapter
- `dm` - DynamoModel
- `dwm` - DynamoWorkerMetadata

## Container Images

### NGC Sources

**Pre-release (nvstaging)** - Latest features for testing:
```
nvcr.io/nvstaging/ai-dynamo/vllm-runtime:0.8.0rc2-amd64
nvcr.io/nvstaging/ai-dynamo/vllm-runtime:0.8.0rc2-cuda13-amd64
nvcr.io/nvstaging/ai-dynamo/sglang-runtime:0.8.0rc2-amd64
nvcr.io/nvstaging/ai-dynamo/tensorrtllm-runtime:0.8.0rc2-amd64
```

**Released (nvidia)** - Stable production:
```
nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.7.1
nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.7.1
nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.7.0.post2
```

### Mirror to Nebius Registry (Faster Pulls)

```bash
# 1. Pull from NGC
docker pull nvcr.io/nvstaging/ai-dynamo/vllm-runtime:0.8.0rc2-amd64

# 2. Retag for Nebius
docker tag nvcr.io/nvstaging/ai-dynamo/vllm-runtime:0.8.0rc2-amd64 \
  cr.eu-north1.nebius.cloud/e00nvy8qsma32vywq9/dynamo:vllm-0.8.0rc2

# 3. Push to Nebius
docker push cr.eu-north1.nebius.cloud/e00nvy8qsma32vywq9/dynamo:vllm-0.8.0rc2
```

**Nebius Registry**: `cr.eu-north1.nebius.cloud/e00nvy8qsma32vywq9/dynamo:<tag>`

Use descriptive tags: `vllm-0.8.0rc2`, `tensorrtllm-0.7.0.post2`, `sglang-0.7.1`

## Prerequisites

### Existing Resources in aflowers-exemplar

**Secrets** (already created):
- `hf-token-secret` - HuggingFace token
- `nvcr-imagepullsecret` - NVCR pull credentials
- `nebius-registry` - Nebius registry credentials

**PVCs** (already created):
- `model-cache` - 25TB, model storage
- `compilation-cache` - 25TB, vLLM compilation artifacts
- `perf-cache` - 25TB, benchmark results

### Create HF Token Secret (if needed)
```bash
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="$HF_TOKEN" \
  -n aflowers-exemplar
```

### Create PVCs (if needed)

See [dgd-templates.md](dgd-templates.md) for PVC definitions using `csi-mounted-fs-path-sc`.

## Deploying DGDs

### Quick Start - Simple Aggregated

```bash
# Apply a simple vLLM aggregated deployment
kubectl apply -f examples/backends/vllm/deploy/agg.yaml -n aflowers-exemplar

# Check status
kubectl get dgd -n aflowers-exemplar
kubectl get pods -n aflowers-exemplar
```

### Deployment Patterns

| Pattern | Use Case | Example |
|---------|----------|---------|
| **Aggregated** | Simple, single-GPU models | `agg.yaml` |
| **Aggregated + Router** | Load-balanced inference | `agg_router.yaml` |
| **Disaggregated** | Separate prefill/decode | `disagg.yaml` |
| **Disaggregated + KV Router** | High throughput with caching | `disagg_router.yaml` |

### Reference Recipes

Full DGD examples at `/home/aflowers/Documents/dynamo/recipes/qwen3-32b/`:
- `vllm/agg-round-robin/deploy.yaml` - Aggregated 8x TP2 workers
- `vllm/disagg-kv-router/deploy.yaml` - Disaggregated 6 prefill + 2 decode

## Common kubectl Commands

### DGD Management
```bash
# List all DGDs
kubectl get dgd -n aflowers-exemplar

# Describe a DGD
kubectl describe dgd <name> -n aflowers-exemplar

# Delete a DGD
kubectl delete dgd <name> -n aflowers-exemplar

# Watch pods come up
kubectl get pods -n aflowers-exemplar -w
```

### Debugging
```bash
# Check pod logs
kubectl logs <pod-name> -n aflowers-exemplar

# Exec into a pod
kubectl exec -it <pod-name> -n aflowers-exemplar -- bash

# Describe pod for events
kubectl describe pod <pod-name> -n aflowers-exemplar
```

### Port Forwarding
```bash
# Forward frontend service
kubectl port-forward svc/<dgd-name>-frontend 8000:8000 -n aflowers-exemplar

# Test endpoint
curl http://localhost:8000/v1/models
```

### Scaling
```bash
# Scale via DGDSA
kubectl scale dgdsa <dgd-name>-<service> --replicas=3 -n aflowers-exemplar

# Check scaling adapters
kubectl get dgdsa -n aflowers-exemplar
```

## Troubleshooting

### GPU Node Health

Check node status - many show GPU issues:
```bash
kubectl get nodes -o custom-columns='NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu,STATUS:.status.conditions[-1].type'
```

Common issues:
- `SysLogsGPUFallenOff` - GPU hardware issue
- `RepeatedXID13OnSameGPCAndTPC` - GPU memory errors
- `SysLogsXIDError` - GPU driver errors

Avoid scheduling on unhealthy nodes or use node selectors.

### Pod Not Starting

1. Check events: `kubectl describe pod <pod> -n aflowers-exemplar`
2. Check image pull: Ensure `nvcr-imagepullsecret` or `nebius-registry` is configured
3. Check GPU availability: `kubectl describe node <node> | grep -A5 "Allocated resources"`

### Model Download Issues

1. Verify `hf-token-secret` exists
2. Check `HF_HOME` env var points to mounted PVC
3. Run model download job first (see recipes)

## Additional Resources

- K8s docs: `/home/aflowers/Documents/dynamo/docs/kubernetes/`
- API reference: `/home/aflowers/Documents/dynamo/docs/kubernetes/api_reference.md`
- vLLM examples: `/home/aflowers/Documents/dynamo/examples/backends/vllm/deploy/`
- Qwen3-32B recipe: `/home/aflowers/Documents/dynamo/recipes/qwen3-32b/`
