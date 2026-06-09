<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Intel XPU Deployment Examples

Hardware-specific deployment templates for Intel XPU GPUs using Kubernetes Dynamic Resource Allocation (DRA).

## Available Templates

| File | Pattern | Description |
|------|---------|-------------|
| `agg_xpu_dra.yaml` | Aggregated | Single worker with XPU target |
| `agg_router_xpu_dra.yaml` | Aggregated + KV Router | 2 workers behind KV-aware router |
| `disagg_xpu_dra.yaml` | Disaggregated | Prefill/decode separation with NixlConnector |
| `disagg_planner_xpu_dra.yaml` | Disaggregated + Planner | Global Planner for throughput scaling |
| `disagg_xpu.yaml` | Disaggregated (Device Plugin) | Traditional device plugin (gpu.intel.com/xe) |

## Prerequisites

1. **Kubernetes v1.34+** with DRA API v1 enabled
2. **Intel GPU DRA Driver** installed with DeviceClass `gpu.intel.com`
3. **Custom XPU runtime image** (build from source with `--device xpu`)
4. **HuggingFace token secret**: `kubectl create secret generic hf-token-secret --from-literal=token=<your-token>`

## Key Differences from NVIDIA Templates

| Aspect | NVIDIA | Intel XPU |
|--------|--------|-----------|
| GPU Allocation | `resources.limits.gpu` | DRA `ResourceClaimTemplate` |
| Device Target | Default (CUDA) | `--device xpu` flag |
| CUDA Graph | Enabled | `--disable-cuda-graph` |
| Grammar Backend | Default | `--grammar-backend none` |
| DeviceClass | `nvidia.com` | `gpu.intel.com` |
| Disagg KV Transfer | Default | `hostIPC: true`, `UCX_TLS=ze_ipc,...` |

Note: Do not set `ZE_AFFINITY_MASK` with DRA - it conflicts and causes SIGSEGV.

## Deploy

```bash
# Apply template (includes ResourceClaimTemplate)
kubectl apply -f xpu/agg_xpu_dra.yaml -n $NAMESPACE

# Verify GPU allocation
kubectl get resourceclaim -n $NAMESPACE
kubectl get resourceslices

# Check deployment status
kubectl get dynamographdeployment -n $NAMESPACE
kubectl get pods -n $NAMESPACE
```

## Testing

```bash
# Port forward to frontend
kubectl port-forward deployment/sglang-agg-xpu-dra-frontend 8000:8000 -n $NAMESPACE

# Test inference
curl localhost:8000/v1/models
curl localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"Hello","max_tokens":20}'
```

## Further Reading

- [Main Deployment README](../README.md) - Overview of all deployment patterns
- [Intel XPU Resource Driver](https://github.com/intel/intel-resource-drivers-for-kubernetes)
