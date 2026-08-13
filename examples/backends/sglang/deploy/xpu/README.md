<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Intel XPU Deployment Examples

Hardware-specific deployment templates for Intel GPUs, with Kubernetes Dynamic Resource Allocation (DRA) and device-plugin variants.

## Available Templates

| File | Pattern | Description |
|------|---------|-------------|
| `agg_xpu_dra.yaml` | Aggregated | Single worker with XPU target |
| `agg_router_xpu_dra.yaml` | Aggregated + KV Router | Two workers behind the KV router |
| `disagg_xpu_dra.yaml` | Disaggregated | Separate prefill and decode workers with NIXL |
| `disagg_planner_xpu_dra.yaml` | Disaggregated + Planner | Dynamo Planner for throughput scaling |
| `disagg_xpu.yaml` | Disaggregated (Device Plugin) | Device plugin resource `gpu.intel.com/xe` |

## Prerequisites

1. **Kubernetes v1.34 or later** with the DRA v1 API enabled
2. **Intel GPU DRA driver** with the `gpu.intel.com` device class
3. **Custom XPU runtime image** built from source with `--device xpu`
4. **Hugging Face token secret** named `hf-token-secret`, with the token stored under the `HF_TOKEN` key

## Key Differences from NVIDIA Templates

| Aspect | NVIDIA | Intel XPU |
|--------|--------|-----------|
| GPU Allocation | `nvidia.com/gpu` resource limit | DRA `ResourceClaimTemplate` or `gpu.intel.com/xe` resource limit |
| Device Target | Default (CUDA) | `--device xpu` flag |
| CUDA Graph | Enabled | `--disable-cuda-graph` |
| Grammar Backend | Default | `--grammar-backend none` |
| DeviceClass | `nvidia.com` | `gpu.intel.com` |
| Disagg KV Transfer | Default | `hostIPC: true`, `UCX_TLS=ze_ipc,...` |

> [!NOTE]
> Do not set `ZE_AFFINITY_MASK` with DRA. It conflicts with DRA device allocation and can cause a segmentation fault.

## Deploy

```bash
# Apply template (includes ResourceClaimTemplate)
kubectl apply -f xpu/agg_xpu_dra.yaml -n ${NAMESPACE}

# Verify GPU allocation
kubectl get resourceclaim -n ${NAMESPACE}
kubectl get resourceslices

# Check deployment status
kubectl get dynamographdeployment -n ${NAMESPACE}
kubectl get pods -n ${NAMESPACE}
```

## Testing

```bash
# Port forward to frontend
kubectl port-forward deployment/sglang-agg-xpu-dra-frontend 8000:8000 -n ${NAMESPACE}

# Test inference
curl localhost:8000/v1/models
curl localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"Hello","max_tokens":20}'
```

## Further Reading

- [Main Deployment README](../README.md) - Overview of all deployment patterns
- [Intel resource drivers for Kubernetes](https://github.com/intel/intel-resource-drivers-for-kubernetes)
