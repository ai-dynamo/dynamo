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
| `agg_tracing_xpu_dra.yaml` | Aggregated + Tracing | Single worker with OpenTelemetry tracing |
| `agg_router_xpu_dra.yaml` | Aggregated + KV Router | Aggregated deployment with KV routing |
| `agg_router_kv_approx_xpu_dra.yaml` | Aggregated + KV Router (Local) | KV routing without NATS dependency |
| `disagg_xpu_dra.yaml` | Disaggregated | Prefill/decode separation with NixlConnector |
| `disagg_tracing_xpu_dra.yaml` | Disaggregated + Tracing | Disaggregated with OpenTelemetry tracing |
| `disagg_planner_xpu_dra.yaml` | Disaggregated + Planner | Global Planner for throughput scaling |
| `disagg_router_xpu_dra.yaml` | Disaggregated + KV Router | Disaggregated with KV cache routing |

## Prerequisites

1. **Kubernetes v1.34+** with DRA API v1 enabled
2. **[Intel resource drivers for Kubernetes](https://github.com/intel/intel-resource-drivers-for-kubernetes)** installed with DeviceClass `gpu.intel.com`
3. **Custom XPU runtime image** built with Intel XPU support:
   ```bash
   python container/render.py --framework=vllm --device=xpu --target=runtime
   docker build -t nvcr.io/nvidia/ai-dynamo/vllm-runtime-xpu:1.3.0.dev2026071601 \
     -f container/vllm-runtime-xpu-amd64-rendered.Dockerfile .
   ```
   See [container/README.md](../../../../../container/README.md) for complete build instructions.
4. **HuggingFace token secret**:
   ```bash
   export HF_TOKEN=your_hf_token
   kubectl create secret generic hf-token-secret \
     --from-literal=HF_TOKEN=${HF_TOKEN} \
     -n ${NAMESPACE}
   ```

## Key Differences from NVIDIA Templates

| Aspect | NVIDIA | Intel XPU |
|--------|--------|-----------|
| GPU Allocation | `resources.limits.gpu` | DRA `ResourceClaimTemplate` |
| Device Target | Default (CUDA) | `VLLM_TARGET_DEVICE: xpu` |
| KV Transfer | Default buffer | `kv_buffer_device: "xpu"` |
| DeviceClass | `nvidia.com` | `gpu.intel.com` |

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
kubectl port-forward deployment/vllm-agg-xpu-dra-frontend 8000:8000 -n $NAMESPACE

# Test inference
curl localhost:8000/v1/models
curl localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"Hello","max_tokens":20}'
```

## Further Reading

- [Main Deployment README](../README.md) - Overview of all deployment patterns
- [Intel resource drivers for Kubernetes](https://github.com/intel/intel-resource-drivers-for-kubernetes)
