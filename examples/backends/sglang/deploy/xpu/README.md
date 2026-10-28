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
| `disagg_xpu.yaml` | Disaggregated (Device Plugin) | Device plugin resource `gpu.intel.com/xe` |

## Prerequisites

Build the XPU runtime image from the current source:

```bash
python3 container/render.py --framework=sglang --device=xpu --target=runtime
docker build -t my-registry/sglang-runtime-xpu:my-tag \
  -f container/sglang-runtime-xpu-amd64-rendered.Dockerfile .
```

The manifests use placeholder images (`my-registry/...:my-tag`) and `imagePullPolicy: IfNotPresent`. Replace `my-registry` and `my-tag` in each manifest with your image, then load it on every eligible Kubernetes node or push it to a registry. The operator reads the Dynamo version from the image tag, so the manifests set no `runtimeVersionOverride`. Use a tag that holds the Dynamo version you built, such as `1.7.0`. If the tag has no version, such as `my-tag` or `latest`, the admission webhook rejects the manifest and asks for `runtimeVersionOverride`.

Create the Hugging Face token secret:

```bash
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="${HF_TOKEN}" \
  -n "${NAMESPACE}"
```

### DRA Templates

The files ending in `_dra.yaml` require:

1. Kubernetes v1.34 or later with the DRA v1 API enabled
2. The [Intel GPU resource driver](https://github.com/intel/intel-resource-drivers-for-kubernetes) with the `gpu.intel.com` device class

### Device Plugin Template

`disagg_xpu.yaml` requires the [Intel device plugins for Kubernetes](https://github.com/intel/intel-device-plugins-for-kubernetes) and allocates `gpu.intel.com/xe`.

## Key Differences from NVIDIA Templates

| Aspect | NVIDIA | Intel XPU |
|--------|--------|-----------|
| GPU Allocation | `nvidia.com/gpu` resource limit | DRA `ResourceClaimTemplate` or `gpu.intel.com/xe` resource limit |
| Device Target | Default (CUDA) | `--device xpu` flag |
| Attention Backend | Default | `--attention-backend intel_xpu` (the default Triton backend returned garbage tokens such as `!!!!` in testing) |
| DeviceClass | `nvidia.com` | `gpu.intel.com` |
| Disagg KV Transfer | Default | `hostIPC: true`, `UCX_TLS=ze_ipc,...` |

> [!NOTE]
> The SGLang workers run through `bash -c` and unset `DYN_FORWARDPASS_METRIC_PORT` before launch. The operator injects this variable, which turns on forward pass metrics, and the XPU SGLang build crashes on the first request because those metrics use `torch.cuda.Event`.

> [!NOTE]
> Do not hardcode `ZE_AFFINITY_MASK` in these examples. DRA or the Intel device plugin selects the allocated device.

`agg_router_xpu_dra.yaml` starts two workers and requires two GPUs.

## Deploy

```bash
kubectl apply \
  -f examples/backends/sglang/deploy/xpu/agg_xpu_dra.yaml \
  -n "${NAMESPACE}"

kubectl get resourceclaim -n "${NAMESPACE}"
kubectl get resourceslices

kubectl get dynamographdeployment -n "${NAMESPACE}"
kubectl get pods -n "${NAMESPACE}"
```

## Testing

```bash
kubectl port-forward deployment/sglang-agg-xpu-dra-frontend 8000:8000 \
  -n "${NAMESPACE}"

curl localhost:8000/v1/models
curl localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"Hello","max_tokens":20}'
```

## Further Reading

- [SGLang Kubernetes deployment configurations](../README.md)
