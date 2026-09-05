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

Build the XPU runtime image from the current source:

```bash
python3 container/render.py --framework=sglang --device=xpu --target=runtime
docker build -t docker.io/library/dynamo-sglang-xpu:latest \
  -f container/sglang-runtime-xpu-amd64-rendered.Dockerfile .
```

For `disagg_planner_xpu_dra.yaml`, also build the Planner image:

```bash
python3 container/render.py --framework=dynamo --target=planner
docker build -t docker.io/library/dynamo-planner:latest \
  -f container/dynamo-planner-cuda13.0-amd64-rendered.Dockerfile .
```

The Planner template uses `optimization_target: sla` and requires Prometheus. Install `kube-prometheus-stack` as described in the [Dynamo Platform installation guide](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/kubernetes/installation/install-dynamo.md). Planner uses `http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090` by default. If your Prometheus service uses a different address, set `PROMETHEUS_ENDPOINT` in the Planner container.

The manifests use `imagePullPolicy: IfNotPresent`. Load these images on every eligible Kubernetes node, or push them to a registry and update the image references. If you build from a Dynamo version other than 1.4.0, update each `runtimeVersionOverride` to match.

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
| CUDA Graph | Enabled | `--disable-cuda-graph` |
| DeviceClass | `nvidia.com` | `gpu.intel.com` |
| Disagg KV Transfer | Default | `hostIPC: true`, `UCX_TLS=ze_ipc,...` |

> [!NOTE]
> Do not hardcode `ZE_AFFINITY_MASK` in these examples. DRA or the Intel device plugin selects the allocated device.

`agg_router_xpu_dra.yaml` starts two workers and requires two GPUs. `disagg_planner_xpu_dra.yaml` starts two decode workers and one prefill worker and requires three GPUs.

The Planner template includes bootstrap profile data for `Qwen/Qwen3-0.6B` because the current XPU SGLang build does not publish live forward pass metrics. Replace this data with measurements from your target XPU hardware before tuning production scaling.

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
