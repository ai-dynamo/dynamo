<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GAIE Deployments on Intel XPU

GAIE deployment templates for Intel XPU using Kubernetes Dynamic Resource Allocation (DRA).

## Available Templates

| File | Pattern | Description |
|------|---------|-------------|
| `agg.yaml` | Aggregated | Single decode worker + Epp, XPU via DRA |
| `disagg.yaml` | Disaggregated | Separate prefill/decode workers + Epp, XPU via DRA |

## Prerequisites

1. **Kubernetes v1.34+** with DRA API v1 enabled.
2. **[Intel resource drivers for Kubernetes](https://github.com/intel/intel-resource-drivers-for-kubernetes)** installed with DeviceClass `gpu.intel.com`.
3. **XPU runtime image** (`vllm-runtime-xpu`) built and tagged for the manifests:
   ```bash
   python container/render.py --framework=vllm --device=xpu --target=runtime
   docker build -t nvcr.io/nvidia/ai-dynamo/vllm-runtime-xpu:my-tag \
     -f container/vllm-runtime-xpu-amd64-rendered.Dockerfile .
   ```
4. **GAIE** installed in-cluster -- see [Gateway API docs](../../../../../../docs/kubernetes/gateway-api/README.mdx).
5. **HuggingFace token secret** (`hf-token-secret`).

## Deploy

```bash
export NAMESPACE=gaie-dynamo
kubectl apply -f agg.yaml -n $NAMESPACE       # or disagg.yaml
kubectl apply -f ../http-route.yaml -n $NAMESPACE

kubectl get resourceclaim -n $NAMESPACE
kubectl get dynamographdeployment -n $NAMESPACE
kubectl get pods -n $NAMESPACE
```

## See Also

- [GAIE Deployment Templates](../)
- [Intel XPU DRA Templates](../../xpu/)
