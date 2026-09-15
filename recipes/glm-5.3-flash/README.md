<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

This recipe references an upstream third-party vLLM container image. NVIDIA does not publish or distribute this image. Users should review the upstream image’s open-source license and codec terms before use or redistribution.

# GLM-5.3-Flash Recipes

Recipes for [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), a mixture-of-experts model with 320B total parameters, 18B active parameters, hybrid sparse and linear attention, and Manifold-Constrained Hyper-Connections (mHC).

## Configurations

Dynamo + vLLM deployment profiles for the GB200 and H200 agentic workload:

|                          | GB200 Aggregated              | GB200 Disaggregated                         | H200 Aggregated             | H200 Disaggregated                      |
| ------------------------ | ----------------------------- | ------------------------------------------- | --------------------------- | --------------------------------------- |
| **GPU**                  | 4x GB200                      | 4x GB200 prefill + 4x GB200 decode          | 8x H200                     | 8x H200 prefill + 8x H200 decode        |
| **Nodes**                | 1                             | 2                                           | 1                           | 2                                       |
| **Mode**                 | Aggregated                    | Prefill/decode disaggregated                | Aggregated                  | Prefill/decode disaggregated            |
| **Framework**            | vLLM                          | vLLM                                        | vLLM                        | vLLM                                    |
| **Precision**            | FP8-quantized checkpoint + BF16 KV cache | FP8-quantized checkpoint + BF16 KV cache | FP8-quantized checkpoint + BF16 KV cache | FP8-quantized checkpoint + BF16 KV cache |
| **Parallelism**          | TP4                           | TP4 prefill / TP4 decode                    | TP8                         | TP8 prefill / TP8 decode                |
| **Routing**              | KV-aware                      | KV-aware                                    | Round-robin                 | KV-aware                                |
| **Speculative decoding** | None                          | None                                        | MTP7                        | MTP7 on prefill and decode              |
| **Context length**       | Up to 1,048,576               | Up to 1,048,576                             | Up to 1,048,576             | Up to 1,048,576                         |
| **KV transfer**          | N/A                           | NIXL/UCX over TCP (MNNVL upgrade: see below) | N/A                         | NIXL with GPU buffers and RDMA resources |


## Supported features

- Modalities: Text; Image (up to 1 image per request, both agg and disagg)
- Reasoning
- Tool calling

## Prerequisites

1. **Dynamo Platform installed** — see [Kubernetes Deployment Guide](../../docs/fern/pages/kubernetes/getting-started/quickstart.mdx).
2. **GLM-5.3-Flash image**: `vllm/vllm-openai:glm53-flash` — a GLM-specific vLLM build with
   GLA/KDA attention kernels. `ai-dynamo` is pip-installed at pod startup.
3. **Hugging Face access** to `zai-org/GLM-5.3-Flash`.

## Quick Start

### 1. Create namespace and secret

```bash
export NAMESPACE=your-namespace
kubectl create namespace ${NAMESPACE}
kubectl label namespace ${NAMESPACE} kai.scheduler/enabled=true
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token" \
  -n ${NAMESPACE}
```

### 2. Create storage

> [!NOTE]
> Edit `model-cache/model-cache.yaml` and set `storageClassName` to a
> ReadWriteMany storage class available on the target cluster.

```bash
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}
```

### 3. Download the model

```bash
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=7200s
```

### 4. Deploy

```bash
SKU=gb200   # or h200
MODE=agg    # or disagg
kubectl apply -f vllm/${MODE}-${SKU}-agentic/deploy.yaml -n ${NAMESPACE}
```

## Limitations

- **GB200 disaggregated:** KV transport uses `cuda_copy+tcp` with the `glm53-flash` image. The
  image's UCX build does not support MNNVL IPC, so `^cuda_ipc` prevents a C-level crash at
  `uct_cuda_ipc_ep_get_zcopy`. To enable MNNVL NVLink KV transfer, rebuild the image on the
  `nvcr.io/nvidia/ai-dynamo/vllm-runtime` base and set `UCX_TLS: cuda_copy,cuda_ipc,tcp` and
  `UCX_CUDA_IPC_ENABLE_MNNVL: "y"`.
- **All targets:** A cuDNN workaround (`torch.backends.cudnn.enabled = False` via `sitecustomize.py`)
  prevents a segfault in GLM `_kpool_*` kernels during multimodal encoder profiling. This workaround
  does not affect image inference.
- **Both disaggregated targets:** `VLLM_SSM_CONV_STATE_LAYOUT=DS` and
  `VLLM_KV_CACHE_LAYOUT=HND` must match on both prefill and
  decode workers; mismatching these produces silent garbage output.
- **Both disaggregated targets:** `n>1` requests are not supported.
