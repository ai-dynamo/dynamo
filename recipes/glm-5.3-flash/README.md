<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GLM-5.3-Flash Recipes

Recipes for [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash).

## Configurations

Dynamo + vLLM deployment profiles for the GB200 and H200 agentic workload:

|                          | GB200 aggregated agentic        | GB200 disaggregated agentic              | H200 aggregated agentic | H200 disaggregated agentic |
| ------------------------ | ------------------------------- | ---------------------------------------- | ----------------------- | -------------------------- |
| **GPU** (per worker)     | 4x GB200                        | 4x GB200 prefill + 4x GB200 decode       | TBD                     | TBD                        |
| **Mode**                 | Aggregated                      | Prefill/decode disaggregated             | Aggregated              | Prefill/decode disaggregated |
| **Framework**            | vLLM                            | vLLM                                     | vLLM                    | vLLM                       |
| **Precision**            | bf16 + fp8 KV                   | bf16 + bf16 KV                           | TBD                     | TBD                        |
| **Parallelism**          | TP4                             | TP4 prefill / TP4 decode                 | TBD                     | TBD                        |
| **Routing**              | KV-aware                        | KV-aware                                 | KV-aware                | KV-aware                   |
| **Speculative decoding** | None                            | None                                     | None                    | None                       |
| **Context length**       | 128,000                         | 128,000                                  | TBD                     | TBD                        |
| **KV transfer**          | N/A                             | NIXL/UCX over tcp (MNNVL upgrade: see README) | N/A              | TBD                        |


## Supported features

- Modalities: Text (multimodal inputs present in model but disabled in these recipes)
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

### 4. Deploy the DGD

```bash
SKU=gb200   # or h200
MODE=agg    # or disagg
kubectl apply -f vllm/${MODE}-${SKU}-agentic/deploy.yaml -n ${NAMESPACE}
```

### 5. Benchmark

See [perf/README.md](perf/README.md) for the full benchmark workflow.

## Optimization targets

| Workload | Median ISL | Median OSL | KV cache hit rate | User output tok/s |
| -------- | ---------- | ---------- | ----------------- | ----------------- |
| Agentic  | 64k        | 400        | 90%               | TBD               |

## Performance results

| Workload | Recipe | SKU | Concurrency | System output tok/s/gpu | User output tok/s (P50) | TTFT P50 (ms) |
| -------- | ------ | --- | ----------- | ----------------------- | ----------------------- | ------------- |
|          |        |     |             |                         |                         |               |

## Limitations

- GB200 disagg KV transport uses `cuda_copy+tcp` with the `glm53-flash` image. The image's UCX
  build does not support MNNVL IPC, so `^cuda_ipc` is required to prevent a C-level crash at
  `uct_cuda_ipc_ep_get_zcopy`. To enable MNNVL NVLink KV transfer, rebuild the image on
  `nvcr.io/nvidia/ai-dynamo/vllm-runtime` base and switch to
  `UCX_TLS: cuda_copy,cuda_ipc,tcp` + `UCX_CUDA_IPC_ENABLE_MNNVL: "y"`.
- A cudnn workaround (`torch.backends.cudnn.enabled = False` via `sitecustomize.py`) is required
  in the disagg recipe to prevent a segfault in GLM `_kpool_*` kernels on GB200.
- `VLLM_SSM_CONV_STATE_LAYOUT=DS` and `VLLM_KV_CACHE_LAYOUT=HND` must match on both prefill and
  decode workers; mismatching these produces silent garbage output.
- H200 recipes are in progress.
- `n>1` requests are not supported with the disaggregated recipe.
