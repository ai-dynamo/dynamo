<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# A.X K2 NVFP4 vLLM Recipes

These experimental recipes serve
[`skt/A.X-K2-NVFP4`](https://huggingface.co/skt/A.X-K2-NVFP4) with Dynamo and
the `axk2-v0.23.0` branch of the SKT-AI vLLM fork. The checkpoint quantizes
routed experts to NVFP4 W4A4, keeps attention and other compute layers in FP8,
and keeps embeddings and the LM head in BF16.

## Deployment Targets

| Variant | Topology | GPUs | Parallelism |
|---|---|---:|---|
| [`agg-b200`](vllm/agg-b200/deploy.yaml) | Aggregated | 4 B200 | TP4 + expert parallel |
| [`disagg-b200`](vllm/disagg-b200/deploy.yaml) | 1 prefill + 1 decode | 8 B200 | TP4 + expert parallel per worker |

Both variants expose the model as `A.X-K2-NVFP4`, select the
`FLASHINFER_MLA_SPARSE` attention backend, enable FlashInfer NVFP4 MoE kernels,
use the `hermes` tool-call parser and `deepseek_v3` reasoning parser, and mount
the `shared-model-cache` ReadWriteMany (RWX) PersistentVolumeClaim (PVC) at
`/models`.

## Prerequisites

- Dynamo Platform with the `nvidia.com/v1beta1` `DynamoGraphDeployment` API
- One 4-GPU B200 node for aggregated serving
- Two 4-GPU B200 nodes and the `rdma/ib` device plugin for disaggregated serving
- An existing `shared-model-cache` RWX PVC
- The A.X K2 NVFP4 checkpoint downloaded into that PVC
- A registry-accessible custom runtime built from [`container/README.md`](container/README.md)
- A `hf-token-secret` secret in the deployment namespace

## Download the Model

The download Job uses the existing PVC. Do not apply
`model-cache/model-cache.yaml` when `shared-model-cache` already exists.

```bash
export NAMESPACE=<your-namespace>

kubectl apply \
  -f model-cache/model-download.yaml \
  -n "${NAMESPACE}"

kubectl wait \
  --for=condition=Complete \
  job/model-download \
  -n "${NAMESPACE}" \
  --timeout=14400s
```

The checkpoint occupies about 398 GB on disk. The Job stores it under the
standard Hugging Face cache path in `/models/hub`.

## Configure the Runtime Image

Replace every placeholder image in the selected manifest with the image built
from [`container/README.md`](container/README.md):

```bash
export AXK2_IMAGE=<your-registry>/dynamo/vllm-runtime:axk2-v0.23.0
export RECIPE=vllm/agg-b200

yq -i \
  '(.spec.components[].podTemplate.spec.containers[] | select(.name == "main").image) = strenv(AXK2_IMAGE)' \
  "${RECIPE}/deploy.yaml"
```

For disaggregated serving, set `RECIPE=vllm/disagg-b200`. If the cluster uses a
different RDMA extended resource, replace `rdma/ib` in that manifest before
deployment.

## Deploy

Start with aggregated serving to validate the custom vLLM image and model
implementation:

```bash
export DEPLOYMENT=axk2-vllm-agg-b200   # axk2-vllm-disagg-b200 for disaggregated serving

kubectl apply -f "${RECIPE}/deploy.yaml" -n "${NAMESPACE}"

kubectl get pods \
  -l nvidia.com/dynamo-graph-deployment-name="${DEPLOYMENT}" \
  -n "${NAMESPACE}" \
  --watch
```

The first worker startup compiles model kernels and can take up to 90 minutes.
`VLLM_CACHE_ROOT`, `TRITON_CACHE_DIR`, and `FLASHINFER_WORKSPACE_BASE` point
into `shared-model-cache` so subsequent workers can reuse the generated
artifacts.

## Smoke Test

```bash
kubectl port-forward \
  "svc/${DEPLOYMENT}-frontend" \
  8000:8000 \
  -n "${NAMESPACE}"
```

In another terminal:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "A.X-K2-NVFP4",
    "messages": [{"role": "user", "content": "A.X K2를 한 문장으로 소개해줘."}],
    "max_tokens": 256,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

For thinking mode, set `chat_template_kwargs.enable_thinking` to `true`.

## Validation Status

The model publisher documents native vLLM serving on 4 B200 GPUs. The custom
Dynamo image performs import, compiled-extension, NIXL, and parser checks during
the build. The aggregated and disaggregated Dynamo deployments still require a
B200 smoke test; treat both as experimental until that validation is recorded.
