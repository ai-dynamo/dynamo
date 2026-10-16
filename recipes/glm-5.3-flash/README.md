<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Recipes for [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), a mixture-of-experts model with 320B total parameters, 18B active parameters, hybrid sparse and linear attention, and Manifold-Constrained Hyper-Connections (mHC).

## SGLang

Deploy GLM-5.3-Flash with NVIDIA Dynamo and SGLang using the B200 and GB200 Kustomize
overlays. These recipes use `RadixArk/GLM-5.3-Flash-NVFP4` at revision
`f46cf340d35a22d0d83d0c1dac8957cf2b1bcd35`.

| Setting | B200 Aggregated | GB200 Aggregated | GB200 Disaggregated |
| --- | --- | --- | --- |
| GPUs | 4x B200 on one node | 4x GB200 | 12x GB200: 8 prefill + 4 decode |
| CPU architecture | amd64 | arm64 | arm64 |
| Workers | 1x TP4 | 1x TP4 | 2x prefill TP4/EP4 + 1x decode TP4 |
| GPUs per worker Pod | 4 | 4 | 4 |
| Weight / KV precision | NVFP4 / FP8 E4M3 | NVFP4 / FP8 E4M3 | NVFP4 / FP8 E4M3 |
| Routing | Round-robin | Round-robin | KV-aware, worker KV events |
| Speculative decoding | EAGLE, 3 steps / 4 draft tokens, ReplaySSM | Same | Same, decode only |
| HiCache CPU capacity | 100 GB per TP rank; ~400 GB per worker Pod | Same | 100 GB per prefill TP rank; ~400 GB per prefill Pod; none on decode |
| Host memory request / limit per worker Pod | 448 GiB / 448 GiB | 448 GiB / 448 GiB | Prefill: 448 GiB / 448 GiB; decode: 256 GiB request, no explicit limit |
| Static memory fraction | 0.70 | 0.70 | Prefill 0.70 / decode 0.82 |
| Maximum running requests | 64 | 64 | Prefill 64 / decode 128 per worker |
| Maximum decode CUDA graph batch size | 64 | 64 | 128 on decode |
| KV transfer | Not applicable | Not applicable | Mooncake over NVLink with MNNVL |
| ComputeDomain | None | Created by overlay | Created by overlay |
| Overlay | `nscale` | `gb200` | `gb200` |
| Input modalities | Text, images, video | Text, images, video | Text, images |

Video input is unavailable in the SGLang disaggregated recipe because frontend decoding is unsupported on this path and the runtime image omits SGLang's video decoder dependencies.

### SGLang Prerequisites

- A Dynamo operator that supports `nvidia.com/v1beta1`, with its discovery and
  event services installed. See the [Kubernetes quickstart](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/kubernetes/getting-started/quickstart.mdx).
- **B200 aggregated:** one amd64 node with four available B200 GPUs. The
  `nscale` overlay uses the `inference` priority class and single-node NVLink/NVLS
  with MNNVL disabled; no ComputeDomain or DRA channel claims are required.
- **GB200:** arm64 nodes with four available GPUs per worker Pod, and the NVIDIA
  ComputeDomain CRD (`resource.nvidia.com/v1beta1`) and DRA driver installed.
  Disaggregated workers must fit in one NVLink clique. The scheduling component
  requires the `nvidia.com/gpu.clique` node label and uses Pod affinity to keep
  the workers together without embedding a site-specific clique ID.
- Standalone Kustomize v5.8.1 for reproducible rendering, and `kubectl`.
- Registry access to the pinned Dynamo image. Configure `imagePullSecrets`
  for frontend and worker Pods in your private overlay when authentication is required.
- A namespace containing a populated ReadWriteMany PVC named
  `shared-model-cache`, accessible to all worker nodes. The SGLang checkpoint
  differs from the checkpoint used by the sibling vLLM recipes.

The GB200 component selects `kubernetes.io/arch=arm64` for all Pods and
`nvidia.com/gpu.product=NVIDIA-GB200` for workers. It tolerates the
`kubernetes.io/arch=arm64:NoSchedule` taint and, on workers,
`nvidia.com/gpu=present:NoSchedule`.

The B200 `nscale` component selects `kubernetes.io/arch=amd64` for both Pods and
`nvidia.com/gpu.product=NVIDIA-B200` for the worker. Both Pods tolerate
`nvidia.com/gpu=true:NoSchedule`. See the [B200 overlay patches](sglang/agg-b200-agentic/kustomize/components/nscale/patch-dgd.yaml).

Adapt labels, taints, and priority class to your cluster in a private overlay.
Keep namespace, registry credentials, storage class, and physical network
bindings in that overlay.

### Populate the Model Cache

Run the commands from the repository root. Set `NAMESPACE` to an existing
namespace. Reuse `shared-model-cache` if it already exists. For a new cache,
set `storageClassName` in `sglang/model-cache/model-cache.yaml` to your cluster's
ReadWriteMany storage class before applying it:

```bash
export NAMESPACE=your-namespace
kubectl apply -f recipes/glm-5.3-flash/sglang/model-cache/model-cache.yaml -n "$NAMESPACE"
```

The download Job uses `hf-token-secret` when present; provide `HF_TOKEN` there
if the checkpoint requires authentication. Download the exact revision before
starting any of the three deployments:

```bash
kubectl apply -f recipes/glm-5.3-flash/sglang/model-cache/model-download.yaml -n "$NAMESPACE"
kubectl wait --for=condition=Complete job/glm53-flash-nvfp4-model-download \
  -n "$NAMESPACE" --timeout=7200s
```

The Job populates the Hugging Face cache under `/shared-model-cache`. All SGLang
workers resolve `--model-path RadixArk/GLM-5.3-Flash-NVFP4` and the pinned
`--revision` from that cache with offline mode enabled.
Frontends obtain model metadata from workers and do not mount the cache.

### Deploy

Choose one target. The disaggregated recipe runs two prefill workers and one
decode worker (2P1D), using 12 GB200 GPUs in one NVLink clique.

**B200 aggregated:**

```bash
export SKU=b200 MODE=agg OVERLAY=nscale
```

**GB200 aggregated:**

```bash
export SKU=gb200 MODE=agg OVERLAY=gb200
```

**GB200 disaggregated:**

```bash
export SKU=gb200 MODE=disagg OVERLAY=gb200
```

Render and apply the selected overlay:

```bash
export RECIPE="recipes/glm-5.3-flash/sglang/${MODE}-${SKU}-agentic"
kustomize build "$RECIPE/kustomize/overlays/$OVERLAY" > "/tmp/glm53-sglang-${MODE}-${SKU}.yaml"
kubectl apply --dry-run=server -f "/tmp/glm53-sglang-${MODE}-${SKU}.yaml" -n "$NAMESPACE"
kubectl apply -f "/tmp/glm53-sglang-${MODE}-${SKU}.yaml" -n "$NAMESPACE"
kubectl get dynamographdeployment,pods -n "$NAMESPACE"
```

Alternatively, apply the checked-in `"$RECIPE/deploy-${OVERLAY}.yaml"` manifest,
which is generated from the same overlay. The GB200 overlays create a
ComputeDomain and wire the worker claims to its channel template. The B200
`nscale` overlay creates only the DynamoGraphDeployment.

Use `kubectl apply -k` only when its embedded Kustomize version matches v5.8.1.
Rendering uses the checkout's shared OpenAPI component; it does not fetch a
schema from GitHub.

### Smoke Test

Wait for the frontend and all workers to become ready, then forward the
frontend Service in one terminal:

```bash
kubectl port-forward "svc/glm53-flash-sglang-${MODE}-${SKU}-frontend" \
  8000:8000 -n "$NAMESPACE"
```

In another terminal, verify model registration and a completion:

```bash
curl --fail-with-body http://localhost:8000/v1/models
curl --fail-with-body http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"RadixArk/GLM-5.3-Flash-NVFP4","messages":[{"role":"user","content":"What is 2 + 2?"}],"max_tokens":128,"temperature":0}'
```

### Configuration Layout

Each target follows the same layout:

```text
sglang/<mode>-<sku>-agentic/
├── .kustomize-matrix.yaml
├── deploy-<overlay>.yaml                     # generated
└── kustomize/
    ├── base/
    │   ├── deploy.yaml
    │   └── kustomization.yaml
    ├── components/                          # target-specific patches
    └── overlays/<overlay>/kustomization.yaml # generated
```

Edit `kustomize/base/deploy.yaml` for serving settings. GB200 components cover
`scheduling`, `compute-domain`, and optional `synthetic-acceptance`. The B200
`nscale` component covers placement and worker probes; B200 also has an optional
`synthetic-acceptance` component.
Regenerate the public overlay and manifest after edits:

```bash
python3 scripts/kustomize-matrix.py unfold "$RECIPE/.kustomize-matrix.yaml"
python3 scripts/kustomize-matrix.py render "$RECIPE/.kustomize-matrix.yaml"
python3 scripts/kustomize-matrix.py check "$RECIPE/.kustomize-matrix.yaml"
```

### Changes from the Source Configurations

- All three recipes use canonical worker cache names, exec-form commands, and
  the standard beta worker security context. All workers enable multimodal
  processing and `--mm-enable-dp-encoder`. Aggregated workers enable frontend
  media decoding; disaggregated workers omit `--frontend-decoding`.
- Every worker base sets `DYN_HEALTH_CHECK_REQUEST_TIMEOUT=300` and a liveness
  probe on `/live`, port `system`, with `periodSeconds: 5`, `timeoutSeconds: 4`,
  and `failureThreshold: 60`.
- The disaggregated decode worker uses a static memory fraction of `0.82`,
  at most 128 running requests, and decode CUDA graphs up to batch size 128.
- GB200 disaggregated routing uses worker KV events. Both prefill and decode
  workers configure a ZMQ publisher with topic `kv-events` and endpoint
  `tcp://*:5557`. The router block size and worker page size are both 64.

### Synthetic Acceptance

Real speculative verification is the default for all three targets. To use
synthetic acceptance on B200 or GB200, select a target below
and compose its optional component. The component sets
`SGLANG_SIMULATE_ACC_LEN=2.5657`, `SGLANG_SIMULATE_ACC_METHOD=match-expected`,
and `SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token` on the aggregated worker
or disaggregated decode worker.

```bash
export TARGET=agg-b200-agentic
export OVERLAY=nscale
export RECIPE="recipes/glm-5.3-flash/sglang/${TARGET}"
python3 scripts/kustomize-matrix.py compose \
  "$RECIPE/kustomize/overlays/$OVERLAY" \
  "$RECIPE/kustomize/components/synthetic-acceptance" \
  > "/tmp/glm53-sglang-${TARGET}-synthetic.yaml"
kubectl apply --dry-run=server -f "/tmp/glm53-sglang-${TARGET}-synthetic.yaml" -n "$NAMESPACE"
kubectl apply -f "/tmp/glm53-sglang-${TARGET}-synthetic.yaml" -n "$NAMESPACE"
```

Synthetic acceptance changes token verification and is for performance
experiments only. Do not use its responses for accuracy evaluation or compare
its throughput directly with real-verification runs.

## vLLM

This recipe references an upstream third-party vLLM container image. NVIDIA does not publish or distribute this image. Users should review the upstream image’s open-source license and codec terms before use or redistribution.

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


### Supported features

- Modalities: Text; Image (up to 1 image per request, both agg and disagg)
- Reasoning
- Tool calling

### vLLM Prerequisites

1. **Dynamo Platform installed** — see [Kubernetes Deployment Guide](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/kubernetes/getting-started/quickstart.mdx).
2. **GLM-5.3-Flash image**: `vllm/vllm-openai:glm53-flash` — a GLM-specific vLLM build with
   GLA/KDA attention kernels. `ai-dynamo` is pip-installed at pod startup.
3. **Hugging Face access** to `zai-org/GLM-5.3-Flash`.

### Quick Start

#### 1. Create namespace and secret

```bash
export NAMESPACE=your-namespace
kubectl create namespace ${NAMESPACE}
kubectl label namespace ${NAMESPACE} kai.scheduler/enabled=true
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token" \
  -n ${NAMESPACE}
```

#### 2. Create storage

> [!NOTE]
> Edit `recipes/glm-5.3-flash/model-cache/model-cache.yaml` and set `storageClassName` to a
> ReadWriteMany storage class available on the target cluster.

```bash
kubectl apply -f recipes/glm-5.3-flash/model-cache/model-cache.yaml -n ${NAMESPACE}
```

#### 3. Download the model

```bash
kubectl apply -f recipes/glm-5.3-flash/model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=7200s
```

#### 4. Deploy

```bash
SKU=gb200   # or h200
MODE=agg    # or disagg
kubectl apply -f recipes/glm-5.3-flash/vllm/${MODE}-${SKU}-agentic/deploy.yaml -n ${NAMESPACE}
```

### Limitations

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

## Benchmarking

Use the [AIPerf benchmark recipe](perf/README.md) for agentic text trace replay against
the SGLang B200 and GB200 deployments. It includes trace staging, target selection,
concurrency settings, and artifact retrieval.
