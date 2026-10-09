<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## GLM-5.3-Flash NVFP4 on B200 and GB200

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
| HiCache CPU capacity | 100 GiB | 100 GiB | 100 GiB per prefill worker |
| Static memory fraction | 0.70 | 0.70 | Prefill 0.70 / decode 0.82 |
| Maximum running requests | 64 | 64 | Prefill 64 / decode 128 per worker |
| Maximum decode CUDA graph batch size | 64 | 64 | 128 on decode |
| KV transfer | Not applicable | Not applicable | Mooncake over NVLink with MNNVL |
| ComputeDomain | None | Created by overlay | Created by overlay |
| Overlay | `nscale` | `gb200` | `gb200` |
| Input modalities | Text, images, video | Text, images, video | Text, images, video |

## Prerequisites

- A Dynamo operator that supports `nvidia.com/v1beta1`, with its discovery and
  event services installed. See the [Kubernetes quickstart](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/kubernetes/getting-started/quickstart.mdx).
- **B200 aggregated:** one amd64 node with four available B200 GPUs. The
  `nscale` overlay uses the `inference` priority class and single-node NVLink/NVLS
  with MNNVL disabled; no ComputeDomain or DRA channel claims are required.
- **GB200:** arm64 nodes with four available GPUs per worker Pod, and the NVIDIA
  ComputeDomain CRD (`resource.nvidia.com/v1beta1`) and DRA driver installed.
  Disaggregated workers must fit in one NVLink clique. The scheduling component
  requires the `nvidia.com/gpu.clique` node label and uses Pod affinity to keep
  the three workers together without embedding a site-specific clique ID.
- Standalone Kustomize v5.8.1 for reproducible rendering, and `kubectl`.
- Registry access to the pinned Dynamo image. Configure `imagePullSecrets`
  for frontend and worker Pods in your private overlay when authentication is required.
- A namespace containing a populated ReadWriteMany PVC named
  `shared-model-cache`, accessible to all worker nodes and the B200 frontend. The SGLang checkpoint
  differs from the checkpoint used by the sibling vLLM recipes.

The GB200 component selects `kubernetes.io/arch=arm64` for all Pods and
`nvidia.com/gpu.product=NVIDIA-GB200` for workers. It tolerates the
`kubernetes.io/arch=arm64:NoSchedule` taint and, on workers,
`nvidia.com/gpu=present:NoSchedule`.

The B200 `nscale` component selects `kubernetes.io/arch=amd64` for both Pods and
`nvidia.com/gpu.product=NVIDIA-B200` for the worker. Both Pods tolerate
`nvidia.com/gpu=true:NoSchedule`. It also mounts the model cache in the frontend
and sets worker health probes. See the [B200 overlay patches](agg-b200-agentic/kustomize/components/nscale/patch-dgd.yaml).

Adapt labels, taints, and priority class to your cluster in a private overlay.
Keep namespace, registry credentials, storage class, and physical network
bindings in that overlay.

## Populate the Model Cache

Run the commands from the repository root. Set `NAMESPACE` to an existing
namespace. Reuse `shared-model-cache` if it already exists. For a new cache,
set `storageClassName` in `model-cache/model-cache.yaml` to your cluster's
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

The Job populates the Hugging Face cache under `/shared-model-cache`. GB200 workers
resolve `--model-path RadixArk/GLM-5.3-Flash-NVFP4` and the pinned `--revision`
from that cache with offline mode enabled. GB200 frontends obtain model
metadata from workers and do not mount the cache. The B200 overlay mounts the
cache in both frontend and worker Pods and passes the pinned snapshot path
explicitly to both, also with offline mode enabled.

## Deploy

Choose one target:

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

## Smoke Test

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

## Configuration Layout

Each target follows the same layout:

```text
<mode>-<sku>-agentic/
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
`nscale` component covers placement, frontend cache access, and worker probes.
Regenerate the public overlay and manifest after edits:

```bash
python3 scripts/kustomize-matrix.py unfold "$RECIPE/.kustomize-matrix.yaml"
python3 scripts/kustomize-matrix.py render "$RECIPE/.kustomize-matrix.yaml"
python3 scripts/kustomize-matrix.py check "$RECIPE/.kustomize-matrix.yaml"
```

## Changes from the Source Configurations

- All three recipes use canonical worker cache names, exec-form commands, and
  the standard beta worker security context. All workers enable multimodal
  processing and frontend media decoding.
- The disaggregated decode worker uses a static memory fraction of `0.82`,
  at most 128 running requests, and decode CUDA graphs up to batch size 128.
- The GB200 recipes omit experiment labels, namespace, pull Secrets, priority
  class, and physical ComputeDomain and clique IDs. Components provide GB200
  placement and fresh ComputeDomains. The B200 `nscale` component sets the
  `inference` priority class and creates no ComputeDomain.
- On GB200, operator defaults supply health probes. The source's custom startup,
  readiness, and liveness budgets are not carried over. If the cluster needs
  longer budgets, use the optional `probes` component from the
  [beta cluster starter](https://github.com/ai-dynamo/dynamo/tree/main/recipes/templates/kustomize).
- The B200 `nscale` component sets explicit worker startup and readiness probes
  on `/health` and a liveness probe on `/live`.
- GB200 disaggregated routing enables `--router-kv-events`. Both prefill and decode
  workers configure a ZMQ publisher with topic `kv-events` and endpoint
  `tcp://*:5557`. The router block size and worker page size are both 64.
- Real speculative verification is the default for all three targets. For
  GB200, reproduce the source's synthetic acceptance length of `2.5657` by
  selecting the aggregated or disaggregated recipe and composing its optional
  component:

```bash
export MODE=agg  # or disagg
export RECIPE="recipes/glm-5.3-flash/sglang/${MODE}-gb200-agentic"
python3 scripts/kustomize-matrix.py compose \
  "$RECIPE/kustomize/overlays/gb200" \
  "$RECIPE/kustomize/components/synthetic-acceptance" \
  > "/tmp/glm53-sglang-${MODE}-synthetic.yaml"
```

Synthetic acceptance changes token verification and is for performance
experiments only. Do not use its responses for accuracy evaluation or compare
its throughput directly with real-verification runs.
