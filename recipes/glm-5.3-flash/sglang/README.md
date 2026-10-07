<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## GLM-5.3-Flash NVFP4 on GB200

Deploy GLM-5.3-Flash with NVIDIA Dynamo and SGLang using the GB200 Kustomize
overlays. These recipes use `RadixArk/GLM-5.3-Flash-NVFP4` at revision
`f46cf340d35a22d0d83d0c1dac8957cf2b1bcd35`. All frontend and worker containers use
`dynamoci.azurecr.io/ai-dynamo/dynamo:1.6.0-ci-d98da56221899faf0f81e340eff1cf1186ca201d-sglang-runtime`.

This Dynamo image builds on `lmsysorg/sglang:v0.5.21-cu130-runtime` with the
GLM-5.3-Flash backports, Transformers 5.19.0, and Tokenizers 0.23.2. It includes
the Dynamo entry points and `Glm5NextProcessor` image/video processing support.
Both recipes enable `--enable-multimodal` and `--frontend-decoding` on all
SGLang workers: the Dynamo frontend decodes images and videos before passing
them to the backend.

| Setting | Aggregated | Disaggregated |
| --- | --- | --- |
| Source configuration | E51 TP4 ReplaySSM + HiCache 100 | D31 2P1D |
| GPUs | 4x GB200 | 12x GB200: 8 prefill + 4 decode |
| Workers | 1x TP4 | 2x prefill TP4/EP4 + 1x decode TP4 |
| GPUs per worker Pod | 4 | 4 |
| Weight / KV precision | NVFP4 / FP8 E4M3 | NVFP4 / FP8 E4M3 |
| Routing | Round-robin | KV-aware, prediction-based |
| Speculative decoding | EAGLE, 3 steps / 4 draft tokens, ReplaySSM | Same, decode only |
| HiCache CPU capacity | 100 GiB | 100 GiB per prefill worker |
| Static memory fraction | 0.70 | Prefill 0.70 / decode 0.78 |
| Maximum running requests | 64 | Prefill 64 / decode 256 per worker |
| KV transfer | Not applicable | Mooncake over NVLink with MNNVL |
| Input modalities | Text, images, video | Text, images, video |

## Prerequisites

- A Dynamo operator that supports `nvidia.com/v1beta1`, with its discovery and
  event services installed. See the [Kubernetes quickstart](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/kubernetes/getting-started/quickstart.mdx).
- GB200 arm64 nodes with four available GPUs per worker Pod, and the NVIDIA
  ComputeDomain CRD (`resource.nvidia.com/v1beta1`) and DRA driver installed.
  Disaggregated workers must fit in one NVLink clique. The scheduling component
  requires the `nvidia.com/gpu.clique` node label and uses Pod affinity to keep
  the three workers together without embedding a site-specific clique ID.
- Standalone Kustomize v5.8.1 for reproducible rendering, and `kubectl`.
- Registry access to the pinned Dynamo image. Configure `imagePullSecrets`
  for frontend and worker Pods in your private overlay when authentication is required.
- A namespace containing a populated ReadWriteMany PVC named
  `shared-model-cache`, accessible to all worker nodes. The SGLang checkpoint
  differs from the checkpoint used by the sibling vLLM recipes.

The GB200 component selects `kubernetes.io/arch=arm64` for all Pods and
`nvidia.com/gpu.product=NVIDIA-GB200` for workers. It tolerates the
`kubernetes.io/arch=arm64:NoSchedule` taint and, on workers,
`nvidia.com/gpu=present:NoSchedule`. Adapt these policies in a private cluster
overlay if your cluster uses different labels or taints. Keep namespace,
priority class, registry credentials, storage class, and any physical network
bindings in that private overlay.

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
starting either deployment:

```bash
kubectl apply -f recipes/glm-5.3-flash/sglang/model-cache/model-download.yaml -n "$NAMESPACE"
kubectl wait --for=condition=Complete job/glm53-flash-nvfp4-model-download \
  -n "$NAMESPACE" --timeout=7200s
```

The Job populates the Hugging Face cache under `/shared-model-cache`. Workers
load the pinned snapshot from that path with offline mode enabled. Frontends
obtain model metadata from workers and do not mount the cache.

## Deploy

Select `agg` or `disagg`. Each overlay creates its own ComputeDomain and wires
the worker Pod claims and container claims to the matching channel template.

```bash
export MODE=agg
export RECIPE="recipes/glm-5.3-flash/sglang/${MODE}-gb200-agentic"
kustomize build "$RECIPE/kustomize/overlays/gb200" > "/tmp/glm53-sglang-${MODE}.yaml"
kubectl apply --dry-run=server -f "/tmp/glm53-sglang-${MODE}.yaml" -n "$NAMESPACE"
kubectl apply -f "/tmp/glm53-sglang-${MODE}.yaml" -n "$NAMESPACE"
kubectl get dynamographdeployment,pods -n "$NAMESPACE"
```

Alternatively, apply the checked-in `"$RECIPE/deploy-gb200.yaml"` manifest,
which is generated from the same overlay. Use `kubectl apply -k` only when its
embedded Kustomize version matches v5.8.1. Rendering uses the checkout's shared
OpenAPI component; it does not fetch a schema from GitHub.

## Smoke Test

Wait for the frontend and all workers to become ready, then forward the
frontend Service in one terminal:

```bash
kubectl port-forward "svc/glm53-flash-sglang-${MODE}-gb200-frontend" \
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

Each mode follows the same layout:

```text
<mode>-gb200-agentic/
├── .kustomize-matrix.yaml
├── deploy-gb200.yaml                         # generated
└── kustomize/
    ├── base/
    │   ├── deploy.yaml
    │   └── kustomization.yaml
    ├── components/
    │   ├── scheduling/
    │   ├── compute-domain/
    │   └── synthetic-acceptance/             # optional, benchmark only
    └── overlays/gb200/kustomization.yaml     # generated
```

Edit the base for serving settings and the components for GB200 placement or
ComputeDomain wiring. Regenerate the public overlay and manifest after edits:

```bash
python3 scripts/kustomize-matrix.py unfold "$RECIPE/.kustomize-matrix.yaml"
python3 scripts/kustomize-matrix.py render "$RECIPE/.kustomize-matrix.yaml"
python3 scripts/kustomize-matrix.py check "$RECIPE/.kustomize-matrix.yaml"
```

## Changes from the Source Configurations

- The source engine tuning, replicas, GPU counts, memory requests, and pinned
  checkpoint are retained. The recipe uses canonical worker cache
  names, exec-form commands, and the standard beta worker security context.
- The source image is replaced by the patched Dynamo SGLang runtime image
  above. All workers enable multimodal processing and frontend media decoding.
- Experiment labels, namespace, pull Secrets, priority class, and physical
  ComputeDomain and clique IDs are omitted. GB200 placement and fresh
  ComputeDomains are composed through components.
- Operator defaults supply health probes. The source's custom startup,
  readiness, and liveness budgets are not carried over. If the cluster needs
  longer budgets, use the optional `probes` component from the
  [beta cluster starter](https://github.com/ai-dynamo/dynamo/tree/main/recipes/templates/kustomize).
- Disaggregated routing explicitly disables KV events because the source
  workers do not configure a publisher. The router uses predictions with a
  block size of 64, matching the workers' page size.
- Real speculative verification is the default. To reproduce the source's
  synthetic acceptance length of `2.5657`, compose the optional component:

```bash
python3 scripts/kustomize-matrix.py compose \
  "$RECIPE/kustomize/overlays/gb200" \
  "$RECIPE/kustomize/components/synthetic-acceptance" \
  > "/tmp/glm53-sglang-${MODE}-synthetic.yaml"
```

Synthetic acceptance changes token verification and is for performance
experiments only. Do not use its responses for accuracy evaluation or compare
its throughput directly with real-verification runs.
