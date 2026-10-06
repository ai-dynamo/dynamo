<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# MiniMax M3

These recipes serve `nvidia/MiniMax-M3-NVFP4` with vLLM on NVIDIA GB200 GPUs.
Both profiles use tensor parallelism (TP) 4, FP8 KV cache, EAGLE3 speculative
decoding, KV-aware routing, and a one-million-token context limit.

## Deployment Profiles

| Profile | Topology | GPUs | KV Storage |
| --- | --- | ---: | --- |
| `agg-gb200-agentic` | 3 aggregated TP4 replicas | 12x GB200 | GPU plus 400 GB CPU offload per replica |
| `disagg-gb200-agentic` | 3 prefill and 3 decode TP4 replicas | 24x GB200 | GPU with NIXL transfer between prefill and decode |

Both profiles use the `bea41d1a13` runtime image with the required
fixes built in, real EAGLE3 verification, and multimodal input enabled. No
runtime patches are applied. Each ConfigMap also contains a benchmark-only
synthetic-acceptance option; that option is not the production default.
See [Deploy](#deploy) for cluster setup and validation of either profile.

## Prepare the Model Cache

From the Dynamo repository root, set `CONTEXT` to your cluster context,
select a namespace, and create the download Job's required Hugging Face token
Secret. The token must have access to
`nvidia/MiniMax-M3-NVFP4` and `Inferact/MiniMax-M3-EAGLE3-GQA`:

```bash
export NAMESPACE=your-namespace
kubectl --context "${CONTEXT}" create namespace "$NAMESPACE"
kubectl --context "${CONTEXT}" create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token" \
  -n "$NAMESPACE"
```

If the namespace and Secret already exist, use them instead. Both deployment
profiles require a populated `shared-model-cache` PVC in that namespace unless
the cluster binding coordinates a different claim for download and serving.

Set `storageClassName` in `model-cache/model-cache.yaml`, then create the cache
with a ReadWriteMany storage class and download the pinned model and EAGLE3
checkpoints:

```bash
kubectl --context "${CONTEXT}" apply -f recipes/minimax-m3/model-cache/model-cache.yaml -n "$NAMESPACE"
kubectl --context "${CONTEXT}" apply -f recipes/minimax-m3/model-cache/model-download.yaml -n "$NAMESPACE"
kubectl --context "${CONTEXT}" wait --for=condition=complete job/minimax-m3-model-download \
  -n "$NAMESPACE" --timeout=14400s
```

## Deploy

The cluster requires the NVIDIA Dynamo platform, the ComputeDomain controller,
and the NVIDIA Dynamic Resource Allocation (DRA) driver. Each worker requests
four GB200 GPUs, 32 CPUs, and 448 GiB of host memory. Disaggregated workers must
share a compatible NVLink domain.

Set `CONTEXT` and `NAMESPACE` to your cluster context and namespace. From the
repository root, select a profile and the private cluster Kustomization:

```bash
export PROFILE=agg-gb200-agentic # or disagg-gb200-agentic
export RECIPE_DIR="$(pwd)/recipes/minimax-m3/vllm/$PROFILE"
export CLUSTER_KUSTOMIZATION=/absolute/path/to/private/minimax-m3-cluster
export DGD="minimax-m3-$PROFILE"
```

Prepare and render the cluster configuration as described in
[Edit and render](#edit-and-render), then apply the rendered manifest:

```bash
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" apply --dry-run=server \
  -f "$CLUSTER_KUSTOMIZATION/rendered.yaml"
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" apply \
  -f "$CLUSTER_KUSTOMIZATION/rendered.yaml"
```

## Edit and render

The profile's `deploy.yaml` is the portable source. For cluster-specific
settings, copy and fill the
[beta cluster Kustomization starter](https://github.com/ai-dynamo/dynamo/blob/main/recipes/templates/kustomize/README.md)
outside the repository. Point its `resources` at the absolute path to the
selected `$RECIPE_DIR/deploy.yaml`; write the resolved path, since YAML does
not expand shell variables. Select `/agg` or `/disagg` Components to match the
profile, and follow the starter's Component order and placeholder preflight.

Configure the cache claim, registry credentials, GB200 placement, and provider
networking in that private composition. Preserve the recipe's ComputeDomain
claim chain and, for disaggregated serving, its NIXL and UCX multi-node CUDA
IPC settings. The pinned image requires access to its private registry;
qualify any replacement image separately.

After editing the recipe or cluster bindings, validate and render from the
repository root with standalone Kustomize v5.8.1, Python 3.9 or later, and
PyYAML:

```bash
python3 scripts/validate-recipe-kustomization.py \
  "$RECIPE_DIR/deploy.yaml" \
  "$CLUSTER_KUSTOMIZATION/kustomization.yaml"
kustomize build --load-restrictor LoadRestrictionsNone \
  "$CLUSTER_KUSTOMIZATION" > "$CLUSTER_KUSTOMIZATION/rendered.yaml"
```

Use `LoadRestrictionsNone` only with reviewed paths. Inspect the rendered
ConfigMap, ComputeDomain, and DynamoGraphDeployment before applying them.
Keep filled cluster values and rendered manifests outside the repository.

## Smoke Test

Wait for the selected deployment, then forward its frontend port:

```bash
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" wait --for=condition=Ready \
  "dynamographdeployment/${DGD}" --timeout=7200s
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" port-forward \
  "service/${DGD}-frontend" 8000:8000
```

In another terminal, verify model discovery and a completion:

```bash
curl --fail-with-body http://localhost:8000/v1/models
curl --fail-with-body http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nvidia/MiniMax-M3-NVFP4","messages":[{"role":"user","content":"What is 2 + 2?"}],"max_tokens":1024}'
```

Expect HTTP 200 and a chat completion. For disaggregated serving, confirm
prefill-to-decode KV transfer uses the intended transport before benchmarking.
See the [benchmark workflow](https://github.com/ai-dynamo/dynamo/blob/main/recipes/minimax-m3/perf/README.md)
for trace staging and execution.


## Synthetic Acceptance for Benchmarks

Both profiles use the ConfigMap's `speculative-config` key for real EAGLE3
verification. For an explicit benchmark-only variant, use a private patch to
change the `SPECULATIVE_CONFIG` environment variable's
`valueFrom.configMapKeyRef.key` to `speculative-config-synthetic` on `Worker`
for aggregated serving, or on both `PrefillWorker` and `DecodeWorker` for
disaggregated serving. That configuration sets synthetic acceptance length to
`2.89`. Render and review the variant separately, and record the image and
rendered configuration with the results. The benchmark Job does not change
this setting.

> [!WARNING]
> Synthetic acceptance bypasses real EAGLE3 verification. Use it only for
> controlled performance experiments, never production responses or accuracy
> evaluation.
