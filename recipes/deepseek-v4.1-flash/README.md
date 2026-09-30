<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4.1-Flash Recipes

Serve `deepseek-ai/DeepSeek-V4.1-Flash` with NVIDIA Dynamo and vLLM on B200 or
GB200. Each target uses eight GPUs: two TP4 workers for aggregated serving, or
one TP4 prefill worker and one TP4 decode worker for disaggregated serving.
The existing SGLang recipes provide aggregated and disaggregated deployment
on GB200.

> [!WARNING]
> The refreshed vLLM targets are experimental and temporarily reference Azure
> CI images. Access to that registry is required. The public
> `1.6.0-deepseek-v4.1-flash-dev.2` runtime is not yet available; replacing an
> image requires validating the resulting deployment again.

## vLLM Targets

These entry points are Kustomize sources. Render them with Kustomize v5.8.1
and compose them with your cluster's scheduling, registry, storage, and network
settings. Rendered deployments are not checked in.

| Target | Kustomization | GPUs | Worker layout | KV transfer |
| --- | --- | ---: | --- | --- |
| B200 aggregated | [agg-b200-agentic](vllm/agg-b200-agentic/kustomization.yaml) | 8 | 2 × TP4 | Not applicable |
| B200 disaggregated | [disagg-b200-agentic](vllm/disagg-b200-agentic/kustomization.yaml) | 8 | 1P1D, TP4 per role | NIXL/UCX over InfiniBand |
| GB200 aggregated | [agg-gb200-agentic](vllm/agg-gb200-agentic/kustomization.yaml) | 8 | 2 × TP4 | Not applicable |
| GB200 disaggregated | [disagg-gb200-agentic](vllm/disagg-gb200-agentic/kustomization.yaml) | 8 | 1P1D, TP4 per role | NIXL/UCX; workers in one NVLink clique |

All four targets use expert parallelism, EPLB with the `torch_gloo`
communicator, `deep_gemm_mega_moe`, MXFP4 sparse-indexer KV, sparse-indexer
logits, and the `FLASHMLA_MEGA_ATTN_DSV41` attention backend. DSpark uses three
draft tokens with adaptive verification disabled. Prefix-cache retention is
1024. The recipes serve text and configure the `deepseek_v41` reasoning and
tool-call parsers.

B200 disaggregation uses the runtime's `nvfp4_ds_mla` KV default; the other
vLLM targets select `fp8_ds_mla`. See each Kustomization for runtime images
and engine settings.

## Prerequisites

- A Kubernetes cluster with a compatible Dynamo operator and Kustomize v5.8.1.
- Eight GPUs of the selected type, with four GPUs available to each worker.
  GB200 requires ARM64 nodes; B200 requires AMD64 nodes.
- A populated ReadWriteMany model-cache PVC. Workers reference
  `shared-model-cache` at `/shared-model-cache`; a private cluster
  Kustomization can bind a different physical PVC name.
- At least 512 GiB of host memory per worker and capacity for its 200 GiB
  shared-memory volume.
- Registry credentials supplied by the cluster configuration for the temporary
  Azure images.
- Disaggregated targets: working RDMA devices and a qualified UCX transport.
  Device resource names and interface bindings belong to the cluster configuration.
- GB200 disaggregated: NVIDIA DRA and ComputeDomain support, with both workers
  placed in one NVLink clique. The source contains the logical ComputeDomain
  and Pod claim references; the cluster owns placement and device realization.

## Prepare the Model Cache

Use the [model-cache manifests](model-cache/) to provision storage and download
the checkpoint. Select a storage class appropriate to your cluster before
applying them. The download Job pins snapshot
`dba1be0a40aa45a94ad051997016db3960a90277`.

All vLLM components run offline. Backend workers mount the populated cache;
the frontend obtains model metadata from the workers.

## Compose and Deploy

1. Select one target from the table above.
2. Copy and fill the [cluster Kustomization starter](../templates/kustomize/README.md).
   Point its resource at the selected target's `kustomize/base/deploy.yaml`.
3. Configure registry access, node placement, cache binding, and provider
   networking in that private copy. For B200 disaggregation, discover the
   InfiniBand resource key and optional `UCX_NET_DEVICES` list. For GKE GB200
   disaggregation, supply the RDMA network annotations/resources and the
   provider's GID index. Keep cluster identities outside this recipe.
4. Render, inspect, and validate the composition using the starter's instructions.
   Exercise the rendered configuration on its intended hardware before using
   it for performance measurements.

To inspect the portable source from the repository root:

```bash
kustomize build recipes/deepseek-v4.1-flash/vllm/agg-gb200-agentic
```

To apply a filled cluster composition:

```bash
export NAMESPACE=your-namespace
export CLUSTER_CONFIG=/path/to/your/filled-cluster-kustomization
set -o pipefail
kustomize build --load-restrictor LoadRestrictionsNone "${CLUSTER_CONFIG}" | \
  kubectl apply --dry-run=server -f - -n "${NAMESPACE}"
kustomize build --load-restrictor LoadRestrictionsNone "${CLUSTER_CONFIG}" | \
  kubectl apply -f - -n "${NAMESPACE}"
```

## Smoke Test

For GB200 aggregated serving:

```bash
kubectl port-forward svc/dsv41-flash-vllm-gb200-agg-frontend 8000:8000 -n "${NAMESPACE}"
```

Replace `gb200-agg` in the service name with `b200-agg`, `gb200-disagg`, or
`b200-disagg` for the other targets. In a separate terminal:

```bash
curl -sS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ai/DeepSeek-V4.1-Flash","messages":[{"role":"user","content":"Reply with exactly: READY"}],"temperature":0,"max_tokens":512}'
```

Verify the response and, for disaggregated serving, a prefill-to-decode KV transfer.

## Performance

Measured vLLM configurations on the 64K-input / 400-output agentic workload,
using eight GPUs per target. Output throughput includes reasoning tokens.
These are selected operating points, not a controlled topology-only comparison;
validation of the refreshed manifests is pending.

| Target | Concurrency | Output tok/s/GPU | Output tok/s/user p50 |
| --- | ---: | ---: | ---: |
| B200 aggregated | 168 | 990.57 | 54.69 |
| B200 disaggregated | 184 | 1,087.71 | 82.12 |
| GB200 aggregated | 168 | 953.08 | 51.83 |
| GB200 disaggregated | 168 | 1,154.87 | 80.85 |

At these operating points, disaggregated configurations show higher output
throughput and lower ITL, with longer TTFT tails. GB200 records 21% higher
output tok/s/GPU and 56% higher p50 output tok/s/user; TTFT p90 rises from
3.52 s to 57.12 s.

See [benchmark instructions and TTFT/ITL distributions](perf/README.md).

## SGLang Targets

The original SGLang targets remain available for GB200:

| Target | Deployment | GPUs | Layout |
| --- | --- | ---: | --- |
| Aggregated | [agg-gb200](sglang/agg-gb200/deploy.yaml) | 8 | 2 × TP4, EP4 |
| Disaggregated | [disagg-gb200](sglang/disagg-gb200/deploy-generic.yaml) | 8 | 1P1D, TP4 and EP4 per role |

These targets use the original SGLang `dev.1` runtime and have no published
performance measurements. Aggregated serving enables DSpark; disaggregated
serving uses Mooncake without speculative decoding. The
[GKE variant](sglang/disagg-gb200/deploy-gke-rdma.yaml) adds provider RDMA settings.

## Editing

Edit each vLLM target's `kustomize/base/deploy.yaml` and Kustomization files.
Build and validate both the portable source and your private cluster
composition. Keep rendered manifests and cluster qualification artifacts
outside the contribution. Update the recipe catalog and documentation when
the runtime, topology, or measured behavior changes.
