<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4.1-Flash Recipes

Serve `deepseek-ai/DeepSeek-V4.1-Flash` with NVIDIA Dynamo and vLLM on B200 or
GB200. These targets use eight GPUs: two TP4 workers for aggregated serving, or
one TP4 prefill worker and one TP4 decode worker for disaggregated serving.
H200 aggregated serving uses four TP4 workers (16 GPUs).
The existing SGLang recipes provide aggregated and disaggregated deployment
on GB200.

## vLLM Targets

These entry points are Kustomize sources. Render them with Kustomize v5.8.1
and compose them with your cluster's scheduling, registry, storage, and network
settings.

| Target | Kustomization | GPUs | Worker layout | KV transfer |
| --- | --- | ---: | --- | --- |
| B200 aggregated | [agg-b200-agentic](vllm/agg-b200-agentic/kustomization.yaml) | 8 | 2 × TP4 | Not applicable |
| B200 disaggregated | [disagg-b200-agentic](vllm/disagg-b200-agentic/kustomization.yaml) | 8 | 1P1D, TP4 per role | NIXL/UCX over InfiniBand |
| GB200 aggregated | [agg-gb200-agentic](vllm/agg-gb200-agentic/kustomization.yaml) | 8 | 2 × TP4 | Not applicable |
| GB200 disaggregated | [disagg-gb200-agentic](vllm/disagg-gb200-agentic/kustomization.yaml) | 8 | 1P1D, TP4 per role | NIXL/UCX; workers in one NVLink clique |
| H200 aggregated | [agg-h200-agentic](vllm/agg-h200-agentic/kustomization.yaml) | 16 | 4 × TP4 | Not applicable |

## Configurations

All vLLM targets use four-way tensor parallelism (TP4), expert parallelism,
expert load balancing (EPLB), KV-aware routing,
and DSpark with three draft tokens, block verification, and adaptive
verification disabled. They serve text with `deepseek_v41` reasoning and
tool-call parsers. 1P1D means one prefill worker and one decode worker.
P/D below means prefill/decode; engine settings apply to each worker.

| Setting | B200 / GB200 aggregated | B200 disaggregated | GB200 disaggregated | H200 aggregated |
| --- | --- | --- | --- | --- |
| Total GPUs | 8 | 8 | 8 | 16 |
| Worker layout | 2 × TP4 | 1P1D, TP4 per role | 1P1D, TP4 per role | 4 × TP4 |
| MoE backend | `deep_gemm_mega_moe` | `deep_gemm_mega_moe` | `deep_gemm_mega_moe` | `flashinfer_cutlass` |
| KV cache dtype | `fp8_ds_mla` | Auto → `nvfp4_ds_mla` | `fp8_ds_mla` | `fp8_ds_mla` |
| Attention backend | `FLASHMLA_MEGA_ATTN_DSV41` | `FLASHMLA_MEGA_ATTN_DSV41` | `FLASHMLA_MEGA_ATTN_DSV41` | Runtime default |
| Sparse indexer | MXFP4 KV, sparse logits enabled | MXFP4 KV, sparse logits enabled | MXFP4 KV, sparse logits enabled | Runtime default |
| EPLB communicator | `torch_gloo` | `torch_gloo` | `torch_gloo` | Runtime default |
| Max context tokens | 1,048,576 | Runtime default | 1,048,576 | Model default (1,048,576) |
| Max sequences | 1,024 | Runtime default | 1,024 | 1,024 |
| Max batched tokens | 16,384 | 32,768 P / runtime default D | 16,384 | 8,192 |
| GPU memory utilization | 0.92 | Runtime default | 0.92 | 0.92 |
| KV block size | 128 | Runtime default | 128 | Runtime default |
| Max CUDA graph capture size | 512 | 512 P / 1,024 D | 512 P / 1,024 D | 512 |
| Long-prefill threshold | Unset | 4,096 P | 4,096 P | Unset |
| Prefix-cache retention interval | 1,024 | 1,024 | 1,024 | 1,024 |
| Conditional disaggregation | N/A | `isl_bounding` | Enabled, default policy | N/A |
| KV transfer | N/A | NIXL/UCX over InfiniBand | NIXL/UCX, one NVLink clique | N/A |

Aggregated routing sets decode-active-request weight to 50. B200 conditional
routing sets effective-input threshold 2,048, input-ratio threshold 0.70, and
decode-busy threshold 0.50. Runtime defaults are intentionally left unset;
see the linked Kustomize sources for flags and container images.

## Prerequisites

- A Kubernetes cluster with a compatible Dynamo operator and Kustomize v5.8.1.
- Eight B200/GB200 GPUs or 16 H200 GPUs, with four GPUs per worker.
  GB200 requires ARM64 nodes; B200 and H200 require AMD64 nodes.
- A populated ReadWriteMany model-cache PVC. Workers reference
  `shared-model-cache` at `/shared-model-cache`.
- At least 512 GiB of host memory per worker and capacity for its 200 GiB
  shared-memory volume.
- Access to the container registry referenced by the selected recipe.
- Disaggregated targets: RDMA devices and UCX configured for the cluster.
  Device resource names and interface bindings belong to the cluster configuration.
- GB200 disaggregated: NVIDIA DRA and ComputeDomain support, with both workers
  placed in the same NVLink domain.

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
   networking. For B200 disaggregation, discover the
   InfiniBand resource key and optional `UCX_NET_DEVICES` list. For GKE GB200
   disaggregation, supply the RDMA network annotations/resources and the
   provider's GID index.
4. Build and apply the configuration as shown below.

To view a target configuration from the repository root:

```bash
kustomize build recipes/deepseek-v4.1-flash/vllm/agg-gb200-agentic
```

To deploy your cluster configuration:

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

Replace `gb200-agg` in the service name with `b200-agg`, `gb200-disagg`,
`b200-disagg`, or `h200-agg` for the other targets. In a separate terminal:

```bash
curl -sS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ai/DeepSeek-V4.1-Flash","messages":[{"role":"user","content":"Reply with exactly: READY"}],"temperature":0,"max_tokens":512}'
```

Check that the response contains an answer.

## Performance

Results for the agentic workload (64K input tokens, 400 output tokens), using
eight B200/GB200 GPUs or 16 H200 GPUs. Output throughput includes reasoning tokens.

| Target | Concurrency | Output tok/s/GPU | Output tok/s/user p50 |
| --- | ---: | ---: | ---: |
| B200 aggregated | 168 | 990.57 | 54.69 |
| B200 disaggregated | 184 | 1,087.71 | 82.12 |
| GB200 aggregated | 168 | 953.08 | 51.83 |
| GB200 disaggregated | 168 | 1,154.87 | 80.85 |
| H200 aggregated | 80 | 209.18 | 51.32 |

The measured disaggregated configurations have higher output throughput and
lower inter-token latency (ITL), with longer time-to-first-token (TTFT) tails. GB200 records 21% higher
output tok/s/GPU and 56% higher p50 output tok/s/user; TTFT p90 rises from
3.52 s to 57.12 s.

See [benchmark instructions and TTFT/ITL distributions](perf/README.md).

## SGLang Targets

SGLang recipes for GB200:

| Target | Deployment | GPUs | Layout |
| --- | --- | ---: | --- |
| Aggregated | [agg-gb200](sglang/agg-gb200/deploy.yaml) | 8 | 2 × TP4, EP4 |
| Disaggregated | [disagg-gb200](sglang/disagg-gb200/deploy-generic.yaml) | 8 | 1P1D, TP4 and EP4 per role |

These targets use the SGLang `dev.1` runtime. Aggregated serving enables DSpark; disaggregated
serving uses Mooncake without speculative decoding. The
[GKE variant](sglang/disagg-gb200/deploy-gke-rdma.yaml) adds provider RDMA settings.

## Editing

Edit the Kustomize sources and run `kustomize build` to check the result.
Keep generated manifests and cluster-specific settings outside the recipe.
