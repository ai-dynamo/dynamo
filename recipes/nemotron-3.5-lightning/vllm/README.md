<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Nemotron 3.5 Lightning vLLM Recipes

Dynamo vLLM recipes for Nemotron 3.5 Lightning. H100 and H200 use
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`, while B200 and GB200 use
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`. Aggregate and
disaggregated configurations are available as explained below.

## Configurations

| Hardware | Aggregate recipes | Disaggregated recipes | P/D transport |
|---|---|---|---|
| H100 | `agg-h100-{mtp,dflash,dspark}` | `disagg-h100-{dflash,dspark}` | AWS EFA / Libfabric |
| H200 | `agg-h200-{mtp,dflash,dspark}` | `disagg-h200-{dflash,dspark}` | UCX over IB/RDMA |
| B200 | `agg-b200-{mtp,dspark}-bf16` | `disagg-b200-dspark-bf16` | UCX over IB/RDMA |
| GB200 | `agg-gb200-{mtp,dspark}-bf16` | `disagg-gb200-{dflash,dspark}-bf16` | UCX/NIXL, cluster-specific fabric resources |

B200 and GB200 directories use BF16 weights and include a `-bf16` suffix, for
example `agg-b200-dspark-bf16` and `disagg-gb200-dspark-bf16`.

Each recipe identifier corresponds to `identifier/deploy.yaml` in this directory.

### KV-aware aggregate DSpark variants

For users who want the highest-performing aggregate DSpark shape with KV-aware routing ready to deploy, the repo also includes event-backed KV-router variants. These keep the vLLM aggregate DSpark worker configuration, run four aggregate replicas, and publish KV-cache events to the frontend KV router. Start benchmark concurrency at approximately 4x the matching non-KV aggregate recommendation.

| Recipe | Worker replicas | Base concurrency | Suggested KV-router concurrency |
|---|---:|---:|---:|
| `agg-h100-dspark-kv-router` | 4 | 20 | 80 |
| `agg-h200-dspark-kv-router` | 4 | 20 | 80 |
| `agg-b200-dspark-kv-router-bf16` | 4 | 50 | 192 |

These recipes are provided for workloads that can benefit from distributing
reusable KV state across several workers, for example when a single worker sees
enough distinct long-prefix families to create KV-cache pressure. For the current
validation trace, KV-aware routing was not promoted over the single-worker
aggregate DSpark recipe: even H100, the lowest-VRAM SKU tested here, peaked at
only 38.1% GPU KV-cache usage in the one-worker full-trace run. The single
worker therefore had enough cache headroom to keep useful prefixes resident
without meaningful eviction pressure, while adding replicas split work across
more GPUs.

## Prerequisites

1. Dynamo Platform and the `DynamoGraphDeployment` CRD installed on the target
   Kubernetes cluster.
2. The shared model cache created from `../model-cache/model-cache.yaml` and
   populated with the matching NVFP4 or BF16 download Job.
3. A Hugging Face secret named `hf-token-secret` and an image-pull secret when
   required by the cluster.
4. For disaggregated recipes that use AWS EFA, EFA userspace must be prepared by
   the user before deployment. These manifests request `vpc.amazonaws.com/efa`
   and select the NIXL LIBFABRIC backend, but they do not install AWS EFA
   packages during pod startup because non-AWS clusters should not run
   AWS-specific setup. See [EFA setup](efa.md) for validation and
   deployment-time setup options.

## Synthetic acceptance configuration

Each manifest defines `speculative-config` for standard serving and
`speculative-config-synthetic` for benchmark reproduction. Select the
synthetic configuration by changing the `SPECULATIVE_CONFIG` ConfigMap key.

The synthetic acceptance lengths used in these configurations are based on
internal measurements.

## Notes

- GB200 recipes use ARM-node scheduling.
- B200 recipes disable the FlashInfer FP8 ScaledMM linear kernel.
- Disaggregated recipes require matching speculative-decoding settings on the
  prefill and decode workers for compatible NIXL cache metadata.
- AWS EFA disaggregated recipes require the EFA userspace stack before
  deployment.
- Replace `your-image-pull-secret` and any cluster-specific networking resource
  names with values available in the target cluster.
