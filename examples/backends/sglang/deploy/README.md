<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SGLang Kubernetes Deployment Examples

This directory holds `DynamoGraphDeployment` (DGD) manifests for SGLang. They use the
`nvidia.com/v1beta1` API.

For prerequisites, deployment steps, and configuration, see the
[Examples catalog](../../../../docs/fern/pages/recipes/examples/overview.mdx), which renders each manifest with a deployment customizer, and
[Deploy with DGD](../../../../docs/fern/pages/kubernetes/model-deployment/deploy-with-dgd.md). For
SGLang worker flags, see the [SGLang backend guide](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/backends/sglang/overview.md).

## Contents

| Path | Docs example |
|---|---|
| `agg.yaml` | [Aggregated Serving](../../../../docs/fern/pages/recipes/examples/aggregated.mdx) |
| `agg_router.yaml` | [Aggregated KV-Aware Routing](../../../../docs/fern/pages/recipes/examples/aggregated-kv-routing.mdx) |
| `agg_embed.yaml` | [Text Embeddings](../../../../docs/fern/pages/recipes/examples/embeddings.mdx) |
| `agg_gms.yaml` | [GPU Memory Service](../../../../docs/fern/pages/recipes/examples/gpu-memory-service.mdx) |
| `agg_image_diffusion.yaml` | [Image Generation](../../../../docs/fern/pages/recipes/examples/image-generation.mdx) |
| `agg_llm_diffusion.yaml` | [Diffusion Language Models](../../../../docs/fern/pages/recipes/examples/text-diffusion.mdx) |
| `agg_logging.yaml` | [Logging and Metrics](../../../../docs/fern/pages/recipes/examples/logging-metrics.mdx) |
| `disagg.yaml` | [Disaggregated Serving](../../../../docs/fern/pages/recipes/examples/disaggregated.mdx) |
| `disagg_planner.yaml` | [Planner Autoscaling](../../../../docs/fern/pages/recipes/examples/planner.mdx) |
| `disagg-multinode.yaml` | [Multinode Disaggregation](../../../../docs/fern/pages/recipes/examples/multinode.mdx) |
| `gaie/` | [Gateway API Routing](../../../../docs/fern/pages/recipes/examples/gateway-api.mdx); `gaie/switchyard/` adds [Switchyard model routing](../../../../docs/fern/pages/kubernetes/kv-aware-routing/switchyard.md) |
| `xpu/` | [Deploy SGLang on Intel XPU](../../../../docs/fern/pages/kubernetes/model-deployment/deploy-on-intel-gpus.mdx#deploy-sglang-on-intel-xpu) |
