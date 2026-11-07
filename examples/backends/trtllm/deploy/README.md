<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# TensorRT-LLM Kubernetes Deployment Examples

This directory holds `DynamoGraphDeployment` (DGD) manifests for TensorRT-LLM. They use the
`nvidia.com/v1beta1` API.

For prerequisites, deployment steps, and configuration, see the
[Examples catalog](../../../../docs/fern/pages/recipes/examples/overview.mdx), which renders each manifest with a deployment customizer, and
[Deploy with DGD](../../../../docs/fern/pages/kubernetes/model-deployment/deploy-with-dgd.md). For
TensorRT-LLM worker flags, custom containers, and KV cache transfer, see the
[TensorRT-LLM backend guide](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/backends/tensorrt-llm/overview.md).

## Contents

| Path | Docs example |
|---|---|
| `agg.yaml`, `agg-with-config.yaml` | [Aggregated Serving](../../../../docs/fern/pages/recipes/examples/aggregated.mdx) |
| `agg_router.yaml` | [Aggregated KV-Aware Routing](../../../../docs/fern/pages/recipes/examples/aggregated-kv-routing.mdx) |
| `agg_image_diffusion.yaml` | [Image Generation](../../../../docs/fern/pages/recipes/examples/image-generation.mdx) |
| `disagg.yaml` | [Disaggregated Serving](../../../../docs/fern/pages/recipes/examples/disaggregated.mdx) |
| `disagg_router.yaml` | [Disaggregated KV-Aware Routing](../../../../docs/fern/pages/recipes/examples/disaggregated-kv-routing.mdx) |
| `disagg_planner.yaml` | [Planner Autoscaling](../../../../docs/fern/pages/recipes/examples/planner.mdx) |
| `disagg-multinode.yaml` | [Multinode Disaggregation](../../../../docs/fern/pages/recipes/examples/multinode.mdx) |
| `snapshot-restore.yaml`, `snapshot/` | [Snapshot Restore](../../../../docs/fern/pages/recipes/examples/snapshot-restore.mdx) |
