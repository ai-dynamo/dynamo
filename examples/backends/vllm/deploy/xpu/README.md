<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# vLLM Intel XPU Deployment Examples

This directory holds vLLM `DynamoGraphDeployment` manifests for Intel XPU GPUs. Each manifest
includes the `ResourceClaimTemplate` that allocates the GPU through Kubernetes Dynamic Resource
Allocation (DRA).

For prerequisites, the XPU image build, deployment steps, and the differences from the NVIDIA
templates, see [Deploy on Intel GPUs with DRA](../../../../../docs/fern/pages/kubernetes/model-deployment/deploy-on-intel-gpus.mdx).

## Contents

| Path | Docs example |
|---|---|
| `agg_xpu_dra.yaml` | [Aggregated Serving](../../../../../docs/fern/pages/recipes/examples/aggregated.mdx) |
| `agg_router_xpu_dra.yaml` | [Aggregated KV-Aware Routing](../../../../../docs/fern/pages/recipes/examples/aggregated-kv-routing.mdx) |
| `agg_router_kv_approx_xpu_dra.yaml` | [Approximate Cache Tracking](../../../../../docs/fern/pages/recipes/examples/approximate-kv-routing.mdx) |
| `agg_tracing_xpu_dra.yaml`, `disagg_tracing_xpu_dra.yaml` | [OpenTelemetry Tracing](../../../../../docs/fern/pages/recipes/examples/tracing.mdx) |
| `agg_lora_xpu_dra.yaml`, `agg_qwen_lora_xpu_dra.yaml` | [LoRA Serving](../../../../../docs/fern/pages/recipes/examples/lora.mdx) |
| `disagg_xpu_dra.yaml` | [Disaggregated Serving](../../../../../docs/fern/pages/recipes/examples/disaggregated.mdx) |
| `disagg_router_xpu_dra.yaml` | [Disaggregated KV-Aware Routing](../../../../../docs/fern/pages/recipes/examples/disaggregated-kv-routing.mdx) |
| `disagg_planner_xpu_dra.yaml` | [Planner Autoscaling](../../../../../docs/fern/pages/recipes/examples/planner.mdx) |
