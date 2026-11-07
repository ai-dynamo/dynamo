<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SGLang Intel XPU Deployment Examples

This directory holds SGLang `DynamoGraphDeployment` manifests for Intel GPUs, with Kubernetes
Dynamic Resource Allocation (DRA) and device-plugin variants.

For the XPU image build, cluster requirements, SGLang XPU settings, and deployment steps, see
[Deploy SGLang on Intel XPU](../../../../../docs/fern/pages/kubernetes/model-deployment/deploy-on-intel-gpus.mdx#deploy-sglang-on-intel-xpu).

## Contents

| Path | Docs example |
|---|---|
| `agg_xpu_dra.yaml` | [Aggregated Serving](../../../../../docs/fern/pages/recipes/examples/aggregated.mdx) |
| `agg_router_xpu_dra.yaml` | [Aggregated KV-Aware Routing](../../../../../docs/fern/pages/recipes/examples/aggregated-kv-routing.mdx) |
| `disagg_xpu_dra.yaml`, `disagg_xpu.yaml` | [Disaggregated Serving](../../../../../docs/fern/pages/recipes/examples/disaggregated.mdx) |
