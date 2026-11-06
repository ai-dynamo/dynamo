<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# vLLM Kubernetes Deployment Examples

This directory holds `DynamoGraphDeployment` (DGD) manifests for vLLM. They use the
`nvidia.com/v1beta1` API.

For prerequisites, deployment steps, and configuration, see the
[Examples catalog](../../../../docs/fern/pages/recipes/examples/overview.mdx), which renders each manifest with a deployment customizer, and
[Deploy with DGD](../../../../docs/fern/pages/kubernetes/model-deployment/deploy-with-dgd.md). For
vLLM worker flags, see the [vLLM backend guide](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/backends/vllm/overview.md).

## Contents

| Path | Docs example |
|---|---|
| `agg.yaml` | [Aggregated Serving](../../../../docs/fern/pages/recipes/examples/aggregated.mdx) |
| `agg_router.yaml` | [Aggregated KV-Aware Routing](../../../../docs/fern/pages/recipes/examples/aggregated-kv-routing.mdx) |
| `agg_router_kv_approx.yaml` | [Approximate Cache Tracking](../../../../docs/fern/pages/recipes/examples/approximate-kv-routing.mdx) |
| `agg_asr.yaml`, `agg_asr_realtime.yaml` | [Audio and Realtime Serving](../../../../docs/fern/pages/recipes/examples/audio.mdx) |
| `agg_failover.yaml`, `gms-failover.yaml` | [GPU Failover](../../../../docs/fern/pages/recipes/examples/failover.mdx) |
| `agg_gms.yaml` | [GPU Memory Service](../../../../docs/fern/pages/recipes/examples/gpu-memory-service.mdx) |
| `agg_kvbm.yaml`, `disagg_kvbm*.yaml` | [KV Block Manager](../../../../docs/fern/pages/recipes/examples/kvbm.mdx) |
| `agg_omni_audio.yaml` | [Audio Generation](../../../../docs/fern/pages/recipes/examples/audio-generation.mdx) |
| `agg_omni_image.yaml` | [Image Generation](../../../../docs/fern/pages/recipes/examples/image-generation.mdx) |
| `agg_tracing.yaml`, `disagg_tracing.yaml` | [OpenTelemetry Tracing](../../../../docs/fern/pages/recipes/examples/tracing.mdx) |
| `disagg.yaml` | [Disaggregated Serving](../../../../docs/fern/pages/recipes/examples/disaggregated.mdx) |
| `disagg_router.yaml` | [Disaggregated KV-Aware Routing](../../../../docs/fern/pages/recipes/examples/disaggregated-kv-routing.mdx) |
| `disagg_planner.yaml` | [Planner Autoscaling](../../../../docs/fern/pages/recipes/examples/planner.mdx) |
| `disagg-multinode.yaml` | [Multinode Disaggregation](../../../../docs/fern/pages/recipes/examples/multinode.mdx) |
| `gaie/` | [Gateway API Routing](../../../../docs/fern/pages/recipes/examples/gateway-api.mdx) |
| `kvcr/` | [KV Cache Runner](../../../../docs/fern/pages/recipes/examples/kvcr.mdx) |
| `lora/` | [LoRA Serving](../../../../docs/fern/pages/recipes/examples/lora.mdx) |
| `v1beta1/` | [LMCache](../../../../docs/fern/pages/recipes/examples/lmcache.mdx) |
| `xpu/` | [Deploy on Intel GPUs](../../../../docs/fern/pages/kubernetes/model-deployment/deploy-on-intel-gpus.mdx) |
