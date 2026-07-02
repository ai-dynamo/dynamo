---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Feature Guides
subtitle: Start with Dynamo's core serving optimizations, then branch into operations and model capabilities.
---

Use these guides after you have Dynamo running and want to improve serving behavior, operate a deployment, or adapt Dynamo to a new workload.

## Recommended path

Most deployments start with the core performance loop:

| Step | Guide | Use when |
|---|---|---|
| 1 | [KV Cache Aware Routing](../components/router/router-guide.md) | Route requests to workers that already hold useful KV cache. |
| 2 | [Disaggregated Serving](disaggregated-serving/README.md) | Scale prefill and decode workers independently. |
| 3 | [KV Cache Offloading](../components/kvbm/kvbm-guide.md) | Extend usable cache capacity beyond GPU memory. |
| 4 | [Benchmarking](../benchmarks/benchmarking.md) | Compare configurations before you move to production. |

## Where to go next

| Goal | Start with |
|---|---|
| Make serving more resilient | [Fault Tolerance](../fault-tolerance/README.md) |
| Monitor local deployments | [Observability (Local)](../observability/README.md) |
| Reproduce traffic without a full engine | [Mocker Engine Simulation](../mocker/mocker.md) |
| Add structured model outputs | [Tool Calling](../tool-calling/README.md) and [Reasoning](../reasoning/README.md) |
| Build agent workloads | [Agents](../agents/README.md) |
| Serve specialized workloads | [LoRA Adapters](lora/README.md), [Multimodal](multimodal/README.md), and [Diffusion](diffusion/README.md) |

For cluster deployments, pair these guides with the [Kubernetes Deployment](../kubernetes/README.md) docs. The same features can be explored locally, then expressed through Dynamo's Kubernetes-native CRDs and operator when you move to a shared GPU cluster.
