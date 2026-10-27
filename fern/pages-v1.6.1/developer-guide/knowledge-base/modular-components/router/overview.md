---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Router
subtitle: Choose a request path, understand worker selection, and operate KV-aware routing.
---

The Dynamo KV router selects an eligible worker using reusable KV cache state and projected active load. Use it when prefix reuse should influence placement; use a non-KV routing mode when your deployment needs only load balancing or an externally selected worker.

## Start With Your Request Path

Choose the page that matches where requests enter Dynamo:

| Request path | Start here |
| --- | --- |
| Local Dynamo Frontend | [KV-Aware Routing for the CLI](../../../../cli/kv-aware-routing/overview.mdx) |
| Dynamo Frontend on Kubernetes | [Dynamo Frontend Routing](../../../../kubernetes/kv-aware-routing/dynamo-frontend.md) |
| Kubernetes Gateway API | [KV-Aware Routing on Kubernetes](../../../../kubernetes/kv-aware-routing/overview.md) |
| Custom or multi-tier request path | [Standalone Router](../../../../cli/kv-aware-routing/standalone-router.md) |

For a topology comparison and the configuration boundary between the Frontend and workers, see the [Router Guide](router-guide.md).

## Follow a Request Through the Router

1. The request host tokenizes and normalizes the request.
2. The router filters workers that cannot serve it.
3. The router scores eligible workers using cache overlap and projected load.
4. Aggregated deployments run on the selected worker, while disaggregated deployments select separate prefill and decode workers. Workers publish KV lifecycle events when available; set `--no-router-kv-events` to predict cache state when the router cannot consume them.

Read [KV-Aware Routing](../../concepts/system-architecture/kv-aware-routing.md) for the architecture-level flow. Read [Routing Concepts](routing-concepts.md) for the cost model, [Router Filtering](worker-filtering.md) for eligibility, and [Deficit Round Robin Queue Scheduling](deficit-round-robin.md) for policy-class arbitration.

## Continue by Task

| Need | Use |
| --- | --- |
| Tune cache and load tradeoffs | [Configuration and Tuning](configuration-and-tuning.md) |
| Run separate prefill and decode pools | [Disaggregated Serving](disaggregated-serving.md) |
| Recover router state or run replicas | [Router Operations](router-operations.md) |
| Run router primitives outside the Frontend | [Standalone Services](standalone-indexer.md) |
| Change or test router behavior | [Develop the Router](router-design.md) |

The [Frontend Configuration Reference](../../../../reference/components/frontend-configuration.mdx#router) is the canonical source for embedded-router flags, environment variables, defaults, and boolean forms.
