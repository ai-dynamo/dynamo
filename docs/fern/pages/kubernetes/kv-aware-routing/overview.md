---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: KV-Aware Routing on Kubernetes
sidebar-title: Overview
subtitle: Choose where Dynamo makes cache-aware worker-selection decisions in a Kubernetes deployment.
---

Dynamo supports two KV-aware request-routing topologies on Kubernetes. They use the same routing
concepts but place worker selection in different components. Choose based on how your platform accepts
and governs traffic, not because one topology uses a different KV-routing algorithm.

## Recommendation

Use **Dynamo Frontend routing** unless your platform already needs Kubernetes Gateway API. It has
fewer components, requires no Gateway API installation, and works with a standard
`DynamoGraphDeployment`.

Use **Gateway API routing** when the Gateway must own the external request path—for example, when a
platform team standardizes ingress, authentication, rate limiting, traffic policy, or gateway-level
telemetry through Gateway API.

| Choose | When it fits | Additional infrastructure | Where routing is configured |
|---|---|---|---|
| Using the Dynamo Frontend | Clients can call one Dynamo Frontend Service directly and you want the simplest deployment. | None beyond the normal Dynamo platform. | Frontend with `--router-mode kv`. |
| Using GAIE with Dynamo | Traffic must enter through a Kubernetes Gateway or share platform-managed gateway policy. | Gateway API, GAIE, a compatible Gateway implementation, and a Dynamo EPP. | EPP; worker Frontend sidecars use `--router-mode direct`. |
| Using GAIE with vanilla vLLM | You already run upstream vLLM pods and want to add Dynamo's advanced KV-aware routing without migrating the fleet to Dynamo workers. | Gateway API, GAIE, a compatible Gateway implementation, and standalone Dynamo EPP support. | EPP with direct vLLM worker discovery and vLLM token rendering. |

The Frontend and GAIE topologies are alternatives for a request path. Do not configure the Frontend
to select a worker after the EPP has already selected one.

## Dynamo Frontend Routing

The Dynamo Frontend receives the client request, tokenizes it, scores eligible workers, and forwards
the request to the selected worker. Router flags and environment variables belong on the Frontend
component.

Choose this topology when:

- Each deployment can expose its own Frontend Service.
- You do not need Gateway API policy in the request path.
- You want the fewest routing components to install and operate.
- Dynamo should own both HTTP request handling and worker selection.

This topology works out of the box with the normal Dynamo platform; no Gateway API components are
required. Use [the Dynamo Frontend](dynamo-frontend.md) to enable KV-aware worker
selection in a DGD.

## GAIE Routing

Gateway API routing separates traffic entry from worker selection. A Kubernetes `Gateway` receives
the request and asks the Dynamo Endpoint Picker Plugin (EPP) to select an endpoint before forwarding
the request to a worker pod.

Choose this topology when:

- Your organization already uses Gateway API as the standard ingress layer.
- Authentication, authorization, rate limiting, traffic policy, or telemetry belongs at the Gateway.
- Multiple model routes should share a platform-managed address and listener.
- A platform team manages the Gateway while an application team manages the DGD and `HTTPRoute`.

Gateway API routing requires a separate installation. See
[Install Gateway API Inference Extension](../installation/gateway-api-routing.mdx), then
[GAIE with Dynamo](gateway-api.mdx) or [GAIE with vanilla vLLM](vanilla-vllm-onramp.mdx).

## What Is the Dynamo EPP?

The Dynamo **Endpoint Picker Plugin (EPP)** is a service that implements the GAIE endpoint-selection
protocol. The Gateway calls it before forwarding an inference request. The EPP tokenizes the request,
reads Dynamo worker and KV cache state, scores eligible endpoints, and returns the selected worker to
the Gateway.

The EPP is not a public inference endpoint and does not run the model. The Dynamo operator deploys it
when a DGD contains a component with `type: epp`. In this topology, worker Frontend sidecars run with
`--router-mode direct` because the EPP has already made the routing decision.

For the operator-managed setup and request flow, see
[Using GAIE with Dynamo](gateway-api.mdx).

## Shared Routing Behavior

Both topologies can use worker-published KV cache events to account for prefix overlap and active
load. Both can also operate with predicted state when precise KV events are unavailable. The
selection algorithm and tuning concepts are shared; only the component hosting them changes.

For the cost model, see [Routing Concepts](../../developer-guide/knowledge-base/modular-components/router/routing-concepts.md). For shared tuning
guidance, see [Configuration and Tuning](../../developer-guide/knowledge-base/modular-components/router/configuration-and-tuning.md).
