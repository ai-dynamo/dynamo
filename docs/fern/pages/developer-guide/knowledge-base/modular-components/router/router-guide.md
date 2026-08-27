---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Router Guide
subtitle: Choose where routing runs and how worker-set configuration takes effect.
---

Use this guide to choose a router deployment topology. It does not replace the local or Kubernetes setup guides, which own the commands and manifests for each request path.

## Choose a Deployment Topology

| Topology | Where selection runs | Use it when | Setup guide |
| --- | --- | --- | --- |
| Frontend-embedded | Dynamo Frontend | Clients can call one Dynamo service directly. | [KV-Aware Routing for the CLI](../../../../cli/kv-aware-routing/overview.mdx) or [Dynamo Frontend Routing](../../../../kubernetes/kv-aware-routing/dynamo-frontend.md) |
| Gateway API | Dynamo Endpoint Picker Plugin (EPP) | A Kubernetes Gateway owns ingress policy and endpoint selection. | [KV-Aware Routing on Kubernetes](../../../../kubernetes/kv-aware-routing/overview.md) |
| Standalone | Dynamo router service | A custom request path or multi-tier architecture needs selection outside the Frontend. | [Standalone Router](../../../../cli/kv-aware-routing/standalone-router.md) |

The Frontend and EPP use the same worker selection behavior. Do not configure the Frontend to make another selection after an EPP has selected a worker.

## Choose a Routing Mode

Use `kv` mode when cache reuse should affect placement. Workers publish KV lifecycle events so the router can measure per-worker prefix overlap. Without worker events, `kv` mode uses load-only scoring; set `--no-router-kv-events` to predict cache state from routing decisions instead.

Use `round-robin`, `random`, `power-of-two`, `least-loaded`, or `device-aware-weighted` when cache state is not part of the decision. Use `direct` when an upstream component has already selected a worker. See [Routing Concepts](routing-concepts.md#basic-routing) for policy behavior and [Configuration and Tuning](configuration-and-tuning.md#kv-event-transport) for event transport.

> [!IMPORTANT]
> With `DYN_LORA_ENABLED`, use KV, random, or round-robin routing. Direct, power-of-two, least-loaded, and device-aware-weighted modes are not LoRA-aware and fail startup. Session affinity with LoRA is supported only in KV mode; random and round-robin plus affinity are rejected.

## Configure Worker Sets Consistently

The Frontend's `--router-mode` sets the default routing configuration. A worker set can advertise its own router configuration, which replaces the Frontend configuration for requests routed to that set.

A worker set contains replicas with the same namespace, component, endpoint, model, and worker type. Every replica in a set must publish the same router configuration. If replicas disagree, Dynamo treats the set as a conflict and does not admit any of its instances.

Worker-set configuration replaces rather than merges with the Frontend configuration. When a worker advertises `--router-mode kv`, restate every non-default router setting that it needs; omitting a setting selects that setting's default instead of inheriting the Frontend value. See [Configuration Scope and Precedence](configuration-and-tuning.md#configuration-scope-and-precedence) for examples and environment-variable implications.

## Choose Distributed Features Deliberately

Use [Disaggregated Serving](disaggregated-serving.md) for separate prefill and decode pools. Add [Topology-Aware KV Transfer](topology-aware-kv-transfer.md) when decode selection must honor transfer locality, and use the [Offloading Support Matrix](offloading-support-matrix.md) when lower cache tiers should influence routing.

For replicas, event recovery, and remote indexers, see [Router Operations](router-operations.md). For APIs that separate index, selection, or load tracking from the router process, see [Standalone Services](standalone-indexer.md).
