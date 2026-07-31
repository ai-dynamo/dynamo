---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Introduction
subtitle: Configure how Dynamo handles worker failures, overload, and shutdown in production.
---

Dynamo keeps LLM inference reliable through two kinds of fault-tolerance behavior:

- **Configurable behaviors** — off or conservative by default, these need you to opt in and tune them for your workload. They are the focus of this section.
- **Built-in behaviors** — automatic runtime mechanisms (failure detection, service discovery, request cancellation) that require no configuration. These are covered in the Knowledge Base, linked at the bottom.

This section walks through the three behaviors you configure yourself.

## Configurable behaviors

Each guide is short and practical: what the behavior does, how to turn it on, and how to check it works.

- **[Request Migration](request-migration.md)** — If a worker breaks in the middle of answering a request, another worker picks it up and finishes the answer. The user never notices. **Off by default** — you turn it on.
- **[Request Rejection](request-rejection.md)** — When every worker is too busy, Dynamo returns HTTP 529 so clients can retry instead of increasing latency for everyone. **Off by default** — you turn it on and set how busy is "too busy."
- **[Graceful Shutdown](graceful-shutdown.md)** — When a worker is asked to stop, it finishes the requests it's already handling before shutting down, instead of dropping them. **On by default** — you adjust how long it waits.

## Built-in behaviors (Knowledge Base)

These operate automatically and are documented as architecture references, not configuration guides:

- [Request Cancellation](../../developer-guide/knowledge-base/concepts/fault-tolerance/request-cancellation-architecture.md) — the frontend and runtime abort in-flight requests when a client disconnects.
- [Fault Tolerance Testing](../../developer-guide/knowledge-base/concepts/fault-tolerance/fault-tolerance-testing.md) — the framework for validating these behaviors (cancellation, migration, etcd HA failover, hardware fault injection).
- [Health Check Reference](../../reference/observability/health-checks.mdx) — liveness/readiness endpoints and engine monitoring that drive failure detection.
- [Shadow Engine Failover](../../developer-guide/knowledge-base/kubernetes/kubernetes-operator/shadow-engine-failover.md) — same-node active/passive engine recovery for Kubernetes (does not preserve in-flight requests or KV cache state).
- [Distributed Runtime](../../developer-guide/knowledge-base/concepts/system-architecture/distributed-runtime.md) — the service discovery and lease mechanism that detects worker loss and reroutes traffic.

## Configuration reference

Every flag and environment variable for the configurable behaviors is cataloged in the Reference tab:

- [Frontend Configuration](../../reference/components/frontend-configuration.mdx) — migration limits, independent busy thresholds, overload status, and the threshold API.
- [Runtime Configuration](../../reference/components/runtime-configuration.mdx) — local worker inhibition and worker-side engine and queue limits.
- [Observability Environment Variables](../../reference/observability/environment-variables.mdx) — health-check and system-port variables.
