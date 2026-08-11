---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Fault Tolerance Overview
sidebar-title: Overview
subtitle: Choose request-level and worker-level recovery behaviors for production Dynamo deployments.
---

Dynamo fault tolerance has two layers:

- **Request fault tolerance** protects the client-visible request path. Use these guides when you need to recover in-flight requests, reject new work under overload, or stop wasted work after client disconnects.
- **Worker fault tolerance** protects serving capacity as workers drain, fail, or recover. Use these guides when you need Kubernetes pods to shut down cleanly, recover engines locally, or understand how Dynamo discovers and routes around worker loss.

Most production deployments need both. Request fault tolerance keeps individual generations from failing unnecessarily, while worker fault tolerance keeps the worker pool stable as Kubernetes reschedules pods or hardware faults occur.

## Request Fault Tolerance

These behaviors operate at the request boundary: an incoming request, an in-flight generation, or a client connection.

- **[Request Migration](request-migration.md)** — Recovers an in-flight generation when a worker fails mid-request by moving the request to another healthy worker. **Off by default** — enable it when you want best-effort continuity for long-running generations.
- **[Request Rejection](request-rejection.md)** — Rejects new requests with HTTP 529 when every worker is too busy, so clients can retry instead of adding queueing delay for everyone. **Off by default** — enable it when you want explicit overload behavior.
- **[Request Cancellation](../../developer-guide/knowledge-base/concepts/fault-tolerance/request-cancellation-architecture.md)** — Stops frontend and runtime work when the client disconnects. This is a built-in runtime behavior and does not require workload configuration.

## Worker Fault Tolerance

These behaviors operate at the worker and engine lifecycle boundary: planned shutdown, pod failure, engine failure, and service discovery.

- **[Graceful Shutdown](graceful-shutdown.md)** — Lets a worker finish the requests it is already handling before Kubernetes terminates the pod. **On by default** — tune the grace period to match your rollout and scale-down behavior.
- **[Shadow Engine Failover](../../developer-guide/knowledge-base/kubernetes/kubernetes-operator/shadow-engine-failover.md)** — Runs an active/passive engine pair on the same node so a shadow engine can take over locally after an engine failure. It does not preserve in-flight requests or KV cache state.
- **[Health Check Reference](../../reference/observability/health-checks.mdx)** — Documents the liveness, readiness, and engine-monitoring endpoints used to detect unhealthy workers.
- **[Distributed Runtime](../../developer-guide/knowledge-base/concepts/system-architecture/architecture.md#distributed-runtime)** — Explains the service discovery and lease mechanisms Dynamo uses to detect worker loss and route new traffic to healthy capacity.

## Testing and References

Use these Knowledge Base pages when you want the deeper implementation model or validation details:

- [Fault Tolerance Testing](../../developer-guide/knowledge-base/concepts/fault-tolerance/fault-tolerance-testing.md) — the framework for validating these behaviors (cancellation, migration, etcd HA failover, hardware fault injection).
- [Request Migration Architecture](../../developer-guide/knowledge-base/concepts/fault-tolerance/request-migration-architecture.md) — pipeline position, token-state tracking, and worker-failure scenarios.

## Configuration Reference

Every flag and environment variable for configurable fault tolerance behavior is cataloged in the Reference tab:

- [Frontend Configuration](../../reference/components/frontend-configuration.mdx) — migration limits, independent busy thresholds, overload status, and the threshold API.
- [Runtime Configuration](../../reference/components/runtime-configuration.mdx) — local worker inhibition and worker-side engine and queue limits.
- [Observability Environment Variables](../../reference/observability/environment-variables.mdx) — health-check and system-port variables.
