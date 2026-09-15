---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Graceful Shutdown
subtitle: Let workers finish in-flight requests and release resources cleanly when a pod is terminated.
---

When Kubernetes terminates a pod (rollout, scale-down, node drain), Dynamo workers stop accepting new requests, keep serving in-flight ones through a grace period, then release engine and connection resources before exiting. This is **on by default** — every component handles `SIGTERM`/`SIGINT` and drains automatically. The steps below tune *how long* it waits and make sure interrupted requests are recovered.

The knobs are three timeouts plus enabling migration. The default flow: endpoints unregister from discovery immediately, workers serve for a short grace period, then endpoints drain (bounded by a timeout) before resources are cleaned up.

> **How it works:** the signal handlers, the `graceful_shutdown()` sequence, per-backend `cleanup()` code, and error-initiated shutdown are documented in [Graceful Shutdown Architecture](../../developer-guide/knowledge-base/concepts/fault-tolerance/graceful-shutdown-architecture.md).

<Steps toc={true} tocDepth={2}>

<Step title="Set the pod termination grace period">

Kubernetes gives a terminating pod `terminationGracePeriodSeconds` to exit before it sends `SIGKILL`.
Dynamo operator-created pods default to **60 seconds**. Set a longer value when expected generation
time or high utilization can keep admitted requests active beyond that window:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
spec:
  services:
    worker:
      extraPodSpec:
        terminationGracePeriodSeconds: 180  # allow time for request draining
```

Rough guidance:

| Workload | Suggested `terminationGracePeriodSeconds` |
|----------|-------------------------------------------|
| Short requests (< 10s) | 60s |
| Long generation (> 30s) or high utilization | 120s+ |

</Step>

<Step title="Tune the drain windows">

Three environment variables control Dynamo's internal draining. Set the HTTP timeout on the Frontend and the runtime values on worker components:

| Variable | Default | Purpose |
|----------|---------|---------|
| `DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS` | `5` | How long the Frontend waits for admitted HTTP and WebSocket inference requests to finish before it cancels runtime state. |
| `DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS` | `5` | How long workers keep serving after endpoints unregister from discovery, before endpoints are invalidated. |
| `DYN_RUNTIME_GRACEFUL_SHUTDOWN_TIMEOUT_SECS` | `900` | Upper bound on waiting for in-flight requests to finish. If draining exceeds this, Dynamo logs the remaining endpoint count and tears down anyway. |

The defaults are sound for most deployments. Raise the relevant timeout only for long generations or sustained high utilization. Keep every internal timeout below `terminationGracePeriodSeconds` so Dynamo can finish its own cleanup before Kubernetes force-kills the pod.

</Step>

<Step title="Enable migration so drained requests retry">

Draining lets *current* requests finish, but a request interrupted by an unexpected worker loss still needs somewhere to go. Enable [request migration](request-migration.md) on the Frontend so disconnected streams are retried on healthy workers. Set `DYN_MIGRATION_LIMIT` in the Frontend `env:` (or `--migration-limit` in its `args:`):

```yaml
  - name: Frontend
    type: frontend
    replicas: 1
    podTemplate:
      spec:
        containers:
        - name: main
          image: ${RUNTIME_IMAGE}
          env:
          - name: DYN_MIGRATION_LIMIT
            value: "3"                 # allow up to 3 migration attempts
```

Backend workers always drain with `graceful_shutdown=True`; they don't need any migration configuration themselves. See [Request Migration](request-migration.md) for the full walkthrough.

</Step>

<Step title="Verify graceful shutdown">

Trigger a shutdown (for example `kubectl delete pod <worker-pod>` or a rollout) and watch the worker logs for the shutdown sequence:

```
INFO  Received shutdown signal, shutting down DistributedRuntime
INFO  DistributedRuntime shutdown complete
DEBUG Cleaning up worker
```

During Frontend shutdown, `/health` returns 503 so readiness routing stops, while `/live` remains
200 so Kubernetes does not restart the process during the drain. New OpenAI-compatible requests return
503, but admitted response bodies and accepted `/v1/realtime` WebSocket sessions continue until they
finish or `DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS` expires.

During worker shutdown, endpoints unregister and stop receiving new work while admitted requests
complete. If a pod receives `SIGKILL` before draining finishes, increase
`terminationGracePeriodSeconds` (step 1) or lower the relevant internal timeout (step 2).

</Step>

</Steps>

## Custom workers

If you author your own worker with the Dynamo SDK, the `graceful_shutdown` parameter on `serve_endpoint()` controls whether that endpoint waits for in-flight requests (`True`) or returns immediately (`False`). Backend workers default to `True`. For the parameter, the shutdown sequence, and per-backend cleanup patterns, see [Graceful Shutdown Architecture](../../developer-guide/knowledge-base/concepts/fault-tolerance/graceful-shutdown-architecture.md) and the [Writing Python Workers](../../developer-guide/advanced-customizations/writing-custom-backends/writing-python-workers.md) guide.

## Related Documentation

- [Graceful Shutdown Architecture](../../developer-guide/knowledge-base/concepts/fault-tolerance/graceful-shutdown-architecture.md) - Signal handling, drain sequence, and resource cleanup internals
- [Request Migration](request-migration.md) - How interrupted requests migrate to healthy workers
- [Request Cancellation Architecture](../../developer-guide/knowledge-base/concepts/fault-tolerance/request-cancellation-architecture.md) - Canceling in-flight requests
- [Health Check Reference](../../reference/observability/health-checks.mdx) - Liveness and readiness endpoints
