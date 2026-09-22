---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Graceful Shutdown
subtitle: Let workers finish in-flight requests and release resources cleanly when a pod is terminated.
---

When Kubernetes terminates a pod (rollout, scale-down, node drain), Dynamo workers stop accepting new requests, keep serving in-flight ones through a grace period, then release engine and connection resources before exiting. This is **on by default** — every component handles `SIGTERM`/`SIGINT` and drains automatically. The steps below tune *how long* it waits and make sure interrupted requests are recovered.

Once shutdown proceeds past the grace period and any backend-specific draining, workers initiate cancellation of unfinished requests. The Frontend can migrate these requests to a healthy worker when migration is enabled and policy permits it; otherwise the client receives an error. Exhausted retries or an exceeded sequence-length cap can prevent recovery.

The knobs set a total deadline, stage caps, and a policy, plus enabling migration. The default flow: endpoints unregister from discovery immediately, workers serve for a short grace period, then endpoints drain before resources are cleaned up, all inside the total deadline.

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

Set the HTTP timeout on the Frontend and the shutdown budgets on worker components:

| Variable | Default | Purpose |
|----------|---------|---------|
| `DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS` | `5` | How long the Frontend waits for admitted HTTP and WebSocket inference requests to finish before it cancels runtime state. |
| `DYN_WORKER_SHUTDOWN_TOTAL_TIMEOUT_SECS` | `30` | Total SIGTERM-to-exit budget, including router grace, draining, engine cleanup, and runtime teardown. Same default in debug and release. |
| `DYN_WORKER_SHUTDOWN_ROUTER_GRACE_SECS` | `5` | Time to serve already-routed requests after discovery unregister, before closing local admission. |
| `DYN_WORKER_SHUTDOWN_KV_TRANSFER_TIMEOUT_SECS` | `30` | Prefill KV-transfer drain cap, bounded by the total after reserving cleanup. |
| `DYN_RUNTIME_GRACEFUL_SHUTDOWN_TIMEOUT_SECS` | `900` | Upper bound on waiting for in-flight requests to finish, for a process that is *not* a backend worker (the frontend, an embedded runtime). A backend worker's drain is bounded by `DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT` instead — the same deadline its stages spend against, so the drain and the wait for it cannot disagree. |
| `DYN_WORKER_SHUTDOWN_INFLIGHT_TIMEOUT_SECS` | uncapped | Cap on a worker waiting for admitted requests to finish, *within* the total budget. Unset means the stage is bounded only by what is left of `DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT`; set it to make the barrier give up sooner and leave more of the budget for KV drain and cleanup. Raising it past the total has no effect — raise the total instead. |
| `DYN_WORKER_SHUTDOWN_KV_TRANSFER_FALLBACK` | engine's choice | `wait` or `skip`. What a prefill worker does when its engine cannot report KV-transfer state. Overrides the engine's own declaration. Leave unset unless you know the engine holds no KV a decode peer could still be reading — `skip` can free GPU memory mid-transfer. |
| `DYN_WORKER_SHUTDOWN_CLEANUP_TIMEOUT_SECS` | `5` | Shared allowance for engine cleanup and runtime teardown, reserved inside the total. Applies to Rust sidecars and Python in-process workers. |

The legacy names `DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT`, `DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS`, and `DYN_PREFILL_DRAIN_TIMEOUT_S` remain aliases for total, router grace, and KV drain respectively. A non-empty canonical value takes precedence; an invalid canonical value uses the default, not the alias. Total seconds must be a positive integer no larger than 315360000. Stage caps accept finite seconds, including fractions; negative drain or grace caps clamp to zero, and non-positive cleanup caps use the default. Invalid or oversized values log a warning and use the default. SDK `ShutdownConfig` fields take precedence over environment values.

The total now includes router grace and cleanup: neither is added afterward. Increase an existing total explicitly if the previous additive behavior was required. The default debug total changes from 5 to 30 seconds to leave room for grace, draining, and cleanup.

> [!IMPORTANT]
> Workers spend one total measured from SIGTERM. Before cleanup, each stage gets at most `min(stage cap, remaining total minus cleanup reserve)`. Engine cleanup and transport teardown then share `min(cleanup cap, remaining total)`. The watchdog does not extend the total. Set `terminationGracePeriodSeconds` to at least **total + 5 seconds**, plus any time needed by Kubernetes pre-stop hooks. Defaults give a 30-second worker bound, inside the operator's 60-second pod grace period.
>
> The frontend's `DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS` is separate; it bounds a different process.

The operator validates explicit literal total-budget settings after pod overrides are merged. It rejects an insufficient pod grace period or an unresolved `valueFrom` total. It does not change existing templates to inject a budget. For budgets set through `envFrom`, image defaults, or shell exports, declare a literal total in the pod template as well; otherwise the operator cannot validate the effective value. Programmatic SDK overrides must also be reflected in the pod configuration.

</Step>

<Step title="Enable migration so drained requests retry">

Draining lets *current* requests finish during the grace period set by `DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS`, but a request interrupted after the shutdown grace period expires or by an unexpected worker loss still needs somewhere to go. Enable [request migration](request-migration.md) on the Frontend for rolling upgrades and scale-downs as well, so eligible interrupted requests are retried on healthy workers. Set `DYN_MIGRATION_LIMIT` in the Frontend `env:` (or `--migration-limit` in its `args:`):

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

```text
INFO  Received shutdown signal, shutting down DistributedRuntime
INFO  DistributedRuntime shutdown complete
DEBUG Cleaning up worker
```

During Frontend shutdown, `/health` returns 503 so readiness routing stops, while `/live` remains
200 so Kubernetes does not restart the process during the drain. New OpenAI-compatible requests return
503, but admitted response bodies and accepted `/v1/realtime` WebSocket sessions continue until they
finish or `DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS` expires.

During worker shutdown, endpoints unregister and stop receiving new work while admitted requests
have time to complete. If a pod receives `SIGKILL` before draining finishes, increase
`terminationGracePeriodSeconds` (step 1) or lower the relevant internal timeout (step 2).

</Step>

</Steps>

## Custom workers

Stage logs report start, outcome, elapsed time, and remaining total. Inspect `dynamo_component_shutdown_stage_seconds{stage,reason}`, `dynamo_component_shutdown_remaining_seconds`, `dynamo_component_shutdown_inflight_requests`, and `dynamo_component_shutdown_kv_quiescent` during shutdown. Remaining time and inflight counts are snapshots at stage transitions. KV quiescence is `1` for confirmed idle, `0` for confirmed busy, and `-1` for unsupported or unknown status. An `unsupported` outcome denotes the declared fallback, not proof that transfers completed. Collect logs before pod removal; the metrics endpoint disappears with the process.

If you author your own worker with the Dynamo SDK, the `graceful_shutdown` parameter on `serve_endpoint()` controls whether that endpoint waits for in-flight requests (`True`) or returns immediately (`False`). Backend workers default to `True`. For the parameter, the shutdown sequence, and per-backend cleanup patterns, see [Graceful Shutdown Architecture](../../developer-guide/knowledge-base/concepts/fault-tolerance/graceful-shutdown-architecture.md) and the [Writing Python Workers](../../developer-guide/advanced-customizations/writing-custom-backends/writing-python-workers.md) guide.

## Related Documentation

- [Graceful Shutdown Architecture](../../developer-guide/knowledge-base/concepts/fault-tolerance/graceful-shutdown-architecture.md) - Signal handling, drain sequence, and resource cleanup internals
- [Request Migration](request-migration.md) - How interrupted requests migrate to healthy workers
- [Request Cancellation Architecture](../../developer-guide/knowledge-base/concepts/fault-tolerance/request-cancellation-architecture.md) - Canceling in-flight requests
- [Health Check Reference](../../reference/observability/health-checks.mdx) - Liveness and readiness endpoints
