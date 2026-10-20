---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Health Check Reference
subtitle: Health endpoints, status codes, paths, and active-check configuration
---

Dynamo exposes two related health interfaces: HTTP status endpoints for external observers and
optional canary requests that actively exercise idle worker endpoints. For a local procedure, see
[Observe a Local Deployment](../../cli/operations/observability.mdx#check-deployment-health).

## Endpoint Summary

| Process | Default port | Path | Behavior |
| --- | --- | --- | --- |
| Frontend | `DYN_HTTP_PORT`, default `8000` | `/live` | Reports whether the HTTP service is running or shutting down |
| Frontend | `DYN_HTTP_PORT`, default `8000` | `/health` | Reports frontend health plus discovered endpoints and instances |
| Frontend | `DYN_HTTP_PORT`, default `8000` | `/v1/models/<model>/ready` | Reports whether the selected model has ready topology, a usable serving engine, and enabled HTTP routes |
| Worker or standalone router | `DYN_SYSTEM_PORT`, disabled by default | `/live` | Reports system status |
| Worker or standalone router | `DYN_SYSTEM_PORT`, disabled by default | `/health` | Reports system status |

Set `DYN_SYSTEM_HEALTH_PATH` or `DYN_SYSTEM_LIVE_PATH` before process startup to replace the worker
or router paths. For the frontend, use `DYN_HTTP_SVC_HEALTH_PATH` or `DYN_HTTP_SVC_LIVE_PATH`.

## Frontend Responses

A live frontend returns `200 OK`:

```json
{
  "message": "Service is live",
  "status": "live"
}
```

During graceful shutdown, `/live` remains `200 OK` while the Frontend is draining admitted requests.
After the drain finishes or times out and the service enters the stopping state, `/live` returns
`503 Service Unavailable` with status `shutting_down`.

The frontend `/health` endpoint returns `200 OK` while ready and lists the endpoints and instances currently
visible through discovery:

```json
{
  "status": "healthy",
  "endpoints": [
    "dyn://dynamo.backend.generate"
  ],
  "instances": [
    {
      "namespace": "dynamo",
      "component": "backend",
      "endpoint": "generate",
      "instance_id": 7587888160958628000
    }
  ]
}
```

An empty `endpoints` or `instances` array means the frontend does not currently discover workers; it
does not change the frontend response status by itself. As soon as graceful draining starts, `/health`
returns `503 Service Unavailable` with status `not_ready` and a `stage` value that identifies the
current lifecycle stage. This removes the Frontend from readiness routing before admitted responses
finish.

## Model-Specific Serving Readiness

`GET /v1/models/<model>/ready` checks the selected model's declared worker topology,
usable serving engine, and enabled HTTP inference routes. It does not send a canary inference
request or wait for every worker replica or KV-cache event to arrive. For disaggregated deployments,
the required worker roles must be available before the topology is ready.

| HTTP status | Meaning |
| --- | --- |
| `200 OK` | The selected model is ready to serve through its registered HTTP routes |
| `503 Service Unavailable` | The model is known, but topology, engine initialization, route activation, or frontend lifecycle state prevents serving |
| `404 Not Found` | The model is not registered in the frontend's local catalog |

The response retains diagnostic fields such as `model`, `ready`, `reason`, and `namespaces`.
Namespace details describe worker topology; top-level readiness also accounts for the frontend's
local serving state. A diagnostic client can therefore inspect a registered prefill or decode role
in a `503` response even while its required peer is absent. Registration of one role does not mean
the model is ready to serve.

`/health` and `/live` remain independent of model readiness. Operators opt in to model-specific
Kubernetes readiness by selecting this endpoint as the frontend readiness probe; see
[Configure Kubernetes Health Probes](../../kubernetes/operations/observability.mdx#check-and-customize-health-probes).
Changing a probe does not add inference request gating or prevent requests over existing connections.

<Warning>
Previously, this endpoint returned `200 OK` with `ready: false` for known but unready models.
It now returns `503 Service Unavailable` with the diagnostic JSON body preserved. This response
change affects diagnostic clients even if operators keep probing `/health`.

Clients that need discovery or topology details must parse JSON on both `200` and `503`, rather
than discarding non-success responses or raising an error before reading the body. Clients checking
serving readiness should treat `503` as unready and retry; `404` means the model is not yet known.
</Warning>

## System-Status Responses

Workers and standalone routers expose health endpoints only when `DYN_SYSTEM_PORT` is zero or a
positive port. A negative value disables the system-status server.

Before the component is ready, `/health` and `/live` return `503 Service Unavailable`:

```json
{
  "endpoints": {
    "generate": "notready"
  },
  "status": "notready",
  "uptime": {
    "nanos": 313803539,
    "secs": 12
  }
}
```

The worker `/live` and `/health` paths currently report the same system-status payload and status
code. After startup, both paths return `200 OK` with status `ready`:

```json
{
  "endpoints": {
    "clear_kv_blocks": "ready",
    "generate": "ready",
    "load_metrics": "ready"
  },
  "status": "ready",
  "uptime": {
    "nanos": 356504530,
    "secs": 18
  }
}
```

`DYN_SYSTEM_STARTING_HEALTH_STATUS` controls the initial status. The legacy
`DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS` variable is deprecated and no longer controls endpoint
selection in the current runtime.

## Active Canary Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `DYN_HEALTH_CHECK_ENABLED` | `false` | Enables active checks for registered worker endpoints |
| `DYN_CANARY_WAIT_TIME` | `10` | Seconds without successful endpoint activity before sending a canary |
| `DYN_HEALTH_CHECK_REQUEST_TIMEOUT` | `3` | Seconds to wait for the canary response |
| `DYN_HEALTH_CHECK_PAYLOAD` | Backend default | Optional JSON object or `@/path/to/file.json` payload override for unified backends |

Each backend supplies a minimal inference payload. Dynamo marks the request as a health check and
uses the backend's normal endpoint path, so the check covers request handling and engine execution
rather than only process reachability. For the lifecycle and rationale, see
[Observability Architecture](../../developer-guide/knowledge-base/concepts/observability-architecture.md#active-worker-health-checks).

## Related Reference

- [Environment Variables](environment-variables.mdx#health-checks)
- [Local Observability Stack](local-stack.mdx)
- [Configure Kubernetes Health Probes](../../kubernetes/operations/observability.mdx#check-and-customize-health-probes)
