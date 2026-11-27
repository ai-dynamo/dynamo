---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Held Workers
subtitle: Workers that frontends discover but route no ordinary traffic to
---

**Experimental.** A held worker is discovered by every frontend but takes no ordinary traffic. Each frontend builds its routing pipeline for the worker while it is held, so a controller can check the worker through a frontend before the worker serves clients.

## Routing Behavior

While a worker is held:

- **Automatic selection skips it.** This applies to every router mode: round-robin, random, least-loaded, device-aware and KV. It also applies to session affinity. A session that is bound to a held worker is rebound to a routable worker. Direct mode selects no worker on its own, so there the request is refused.
- **A request that names the worker reaches it.** Name the worker with `nvext.backend_instance_id`, `nvext.decode_worker_id` or `nvext.prefill_worker_id`, or with the `x-dynamo-worker-instance-id` and `x-dynamo-prefill-instance-id` headers. The other routing checks still apply to that request. A worker type that the request does not name is selected automatically, so to reach a held prefill and decode pair, name both workers.
- **It adds no capacity.** A held worker gives its worker set no routing weight and does not make its model ready. A model whose workers are all held is not listed by `/v1/models`, and the frontend serves only the requests that name one of those workers.

Every frontend and router that discovers the worker must support held workers. A frontend that predates this feature ignores the hold and sends ordinary traffic to a held worker.

To see which workers a frontend has discovered as held, query its health route. A held instance carries `"held": true`:

```bash
curl "http://<frontend-address>/health?instances=true"
```

## Start a Worker Held

To start a worker held, set `DYN_START_HELD=true` in its environment. Set it on workers only. The setting is off by default, and it has two requirements:

- **A key-value discovery backend**, such as etcd. With Kubernetes discovery the worker fails at startup, because a worker Pod is held there by its readiness gate.
- **The system status server**, which `DYN_SYSTEM_PORT` enables. The worker fails at startup without it, or when the server cannot start, because nothing could release the hold.

A held worker reports the hold on its system status server:

```bash
curl http://<worker-system-address>/v1/hold
```

```json
{"instance_id": 7587889441921245527, "held": true}
```

While the worker is held, its `/health` route returns `503` with `"held": true`, and its `/live` route is unaffected. A Kubernetes-hosted worker therefore stays NotReady until its hold is released.

## Release a Held Worker

To release a worker, send its instance ID to its release route. The instance ID can be a number or a decimal string:

```bash
curl -X POST http://<worker-system-address>/v1/hold/release \
  -H 'Content-Type: application/json' \
  -d '{"instance_id": "7587889441921245527"}'
```

```json
{"instance_id": 7587889441921245527, "held": false, "released": true}
```

| Status | Meaning |
|---|---|
| `200` | The worker is released. `released` is `false` when it already was, so a retry is safe. |
| `400` | The request body names no instance. |
| `409` | The request names another instance, for example the worker that this process replaced at the same address. The worker keeps its hold. |
| `500` | The worker could not update all of its discovery records. The worker is still held. Send the release again to complete it. |

The worker clears the hold on its own discovery records. Frontends see the change through their discovery watch and keep the pipeline they built for the worker.

The hold lasts for the life of the worker process. A released worker stays released, and a new worker process starts held again.

## Kubernetes Readiness Gate

With Kubernetes discovery, a worker Pod that lists the `nvidia.com/dynamo-frontend-admitted` readiness gate is held from the moment its containers are ready until the Pod is Ready.

To make a frontend discover such Pods, set `DYN_KUBE_DISCOVERY_HELD_PODS=true` in the frontend's environment. The frontend then watches Pods in addition to EndpointSlices, so its ServiceAccount needs `list` and `watch` on Pods. Without the setting, the frontend discovers a gated Pod only when the Pod is Ready.

A gated worker does not see itself in discovery either. A worker that waits for its own endpoint to appear before it serves, as `dynamo.mocker` does, needs the same setting, or a request that names it gets no response.

A frontend lists a new instance on its health route a moment before it can route to it, because it builds the model's pipeline after it discovers the worker. A controller that checks a held worker through a frontend must retry the check until it succeeds.
