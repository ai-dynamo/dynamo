<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Global Frontend

An HTTP entry point that forwards requests to explicitly configured Dynamo pool
frontends. It reads only each request's `model` and passes the request and the
pool's response through as they are; the pool frontend owns validation,
preprocessing, generation, and the response format. Pool identity types live
in `dynamo-global-common`, and pool selection in `dynamo-global-router`.

```bash
cargo run -p dynamo-global-frontend -- --port 8000 \
  --pool ohio/dynamo/blue=Qwen/Qwen3-0.6B@http://127.0.0.1:8001 \
  --pool oregon/dynamo/green=Qwen/Qwen3-0.6B@http://127.0.0.1:8002
```

Each `--pool` uses `<cluster>/<namespace>/<DGD name>=<model>@<frontend URL>`,
where the cluster is the ID the pool's PoolRelay runs with (`--cluster-id`).
Repeat the flag for different pools. Comma-separated URLs after `@` configure
multiple frontends for the same pool. Pool identities are derived from the
cluster, namespace, and DGD name; duplicate identities are rejected. Until pools
come from Global Discovery, the cluster also names the pool's region.

Every `POST /v1/...` request (chat completions, completions, Responses,
embeddings, Anthropic messages, and so on) goes to the same route on the
selected pool's frontend, with the client's headers except hop-by-hop ones,
including `x-dynamo-session-id`, `x-request-id`, and `authorization`. The pool's
status, headers, and body come back unchanged, streamed or not, including its
error responses and SSE keep-alives. A request whose model no pool serves gets
404, and one whose pool cannot be reached gets 503. `/v1/models` lists the
configured models, `/health` is healthy while some pool has a frontend, and
`/live` always answers.

## Pool selection

`--router-policy` chooses how each request picks a pool among those serving its
model; every policy then rotates across the pool's frontends.

| Policy | Picks |
|---|---|
| `round-robin` (default) | Each pool in turn. |
| `least-in-flight` | The pool with the fewest of this process's requests in flight. |
| `affinity` | The session's pool, nearest region first, moving only for a pool with much more room. |

`affinity` keeps each session's turns on one pool, so the pool's KV cache for
the session is reused. It compares pools by how full they are against their own
capacity: the fuller of their request slots (requests in flight over the
requests their workers can run at once) and their KV blocks. So pools of any
speed or size compare. A request goes to its session's pool, or, for a new
session or a request without a session ID, to the least full pool in the
nearest region. It goes elsewhere only when another pool has more than
`--headroom-factor` (default 2) times that pool's spare capacity, choosing the
nearest such pool, or when that pool is full, choosing the nearest pool with any
room. The session's pin moves with it. Pins are per process and expire after
`--session-idle-secs` (default 1800) without a request.

`affinity` needs `--region`, this process's region named as `--pool` names
clusters. It reads each pool's load from the pool's PoolRelay, given per pool
over plain HTTP. Each pool needs a Relay of its own, since a Relay reports the
load of everything it watches as one:

```bash
cargo run -p dynamo-global-frontend -- --port 8000 \
  --router-policy affinity --region ohio \
  --pool ohio/dynamo/blue=Qwen/Qwen3-0.6B@http://127.0.0.1:8001 \
  --pool oregon/dynamo/green=Qwen/Qwen3-0.6B@http://127.0.0.1:8002 \
  --pool-relay ohio/dynamo/blue=http://127.0.0.1:50051 \
  --pool-relay oregon/dynamo/green=http://127.0.0.1:50052
```

A report counts for three of its Relay's report windows, so a reconnect or a
briefly incomplete view does not lose a pool's load. A pool without a Relay, or
whose last report has aged out, counts as full: its sessions move to the
nearest pool with room, if any, and new requests try it after the pools whose
load is known. The round trip to each region is the median TCP connect time to
its pools, measured every 30 seconds; the request's own region counts as
zero.

## Metrics

`/metrics`, on the same port, serves the Global Frontend's Prometheus metrics,
named `dynamo_global_frontend_*`: requests per pool by response status,
`no_response` when the pool sends no response, or `cancelled` when the client
leaves before the pool's response headers arrive; requests answered without a
pool; time to first byte and duration per pool; and requests in flight per
pool. The `pool` label is the pool's
`<cluster>/<namespace>/<DGD name>` and `region` its region. On requests, `local`
says whether the pool is in `--region` (`unknown` without it), which gives the
share of requests kept in the region. Nothing is labelled from the request
body. Inter-token latency and token counts need the stream parsed, so they come
from the pool frontends' own metrics. See the
[Metrics Catalog](../../docs/fern/pages/reference/observability/metrics-catalog.mdx#global-frontend-metrics)
for each family.

## Deployment notes

- The Global Frontend holds a request's body, as received, only until it is
  written to the pool, not while the pool queues it. The body limit is
  `DYN_HTTP_BODY_LIMIT_MB` (default 45), as for Dynamo's HTTP service.
- A client that disconnects closes its connection to the pool, which cancels
  the request there.
- Requests to other regions go over long-lived connections that sit idle
  between bursts. Set `net.ipv4.tcp_slow_start_after_idle=0` on the Global
  Frontend's hosts, or each idle connection starts over with a small send
  window and a long prompt costs extra round trips.

## Replica sync

With `--selection core`, each Global Frontend's embedded router core books
requests on pools and, under `--router-policy affinity`, pins sessions to them.
Replicas share those bookings and pins so every frontend sees every frontend's
requests in flight (DEP [#17639](https://github.com/ai-dynamo/dynamo/issues/17639),
companion [#17638](https://github.com/ai-dynamo/dynamo/issues/17638)). Two
transports carry them; a frontend uses one or the other, and `--peer` names
the other frontends.

| Flags | Transport |
|-------|-----------|
| `--peer http://host:port` | The gRPC `ReplicaSync` stream, served on this frontend's API port (`--port`) beside the OpenAI routes. Each peer is another frontend's API address. No extra port. |
| `--replica-sync-port N --peer tcp://host:port` | The router core's own ZMQ mesh. This frontend publishes on port `N`; each peer is another frontend's `--replica-sync-port` endpoint. |

Rules: every `--peer` must use the same scheme; `tcp://` peers need
`--replica-sync-port`; `http://` peers refuse it, since the gRPC stream serves
on the API port; any of these flags needs `--selection core`; `https://` peers
are refused until the stream carries TLS.

The gRPC contract is the `ReplicaSync` service in
`lib/global-common/proto/replica_sync.proto`, package
`dynamo.globalrouter.replica.v1`: a subscriber opens `SubscribeReplicaEvents`
with its process id and `CONTRACT_MARKER` (a mismatch is refused with
`FAILED_PRECONDITION`), receives a snapshot frame, then event batches as they
happen. Each event names its core partition (`model_name`, `routing_group`)
and its publisher (`router_id` / `writer_id`), which a receiver uses to drop
its own echoes; a prompt's `token_sequence` is never carried, since a pool has
no KV blocks the core could match. The snapshot is empty for now: the core
exports neither its live bindings nor its open bookings, so a peer that joins
sees bookings already open only when they free. A subscriber that falls behind
by more than the publisher's frame buffer loses the oldest frames (counted in
`ReplicaSyncStats`), as with the mesh. The dialer reconnects with a growing
delay (0.5 s to 30 s) as the Relay client does.
`dynamo-global-common` generates the code behind its `grpc` feature;
`dynamo_global_router::replica_sync` holds the conversions, the hub, the
server routes and the dialer.

## Limitations

Pools are given on the command line; there is no discovery client, polling, or
registration. There is no automatic failover or retry: unreachable
pools return 503. The listener has no TLS.

## Tests

Run CPU-only tests with:

```bash
cargo test -p dynamo-global-common -p dynamo-global-router -p dynamo-global-frontend
```

The common `PoolId` identifies a DGD using its deployment coordinates. It is
distinct from `dynamo_kv_router::identity::PoolId`, which identifies an indexer
domain/DC lane; Global View integration will need an explicit mapping.
