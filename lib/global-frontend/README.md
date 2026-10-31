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

With `--selection-core`, each Global Frontend's embedded router core books
requests on pools and, under `--router-policy affinity`, pins sessions to them.
With `--replica-sync`, replicas share those bookings and pins so every frontend
sees every frontend's requests in flight (DEP [#17639](https://github.com/ai-dynamo/dynamo/issues/17639),
companion [#17638](https://github.com/ai-dynamo/dynamo/issues/17638)).
`--replica-sync-transport` picks one of two transports:

| `--replica-sync-transport` | Transport | Peers |
|----------------------------|-----------|-------|
| `grpc` (default) | The gRPC `ReplicaSync` stream, served on this frontend's API port (`--port`) beside the OpenAI routes; no extra port. Served from startup, before any peer is known, so a replica can be dialed as soon as it is listed. | Other frontends' API addresses, from Global Discovery or `--peer http://host:port`. |
| `zmq` | The router core's own ZMQ mesh, published on `--replica-sync-port N`. | Other frontends' `--replica-sync-port` endpoints, `--peer tcp://host:port`; static only. |

| Flag | Meaning |
|------|---------|
| `--replica-sync` | Share bookings and pins. Off by default. Needs `--selection-core`. |
| `--replica-sync-transport grpc\|zmq` | The transport. Default `grpc`. |
| `--replica-sync-port N` | `zmq` only: the port this frontend publishes on. Required for `zmq`, refused for `grpc`. |
| `--global-discovery <URL>` | `grpc` only: Global Discovery's base URL. Its `GET /routers` lists every frontend's API URL; every entry but this frontend's becomes a peer, and the list is polled every 30 s and followed as it changes. Needs `--replica-url`. |
| `--replica-url <URL>` | This frontend's own API URL as Global Discovery lists it, so it does not dial itself. Needs `--global-discovery`. |
| `--peer <URL>` | A static peer, as the transport dials it. Repeat the flag for each. Static peers stay whatever discovery lists. |

Rules: every flag here but `--replica-url` needs `--replica-sync`; a peer's
scheme must match the transport; an `https://` peer, on the command line or in
discovery's listing, is refused or skipped (see TLS below). A failed discovery
poll keeps the last listing.

Over `zmq`, the core keeps its own age-based expiry (five minutes) for every
booking, since nothing here ends a peer's booking if that peer is lost; a
request that streams for longer stops counting toward its pool's load once it
expires. Over `grpc` the hub holds the leases for mirrored bookings, and
without replica sync a booking lasts exactly as long as its request.

```bash
cargo run -p dynamo-global-frontend -- --port 8000 --selection-core \
  --router-policy affinity --region ohio \
  --pool ohio/dynamo/blue=Qwen/Qwen3-0.6B@http://127.0.0.1:8001 \
  --pool-relay ohio/dynamo/blue=http://127.0.0.1:50051 \
  --replica-sync --global-discovery http://discovery.dynamo:8080 \
  --replica-url http://10.42.0.7:8000
```

On Kubernetes, a plain Deployment behind a Service is enough: replicas need
no stable names, since each finds the others through Global Discovery and
serves the stream on its API port from startup. `--replica-url` is the pod's
own routable URL, the one the pod publishes to discovery (the pod IP and
`--port`, from the downward API), not the Service's; a Service address would
make the frontend skip or dial a random replica. Rolling restarts need
nothing more: a replaced pod drops out of the listing and its peers stop
dialing it on the next poll.

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
`ReplicaSyncStats`). The dialer reconnects with a growing delay (0.5 s to
30 s) as the Relay client does. `dynamo-global-common` generates the code
behind its `grpc` feature; `dynamo_global_router::replica_sync` holds the
conversions, the hub, the server routes, the dialer, and the peer-set
follower the frontend's discovery poll feeds.

### TLS

Neither the Relay client (`--pool-relay`) nor replica sync carries TLS yet:
both are built on the workspace's `tonic` without a TLS feature, and both
reject `https://` with a clear message. They will gain TLS together, behind
one feature, so the frontend's outbound gRPC is either all plain or all
encrypted; the listener's TLS is a separate concern (see Limitations).

## Limitations

Pools are given on the command line; Global Discovery is read only for the
other frontends, not yet for pools, and the frontend does not register itself.
There is no automatic failover or retry: unreachable pools return 503. The
listener has no TLS.

## Tests

Run CPU-only tests with:

```bash
cargo test -p dynamo-global-common -p dynamo-global-router -p dynamo-global-frontend
```

The common `PoolId` identifies a DGD using its deployment coordinates. It is
distinct from `dynamo_kv_router::identity::PoolId`, which identifies an indexer
domain/DC lane; Global View integration will need an explicit mapping.
