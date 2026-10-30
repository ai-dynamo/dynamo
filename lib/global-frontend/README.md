<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Global Frontend

An HTTP entry point that forwards requests to configured or discovered Dynamo pool
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
cluster, namespace, and DGD name; duplicate identities are rejected. For pools
given on the command line, the cluster also names the pool's region.

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

`affinity` needs `--region`, matching the pool's region (the cluster name for
explicit pools). It reads each pool's load from the pool's PoolRelay, given per pool
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

With `--pool-relay`, the router opens catalog, readiness, serving-load, and
KV-load subscriptions over one connection to the Relay. These feed the same
state validation, replica views, and readiness gate as Relay-initiated streams.
A configured Relay pool is eligible only while its model has a live, ready view;
losing any subscription withdraws that replica's view before reconnecting.
This gate applies to every routing policy. Static pools without `--pool-relay`
retain their existing eligibility.

A report counts for three of its Relay's report windows. A current view with
unknown or expired load counts as full: its sessions move to the nearest pool
with room, if any, and new requests try it after pools whose load is known.
A producer-generation change clears the previous generation's load until a
matching window arrives. The round trip to each region is the median TCP connect time to
its pools, measured every 30 seconds; the request's own region counts as
zero.

## Relay-initiated state

Choose the connection direction for the deployment. Router-initiated mode uses
`--pool-relay` and the Relay's existing subscription RPCs. Relay-initiated mode
uses `PoolViewBroadcaster` and the router's `PoolViewReceiver`. Both apply decoded
state through the same validation and `PoolViews`; connection setup and reconnect
handling remain specific to each direction.

To read candidate pools from Global Discovery and require live Relay state:

```bash
cargo run -p dynamo-global-frontend -- --port 8000 \
  --global-discovery-url http://global-discovery:8080 \
  --relay-view-port 50051
```

Set each Relay's `--global-discovery-url`, `--bind`, and
`--routing-pool <cluster>/<Kubernetes namespace>/<DGD name>`. Scope its discovery
sources to that DGD. Each `GET /routers` endpoint must address a replica's gRPC
receiving port. Relays send a fresh catalog/readiness/load snapshot on every
connection, then updates. Separate Relay views keep active-active reports from
being summed; loss of the last view removes the pool even when a failed GDS read
leaves its candidate list cached. A complete GDS list withdraws omitted pools;
a failed or partial list keeps the previous candidates.

`--relay-view-port` also works with explicit `--pool` candidates. It cannot be
combined with `--pool-relay`; affinity reads the incoming Relay load in this mode.
The receiving listener uses the deployment's trusted network, without mTLS or
source-IP admission.

Use a discovery readiness probe independent of pool availability, such as
`/live`. The existing `/health` requires a routable pool, so using it to discover
new routers could keep Relays from ever finding them.

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

## Limitations

Pool registration remains the Relay deployment's responsibility. There is no
automatic forwarding retry: unreachable pools return 503. The listener has no TLS.

## Tests

Run CPU-only tests with:

```bash
cargo test -p dynamo-global-common -p dynamo-global-router -p dynamo-global-frontend
```

The common `PoolId` identifies a DGD using its deployment coordinates. It is
distinct from `dynamo_kv_router::identity::PoolId`, which identifies an indexer
domain/DC lane. Relay routing scope supplies the DGD boundary; catalog producer
identities retain the indexer-domain/DC scope.
