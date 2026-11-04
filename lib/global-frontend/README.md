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
| `least-in-flight` | The nearest pool with the fewest requests in flight (with `--replica-sync`, every replica's), moving only for a pool with much more room. |
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

`least-in-flight` places requests by the same rule, with requests in flight
over each pool's capacity as the load: this process's requests in flight to the
pool over the requests its workers can run at once, as its Relay reports. Pools
are ordered nearest region first, then less loaded, then fewer in flight; the
request goes to the first, rotating among pools that tie with it, unless
another pool has more than `--headroom-factor` times its spare room. A pool
without a Relay report counts as full, so without `--pool-relay` requests stay
on the nearest pools, fewest in flight first.

`affinity` and `least-in-flight` need `--region`, matching the pool's region
(the cluster name for explicit pools). They read each pool's load from the
pool's PoolRelay, given per pool over plain HTTP. Each pool needs a Relay of
its own, since a Relay reports the load of everything it watches as one:

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
leaves its candidate list cached. The shared `PoolCache` adds and updates pools
from partial GDS lists while retaining omitted pools. Only a complete list
withdraws omitted pools; failed reads keep the last known candidates.

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
pool; time to first byte and duration per pool; requests in flight per
pool; the time to select a pool; and, with `--use-selection-core`, the
router core's affinity pin bookings and, with `--replica-sync`, the
replica-sync stream's peers, frames and losses. The `pool` label is the pool's
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

With `--use-selection-core`, the Global Frontend selects pools through an
embedded router core: each pool is a core worker and the policy is a picker
plugin. Every policy runs on the core, picking as the selectors do. A
least-in-flight pick counts as pending on its pool until the request is
counted, so concurrent picks see each other. Under `affinity` the core's
session table holds each session's pin; a session books only to set, move or
refresh its pin, and the booking is freed at once. Session IDs longer than
256 bytes are refused with 400.

With `--replica-sync`, replicas share their requests in flight and session
pins, so every frontend sees every frontend's load and pins. They travel on the
gRPC `ReplicaSync` stream, served on this frontend's API port (`--port`) beside
the OpenAI routes, from startup, before any peer is known, so a replica can be
dialed as soon as it is listed.

| Flag | Meaning |
|------|---------|
| `--replica-sync` | Share requests in flight and session pins. Off by default. Needs `--use-selection-core`. |
| `--global-discovery <URL>` | Global Discovery's base URL. Its `GET /routers` lists every frontend's API URL; every entry but this frontend's becomes a peer, and the list is polled every 30 s and followed as it changes. Needs `--replica-url`. |
| `--replica-url <URL>` | This frontend's own API URL as Global Discovery lists it, so it does not dial itself. Needs `--global-discovery`. |
| `--peer <URL>` | A static peer, as its API address `http://host:port`. Repeat the flag for each. Static peers stay whatever discovery lists. |

Rules: every flag here but `--replica-url` needs `--replica-sync`; an
`https://` peer, on the command line or in discovery's listing, is refused or
skipped (see TLS below). A failed discovery poll keeps the last listing.

Requests in flight travel on a timer, not per request: every 100 ms each
replica publishes its count per pool, one small frame whatever the request
rate, and a receiver counts a peer's last frame for 500 ms, so a peer that
leaves stops counting within half a second. A peer's count can be up to one
interval old; a replica's own is exact. Session pins travel when they change.
The core's pin bookings are not replicated: they are freed as soon as the pin
is committed.

```bash
cargo run -p dynamo-global-frontend -- --port 8000 --use-selection-core \
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
`FAILED_PRECONDITION`), receives a snapshot frame and, at once, the
publisher's requests in flight per pool (`InflightCount`), then binding
batches as session pins change and `InflightCount` every 100 ms. Each binding
names its core partition (`model_name`, `routing_group`) and its publisher
(`writer_id`), which a receiver uses to drop its own echoes. The snapshot is
empty for now: the core does not export its live bindings, so a peer that
joins learns a session's pin when it next changes or is refreshed. A
subscriber that falls behind
by more than the publisher's frame buffer loses the oldest frames (counted in
`ReplicaSyncStats`). The dialer reconnects with a growing delay (0.5 s to
30 s) as the Relay client does. `dynamo-global-common` generates the code
behind its `grpc` feature; `dynamo_global_router::replica_sync` holds the
conversions, the hub, the server routes, the dialer, and the peer-set
reconciler (`reconcile_peers`) the frontend's discovery poll feeds.

### TLS

Neither the Relay client (`--pool-relay`) nor replica sync carries TLS yet:
both are built on the workspace's `tonic` without a TLS feature, and both
reject `https://` with a clear message. They will gain TLS together, behind
one feature, so the frontend's outbound gRPC is either all plain or all
encrypted; the listener's TLS is a separate concern (see Limitations).

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
