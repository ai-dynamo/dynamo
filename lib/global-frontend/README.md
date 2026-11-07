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
room. The session's pin moves with it. Pins expire after
`--session-idle-secs` (default 1800) without a request, and are held in this
process unless `--affinity-store dynamodb` shares them (see
[Session affinity storage](#session-affinity-storage)).

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

### Session affinity storage

`--affinity-store` decides where `affinity`'s session pins live. `memory`, the
default, keeps them in this process: a restart or a second replica starts with
none. `dynamodb` keeps them in a DynamoDB table, so replicas of this binary
agree on a session's pool and a restart resumes where it left off. No other
component reads or writes the table, and nothing else in Dynamo depends on it.

Affinity is advisory throughout. A lookup or write that fails, or that runs
past `--affinity-timeout-ms` (default 20), leaves the request on the pool the
headroom rules already chose: no request is refused because storage is down.
After `--affinity-breaker-failures` consecutive failures (default 5) the
frontend stops consulting storage for `--affinity-breaker-cooldown-ms` (default
5000), so an outage costs one deadline per cooldown rather than one per
request; a single request then retests it.

A request that keeps its session's pool writes nothing. The binding is written
when a session is new or its pin has expired, when it moves pools, when an
undecodable row is repaired, and once the pin is past half its idle window
(`--session-idle-secs`), which keeps an active session from expiring. So most
turns of a conversation only read, and parallel requests of one session, such
as concurrent tool calls, do not conflict with each other.

Reads are cached in this process for `--affinity-cache-ttl-ms` (default 1000,
0 to disable) across at most `--affinity-cache-entries` sessions (default
100000), so a busy session reads the table about once per TTL rather than once
per request. When another replica moves a session, this replica may keep
routing it to the old pool until its cached entry ages out, which is at most
the TTL. A stale entry never overwrites the newer binding: a write made from it
loses its conditional write and is re-read.

#### Table

Provision the table yourself; the frontend never creates it. It needs a string
partition key `itemHashKey`, a string sort key `sortKey`, and TTL enabled on
`expiresAt`. On-demand capacity suits the access pattern: one `GetItem` per
cache miss, and a `PutItem` only when a binding changes or needs a refresh.

A Global Table (MREC) uses exactly this schema, and its pins are soft: last
writer wins between regions. Conditional writes serialize against the
contacted regional replica only, so when one session reaches frontends in two
regions before either's write has replicated, both writes apply and
replication keeps the later one. Since a kept pin is not rewritten, the
session then settles on that pool in both regions; it is not a global lock.

```bash
aws dynamodb create-table --table-name dynamo-global-affinity \
  --billing-mode PAY_PER_REQUEST \
  --attribute-definitions \
    AttributeName=itemHashKey,AttributeType=S \
    AttributeName=sortKey,AttributeType=S \
  --key-schema \
    AttributeName=itemHashKey,KeyType=HASH \
    AttributeName=sortKey,KeyType=RANGE
aws dynamodb update-time-to-live --table-name dynamo-global-affinity \
  --time-to-live-specification Enabled=true,AttributeName=expiresAt
```

Rows hold `poolId`, `revision`, `updatedAt`, `expiresAt` and `schemaVersion`.
Keys are BLAKE3 digests, so no session ID, credential or model name is stored
in the clear. A row that no reader can decode — a newer `schemaVersion`, or a
damaged write — is replaced by the next request for that session rather than
stranding it, which is also what lets a mixed-version rollout recover on its
own. TTL deletion is DynamoDB's own, within its usual delay; a row past
`expiresAt` is treated as expired on read whether or not it has been swept.

#### Scope and tenancy

`--affinity-scope` is the namespace bindings are shared in, and every replica
that should agree must pass the same one. It is deployment configuration and is
never derived from a request. `--affinity-policy` (default `headroom-v1`)
separates incompatible placement policies within a scope, so rolling out a new
policy does not inherit the old one's pins.

Within a scope, a session's binding is also keyed by a digest of the request's
`authorization` header, so two tenants that reuse one session ID never share a
binding; requests with no credential share one `anonymous` namespace. Note what
this is not: the Global Frontend does not validate credentials, and selection
happens before the pool authorizes the request, so a request the pool will
reject can still create or refresh a binding. Treat the table as writable by
anyone who can reach the frontend, and keep the frontend behind whatever admits
traffic to the deployment. Give each tenant its own `--affinity-scope` where
their bindings must not share a table at all.

#### IAM

The frontend's role needs only item access on the table:

```json
{
  "Effect": "Allow",
  "Action": ["dynamodb:GetItem", "dynamodb:PutItem"],
  "Resource": "arn:aws:dynamodb:<region>:<account>:table/dynamo-global-affinity"
}
```

Credentials and region come from the standard AWS provider chain, with
`--affinity-ddb-region` overriding the region and `--affinity-ddb-endpoint`
pointing at DynamoDB Local for development. `--affinity-consistent-reads`
(default true) asks for strongly consistent reads within the contacted region.

#### Starting it

```bash
cargo run -p dynamo-global-frontend -- --port 8000 \
  --router-policy affinity --region ohio \
  --pool ohio/dynamo/blue=Qwen/Qwen3-0.6B@http://127.0.0.1:8001 \
  --pool oregon/dynamo/green=Qwen/Qwen3-0.6B@http://127.0.0.1:8002 \
  --pool-relay ohio/dynamo/blue=http://127.0.0.1:50051 \
  --pool-relay oregon/dynamo/green=http://127.0.0.1:50052 \
  --affinity-store dynamodb \
  --affinity-scope production \
  --affinity-ddb-table dynamo-global-affinity
```

Every flag has an environment variable (`DYN_GLOBAL_AFFINITY_STORE`,
`DYN_GLOBAL_AFFINITY_SCOPE`, `DYN_GLOBAL_AFFINITY_DDB_TABLE` and the rest), for
deployments that configure the frontend through the environment.

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

Set each Relay's `--global-discovery-url`, `--bind`,
`--routing-pool <cluster>/<Kubernetes namespace>/<DGD name>`,
`--routing-pool-region`, and, with registrations-mode discovery,
`--routing-pool-frontends`. The Relay registers its pool with `POST /pools` and
renews it on the lease heartbeat. Scope its discovery
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

## Router self-registration

With GDS configured for `--router-discovery registrations`, pass both registration flags:

```bash
global-frontend --port 8000 --relay-view-port 50051 \
  --global-discovery-url http://global-discovery:8080 \
  --router-replica-id cluster-a/pod-uid \
  --router-advertise-url http://10.42.0.5:50051
```

The advertised URL must reach this replica's state receiver from every Pool Relay. A Pod IP on a
routed private network works; a Service or load balancer shared by several replicas does not.
Kubernetes can provide `metadata.uid` and `status.podIP` through the Downward API. Bracket IPv6
addresses in URLs. Supply a new replica ID if the endpoint changes; reuse an ID only for the same
incarnation at the same address, such as a container restart in an unchanged Pod.

The frontend registers after binding its listeners, before pools are available, then renews on the
lease interval returned by GDS. Transient errors retry with bounded backoff. Shutdown stops renewal,
and GDS removes the router after its lease expires. Both flags require `--global-discovery-url` and
`--relay-view-port`; omitting both preserves externally managed static/MCS membership.

## Metrics

`/metrics`, on the same port, serves the Global Frontend's Prometheus metrics,
named `dynamo_global_frontend_*`: requests per pool by response status,
`no_response` when the pool sends no response, or `cancelled` when the client
leaves before the pool's response headers arrive; requests answered without a
pool; time to first byte and duration per pool; requests in flight per
pool; and the time to select a pool. With `--router-policy affinity` it also
serves:

- `overhead_affinity_store_ms`, the time each session request spent in affinity
  storage, named and bucketed like the local router's
  `dynamo_router_overhead_shared_cache_query_ms`;
- `affinity_lookups_total` by `outcome`: `hit` or `miss`;
- `affinity_store_errors_total` by `kind`, counting each session request that
  routed without affinity: `backend`, `timeout`, `conflict`, or `breaker_open`
  (storage was not consulted);
- `affinity_store_writes_total` by `reason`: `create`, `move`, `refresh` or
  `repair`;
- `affinity_cache_total` by whether the in-process cache answered a read.

The `pool` label is the pool's
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

## Router core

With `--use-selection-core`, the Global Frontend selects pools through an
embedded router core: each pool is a core worker and the policy is a picker
plugin. Only `round-robin` runs on the core for now, picking as the
round-robin selector does.

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
