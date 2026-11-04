<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Global Frontend

An HTTP entry point that forwards requests to explicitly configured Dynamo pool
frontends. It reads each request's `model`, and under `affinity` the input of a
request that names no session, and passes the request and the
pool's response through as they are, unless `--deltas` sends the request as
a delta the pool frontend rebuilds it from; the pool frontend owns validation,
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
session, to the least full pool in the nearest region. It goes elsewhere only
when another pool has more than `--headroom-factor` (default 2) times that
pool's spare capacity, choosing the nearest such pool, or when that pool is
full, choosing the nearest pool with any room. The session's pin moves with it.
Pins are per process and expire after `--session-idle-secs` (default 1800)
without a request.

A request's session is the one its headers name: `x-dynamo-session-id`, or a
coding agent's own header (`x-claude-code-agent-id` or
`x-claude-code-session-id`, Codex's `thread-id`, OpenCode's `x-session-id`).
A request that names none continues the session of an earlier request whose
whole input its own input begins with, as a conversation's next turn does.
The input is the text of Anthropic's `system`, then each item of `messages` or
of the Responses `input` (a string `input` being a user message). Field order,
cache breakpoints, and reasoning do not count, so a client may change them in
messages it sends again. A request that continues none, such as a
conversation's first turn, starts a session.

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

## Request deltas

With `--deltas`, which needs `affinity`, a request that continues a
conversation goes to the pool frontend that kept the conversation's last
request, as a delta against it: the messages after those it keeps of that
request (`messages`, or a Responses `input` array), and the request's other
fields only if they changed. The pool frontend rebuilds the request the client
sent before anything else reads it. A conversation is the session the request
names, or else the one its input continues, as for `affinity`. A
conversation's first request, and one whose session is placed on another pool,
go whole, asking the pool frontend to keep them; kept requests do not move
between pools.

Pool frontends keep requests when they set `DYN_HTTP_DELTA_STATE_MB`, the
memory for the last request of each conversation. The least used go first,
and each goes after 30 minutes without the conversation's next request. The
Global Frontend sends deltas only to frontends that have kept a request, and
compresses every body it sends them with zstd at `--compression-level`
(default 1; 0 sends them uncompressed). A frontend that no longer has a
delta's parent answers 409, and the Global Frontend sends it the whole
request. A delta extends only a parent sent with the same `authorization`
header.

List each pool frontend in `--pool` itself: behind a load balancer, a delta
reaches the frontend that keeps its parent only by chance, and the others
answer 409.

## Metrics

`/metrics`, on the same port, serves the Global Frontend's Prometheus metrics,
named `dynamo_global_frontend_*`: requests per pool by response status,
`no_response` when the pool sends no response, or `cancelled` when the client
leaves before the pool's response headers arrive; requests answered without a
pool; time to first byte and duration per pool; requests in flight per pool;
and, with `--deltas`, request body bytes sent per pool, whole or as deltas, and
deltas pools refused for lacking their parent. The `pool` label is the pool's
`<cluster>/<namespace>/<DGD name>` and `region` its region. On requests, `local`
says whether the pool is in `--region` (`unknown` without it), which gives the
share of requests kept in the region. Nothing is labelled from the request
body. Inter-token latency and token counts need the stream parsed, so they come
from the pool frontends' own metrics. See the
[Metrics Catalog](../../docs/fern/pages/reference/observability/metrics-catalog.mdx#global-frontend-metrics)
for each family.

## Deployment notes

- The Global Frontend holds a request's body, as received, only until it is
  written to the pool, not while the pool queues it. A request sent as a delta
  is held until the pool answers, to send it whole if the pool lacks its
  parent. The body limit is
  `DYN_HTTP_BODY_LIMIT_MB` (default 45), as for Dynamo's HTTP service.
- A client that disconnects closes its connection to the pool, which cancels
  the request there.
- Requests to other regions go over long-lived connections that sit idle
  between bursts. Set `net.ipv4.tcp_slow_start_after_idle=0` on the Global
  Frontend's hosts, or each idle connection starts over with a small send
  window and a long prompt costs extra round trips.

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
