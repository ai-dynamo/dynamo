<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Global Frontend

An HTTP entry point that forwards requests to explicitly configured Dynamo pool
frontends. It reuses Dynamo's HTTP service and performs no tokenization; the pool
frontend owns preprocessing and generation. Shared pool identity and description
types live in `dynamo-global-common`.

```bash
cargo run -p dynamo-global-frontend -- --port 8000 \
  --pool site-a/dynamo/blue=Qwen/Qwen3-0.6B@http://127.0.0.1:8001 \
  --pool site-b/dynamo/green=Qwen/Qwen3-0.6B@http://127.0.0.1:8002
```

Each `--pool` uses `<site>/<namespace>/<DGD name>=<model>@<frontend URL>`.
Repeat the flag for different pools. Comma-separated URLs after `@` configure
multiple frontends for the same pool. Pool identities are derived from the
site, namespace, and DGD name; duplicate identities are rejected.

`/v1/models` lists the configured models. Chat completions and
completions support streaming and non-streaming requests; the existing Responses
API translation goes through the chat forwarding engine. Non-streaming requests
ask the pool for a stream including usage, which the HTTP service aggregates.
The `x-dynamo-session-id` header is forwarded to the selected pool.

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
sites. It reads each pool's load from the pool's KV DC Relay, given per pool:

```bash
cargo run -p dynamo-global-frontend -- --port 8000 \
  --router-policy affinity --region site-a \
  --pool site-a/dynamo/blue=Qwen/Qwen3-0.6B@http://127.0.0.1:8001 \
  --pool site-b/dynamo/green=Qwen/Qwen3-0.6B@http://127.0.0.1:8002 \
  --pool-relay site-a/dynamo/blue=http://127.0.0.1:50051 \
  --pool-relay site-b/dynamo/green=http://127.0.0.1:50052
```

A pool without a Relay, or whose Relay has not reported in the last 3 seconds,
has no known load: requests leave it for any pool whose load is known, and new
requests try it only after those. The round trip to each region is the median
TCP connect time to its pools, measured every 30 seconds; the request's own
region counts as zero.

## Deployment notes

- The Global Frontend holds a request's prompt only while it parses and
  uploads it, not while the pool queues it.
- Requests to other regions go over long-lived connections that sit idle
  between bursts. Set `net.ipv4.tcp_slow_start_after_idle=0` on the Global
  Frontend's hosts, or each idle connection starts over with a small send
  window and a long prompt costs extra round trips.

## Limitations

This first implementation uses static configuration. The catalog interface also
accepts snapshots for embedding and tests, but there is no discovery client,
polling, or registration. There is no automatic failover or retry: unreachable
pools return 503. Upstream HTTP errors use the existing frontend error policy:
404 remains 404, while an upstream 502 is classified as internal and returns
500; diagnostic bodies are sanitized. The existing SSE
codec does not distinguish a clean upstream EOF from `[DONE]`; premature clean
EOF handling remains a limitation of this draft. Request-ID/trace propagation
and authentication header forwarding are not implemented.

## Tests

Run CPU-only tests with:

```bash
cargo test -p dynamo-global-common -p dynamo-global-router -p dynamo-global-frontend --no-default-features
```

The common `PoolId` identifies a DGD using its deployment coordinates. It is
distinct from `dynamo_kv_router::identity::PoolId`, which identifies an indexer
domain/DC lane; Global View integration will need an explicit mapping.
