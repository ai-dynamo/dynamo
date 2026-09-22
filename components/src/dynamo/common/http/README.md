# `dynamo.common.http`

HTTP fetch client: a facade (`fetch_bytes` / `close_http_client`) over
an `HttpClient` ABC with a single concrete subclass, `AiohttpClient`
(over `aiohttp.ClientSession`). `DYN_HTTP_BACKEND` is retained for
back-compat but only `aiohttp` is supported; any other value warns and
uses aiohttp.

## Why aiohttp

`AiohttpClient` is the single supported backend. Its connector queues
pending connections in `O(1)`, so latency stays close to the offered
rate when one request fans out to many URLs (e.g. 100 image fetches),
and it exposes a `TCPConnector(resolver=...)` DNS hook that pins the
validated DNS answers as a connect-time SSRF backstop against DNS
rebinding — the default client wires a `BlocklistResolver` (see
`_ssrf_resolver.py`). The connector may return private addresses only when the
`DYN_MM_ALLOW_INTERNAL` deployment baseline **and** the request policy both
allow it, so neither side alone can weaken the other. The client keeps one
session per outcome and closes each resolver itself, because aiohttp closes
only a resolver it created.

> [!IMPORTANT]
> The backstop governs **direct** connections only. When a proxy applies
> (`HTTP_PROXY` / `HTTPS_PROXY` — the session runs `trust_env=True`), the
> connector dials the proxy and the *proxy* resolves the origin, out of this
> resolver's sight, so the check cannot govern the destination.
>
> A policy-protected fetch that a proxy would carry therefore **fails closed**.
> Set `DYN_MM_TRUST_EGRESS_PROXY=1` to assert that the proxy enforces
> destination policy itself, and the fetch proceeds. The gate asks aiohttp
> which proxy applies to that specific URL, so `NO_PROXY` is honored and a
> fetch that goes direct is never refused. It does not apply when
> `DYN_MM_ALLOW_INTERNAL=1`, which already permits private destinations.
>
> Note aiohttp never calls a resolver for an IP literal, so literal blocked
> addresses are `validate_url`'s job rather than the backstop's.

See the
[NeMo Gym aiohttp vs httpx note](https://docs.nvidia.com/nemo/gym/latest/infrastructure/engineering-notes/aiohttp-vs-httpx.html)
for the fan-out latency comparison.

> [!NOTE]
> **Deprecated:** `DYN_HTTP_BACKEND` now accepts only `aiohttp` (any other
> value warns and falls back). The httpx-only knobs — `DYN_HTTP_MAX_KEEPALIVE`,
> `DYN_HTTP_POOL_TIMEOUT`, and `DYN_HTTP_CONCURRENCY` (and their `--http-*`
> flags) — are still accepted for backward compatibility but **ignored**;
> aiohttp consumes none of them. They configured a second HTTP backend that
> has been removed.

## Operator-tunable knobs

See
[`http_args.py`](../configuration/groups/http_args.py) for the full
`DYN_HTTP_*` env-var / `--http-*` CLI-flag reference (pool size,
per-call timeout override, aiohttp keepalive, etc.). Legacy
`DYN_MM_HTTP_*` env vars are still honored.
