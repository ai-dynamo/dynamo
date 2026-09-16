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
and it exposes a `TCPConnector(resolver=...)` DNS hook that can pin
validated DNS answers for a connect-time SSRF backstop (the default
client here uses the stock resolver; the pinning lands as a follow-up). See the
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
