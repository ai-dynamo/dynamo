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
and it exposes a `TCPConnector(resolver=...)` DNS hook used for the
connect-time SSRF backstop. See the
[NeMo Gym aiohttp vs httpx note](https://docs.nvidia.com/nemo/gym/latest/infrastructure/engineering-notes/aiohttp-vs-httpx.html)
for the fan-out latency comparison.

> [!NOTE]
> **Deprecated:** `DYN_HTTP_BACKEND` now accepts only `aiohttp` (any
> other value warns and falls back), and the `DYN_HTTP_CONCURRENCY`
> semaphore is a no-op. Both existed for a second HTTP backend that has
> been removed.

## Operator-tunable knobs

See
[`http_args.py`](../configuration/groups/http_args.py) for the full
`DYN_HTTP_*` env-var / `--http-*` CLI-flag reference (pool size,
per-call timeout override, aiohttp keepalive, etc.). Legacy
`DYN_MM_HTTP_*` env vars are still honored.
