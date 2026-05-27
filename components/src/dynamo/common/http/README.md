# `dynamo.common.http`

HTTP fetch client. Backend-neutral facade
(`fetch_bytes` / `close_http_client`) over an `HttpClient` ABC with
two concrete subclasses: `AiohttpClient` (default) and `HttpxClient`.
Backend selection: `DYN_HTTP_BACKEND={aiohttp,httpx}`.

## Why aiohttp is the default

Under high concurrency (e.g. one request fanning out to 100 image
URLs) the httpx backend hits `httpx.PoolTimeout`. aiohttp scales
markedly better on the same workload — see the
[NeMo Gym aiohttp vs httpx note](https://docs.nvidia.com/nemo/gym/latest/infrastructure/engineering-notes/aiohttp-vs-httpx.html)
and [openai-python#1596](https://github.com/openai/openai-python/issues/1596).

Root cause is in httpx's pool. The maintenance routine in
[`httpcore._async.connection_pool`](https://github.com/encode/httpcore/blob/master/httpcore/_async/connection_pool.py#L303-L309)
runs *"whenever a new request is added or removed from the pool"* and
is `O(queue_size × pool_size)` per call, so cost grows quadratically
with backlog. aiohttp's connector queues natively in `O(1)`, which is
why our httpx backend needed a process-wide semaphore
(`DYN_HTTP_CONCURRENCY`) in front of the pool to keep
`PoolTimeout` from leaking up the stack. That semaphore is redundant
under aiohttp.

### Benchmark

500 rps × 10k requests across server-processing-time buckets (lower
is better, all latencies in ms):

```
== request_rate=500 rps  requests=10000 ==

[sweep] mean_ms=50  order=['aiohttp', 'httpx']
backend  wall(s)     avg     p50     p90     p99
httpx       20.1    75.7    51.0   158.7   241.1
aiohttp     20.1    50.7    50.6    50.8    51.1

[sweep] mean_ms=100  order=['httpx', 'aiohttp']
httpx       23.2  1550.4  1484.3  2430.2  3164.9
aiohttp     20.1   101.4   100.6   100.9   101.2

[sweep] mean_ms=200  order=['aiohttp', 'httpx']
httpx       43.2 11833.2 11762.6 21167.1 23025.2
aiohttp     20.8   320.5   231.2   733.3   830.3

[sweep] mean_ms=300  order=['httpx', 'aiohttp']
httpx       62.2 21229.8 21318.0 37834.0 41728.4
aiohttp     32.2  6355.0  6395.1 11119.6 12071.7
```

Both backends are equivalent below saturation (`mean_ms=50`). Past
saturation httpx degrades super-linearly while aiohttp stays close to
the offered rate.

Reproduce:

```bash
python -m benchmarks.multimodal.http.sweep \
    --server-processing-time-means-ms 50,100,200,300 \
    --request-rate 500 \
    --requests 10000 \
    --timeout 60
```

## Operator-tunable knobs

See
[`http_args.py`](../configuration/groups/http_args.py) for the full
`DYN_HTTP_*` env-var / `--http-*` CLI-flag reference (pool size,
per-call timeout override, httpx-only semaphore concurrency, aiohttp
keepalive, etc.). Legacy `DYN_MM_HTTP_*` env vars are still honored.
