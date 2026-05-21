# OpenAI API Surface (Frontend)

The Dynamo Frontend service exposes an OpenAI-compatible HTTP API. Per
, the Frontend lives at `components/src/dynamo/frontend/` in the
Dynamo source. The endpoints and flags vary slightly per release —
this reference targets the 1.2.0 line.

---

## Endpoints

| Endpoint | Method | Purpose | Notes |
|---|---|---|---|
| `/v1/models` | GET | List registered models | Empty until workers register; see for the registration window |
| `/v1/chat/completions` | POST | Chat completions (OpenAI Chat) | Requires a chat template per |
| `/v1/completions` | POST | Text completions | Works for base models without a chat template |
| `/v1/embeddings` | POST | Text embeddings | Gated on `--enable-embeddings` Frontend arg |
| `/v1/realtime` | POST/WebSocket | Realtime streaming | Gated on `--enable-realtime`; PR #9205 (main) — verify the flag is on the target release |
| `/metrics` | GET | Prometheus metrics | Always on; scraped by the observability stack |
| `/health` | GET | Liveness probe | Used by the operator and any load balancer |
| `/ready` | GET | Readiness probe | Returns 200 only after the Frontend has at least one registered model |

The default port is 8000 for the OpenAI surface and 8001 for metrics
(verify against the DGD's `services.Frontend.extraPodSpec.mainContainer.ports`).

---

## Frontend Args

Set under `spec.services.Frontend.extraPodSpec.mainContainer.args` in
the DGD. Authoritative help: run the Frontend image with `--help`:

```bash
kubectl exec <frontend-pod> -- python3 -m dynamo.frontend --help
```

Common args:

| Arg | Purpose |
|---|---|
| `--port 8000` | Override the default port |
| `--host 0.0.0.0` | Override the default bind address |
| `--router kv-aware` | Delegate routing to the KV-aware router pod (requires Router service in the DGD) |
| `--router round-robin` | Simple round-robin across workers (default when no router is in the graph) |
| `--enable-embeddings` | Expose `/v1/embeddings` (requires an embeddings-capable worker) |
| `--enable-realtime` | Expose `/v1/realtime` (release-gated; verify against) |
| `--kserve-grpc` | Expose KServe gRPC ModelReady alongside the REST surface (per-adjacent known issue NVBug 6174719) |
| `--metrics-port 8001` | Move metrics off the default |

---

## Request Shape

### `/v1/chat/completions`

```json
{
  "model": "qwen3-06b",
  "messages": [{"role": "user", "content": "..."}],
  "max_tokens": 256,
  "temperature": 0.7,
  "stream": true
}
```

`stream: true` returns SSE chunks; the standard OpenAI `data: {...}\n\n`
protocol applies.

### `/v1/completions`

```json
{
  "model": "qwen3-06b-base",
  "prompt": "Once upon a time",
  "max_tokens": 256,
  "stream": false
}
```

Use this endpoint when the model lacks a chat template (per).

### `/v1/embeddings`

```json
{
  "model": "text-embedding-3-small",
  "input": ["hello world", "another doc"]
}
```

Requires `--enable-embeddings` on the Frontend AND an embeddings-capable
worker in the DGD.

---

## Response Shape

Matches OpenAI's `chat.completions.chunk` / `chat.completions` /
`completion` / `embedding` response objects. Notable Dynamo additions:

- `cache_hit_rate` (when KV router is in the path) — included in the
  Frontend's `/metrics` and in some AIPerf outputs (per
  [dynamo-benchmark](../../dynamo-benchmark/references/aiperf-invocation.md)).
- Model labels in `/metrics` use the model `id` as registered;
  capitalization is preserved as of RC5 cherry-pick #9775 / DYN-3076
  (the previous behavior lowercased model labels, causing metric
  mismatch with the KV-router).

---

## Authentication

The Dynamo Frontend does not implement auth natively. Two patterns:

| Pattern | Where auth lives |
|---|---|
| Gateway-level | GAIE / kgateway / Istio terminates auth before the Frontend; Frontend trusts the gateway |
| Sidecar | An auth proxy sidecar in the Frontend pod handles auth; the Frontend listens on localhost-only |

See [references/gateway-integration.md](gateway-integration.md) for
the per-gateway auth recipes.

---

## Streaming

Both `/v1/chat/completions` and `/v1/completions` support
`stream: true`. The Frontend translates the worker's token stream
into OpenAI SSE format. Header `Content-Type: text/event-stream` is
set automatically.

For client libraries: the OpenAI Python SDK works against the
Frontend directly with `base_url=http://<frontend>/v1` (no client-
side changes required).
