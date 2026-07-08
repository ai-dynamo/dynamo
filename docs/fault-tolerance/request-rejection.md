---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Request Rejection
subtitle: Shed load with HTTP 503 when workers are overloaded, instead of letting latency collapse for everyone.
---

Request rejection (load shedding) proactively rejects new requests when every worker is overloaded, rather than accepting work that would exhaust GPU memory or degrade latency for all in-flight requests. When all workers exceed their configured busy thresholds, new requests receive an **HTTP 503 (Service Unavailable)**, signaling clients to retry later.

Rejection is **off by default**. The steps below enable it on the Frontend, pick thresholds for your latency/throughput goals, optionally add a hard per-worker cap, and verify it is shedding load as expected.

> **How it works:** the busy-detection formulas, data-parallel rank aggregation, the `KvWorkerMonitor` background task, and the worker-side overflow-channel mechanics are documented in [Request Rejection Architecture](../design-docs/request-rejection.md).

<Steps toc={true} tocDepth={2}>

<Step title="Enable admission control on the Frontend">

Rejection activates only when you set `--admission-control token-capacity` **and** at least one busy threshold. The default (`--admission-control none`) leaves thresholds disabled. Configure them on the **Frontend** component as `args:`:

```yaml
  - name: Frontend
    type: frontend
    replicas: 1
    podTemplate:
      spec:
        containers:
        - name: main
          image: ${RUNTIME_IMAGE}
          command:
          - python3
          - -m
          - dynamo.frontend
          args:
          - --admission-control
          - token-capacity
          - --active-decode-blocks-threshold
          - "0.85"
          - --active-prefill-tokens-threshold
          - "10000"
```

A worker is marked **busy** when **either** threshold is exceeded; requests are rejected only when *all* workers are busy.

| Argument | Type | Description |
|----------|------|-------------|
| `--admission-control` | `token-capacity` \| `none` | Master switch. `token-capacity` applies the busy thresholds; `none` (default) disables them. Must be `token-capacity` to enable rejection. |
| `--active-decode-blocks-threshold` | float (0.0-1.0) | KV cache block utilization threshold |
| `--active-prefill-tokens-threshold` | int | Prefill token count threshold |
| `--active-prefill-tokens-threshold-frac` | float | Prefill token threshold as a fraction of `max_num_batched_tokens` |

Each flag has an environment-variable equivalent (`DYN_ADMISSION_CONTROL`, `DYN_ACTIVE_DECODE_BLOCKS_THRESHOLD`, `DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD`) you can set in the Frontend `env:` block instead — see the [Frontend Configuration Reference](../components/frontend/configuration.md).

</Step>

<Step title="Choose your thresholds">

Thresholds trade tail latency against throughput. Start conservative and loosen as you observe real load. Two starting points:

**Conservative (latency-focused)** — rejects earlier, keeps queue depths and tail latencies low:

```yaml
          - --active-decode-blocks-threshold
          - "0.70"
          - --active-prefill-tokens-threshold
          - "5000"
```

**Aggressive (throughput-focused)** — allows higher worker utilization at the cost of more latency variability:

```yaml
          - --active-decode-blocks-threshold
          - "0.95"
          - --active-prefill-tokens-threshold
          - "20000"
```

For **disaggregated** deployments, set `--active-prefill-tokens-threshold` on prefill workers and `--active-decode-blocks-threshold` on decode workers. To turn rejection back off entirely, set `--admission-control none` (or omit the threshold args).

</Step>

<Step title="Adjust thresholds at runtime">

Optional. You can change thresholds without redeploying via the Frontend's `/busy_threshold` endpoint. Port-forward first:

```bash
kubectl port-forward svc/<deployment-name>-frontend 8000:8000 -n ${NAMESPACE}
```

Set thresholds for a model:

```bash
curl -X POST http://localhost:8000/busy_threshold \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "active_decode_blocks_threshold": 0.85,
    "active_prefill_tokens_threshold": 10000
  }'
```

Read the current thresholds:

```bash
curl http://localhost:8000/busy_threshold
```

</Step>

<Step title="Add a worker-side hard cap">

Optional, and independent of the frontend thresholds above. A worker can also enforce a hard concurrency cap at its own request-plane ingress. Set `--engine-request-limit N` (or `DYN_ENGINE_REQUEST_LIMIT`) on the **worker** component:

| Flag | Env var | Meaning |
| --- | --- | --- |
| `--engine-request-limit N` | `DYN_ENGINE_REQUEST_LIMIT` | Max requests handled concurrently by the engine. Setting this enables worker-side rejection. |
| _(env-only)_ | `DYN_DYNAMO_REQUEST_QUEUE_LIMIT` | Overflow-queue size for requests waiting in Dynamo. Advanced override; defaults to **16**, must be **≥ 2**. Only applies when the engine limit is set. |

Once the engine's `N` slots and the small overflow queue are both full, the worker rejects with `Server overloaded: worker at capacity`, which the frontend surfaces as **HTTP 503**. The effective hard cap is **N + Q** in-flight requests per worker. For the queue-sizing details, see [Worker-Side Request Admission](../design-docs/request-rejection.md#worker-side-request-admission) in the architecture reference.

</Step>

<Step title="Verify and monitor">

Before enabling rejection in production, observe real worker load so your thresholds match reality. Scrape the Frontend `/metrics` endpoint:

```bash
kubectl port-forward svc/<deployment-name>-frontend 8000:8000 -n ${NAMESPACE}
watch -n 1 'curl -s localhost:8000/metrics | grep kv_blocks'
```

Once enabled, track rejections with the `dynamo_frontend_model_rejection_total` counter (labeled by `model` and `endpoint`):

```text
dynamo_frontend_model_rejection_total{endpoint="chat_completions",model="Qwen/Qwen3-0.6B"} 32
dynamo_frontend_model_rejection_total{endpoint="completions",model="Qwen/Qwen3-0.6B"} 5
```

A high rejection rate means thresholds are too tight, workers are under-provisioned, or autoscaling isn't keeping up. For full field definitions — including the worker-side admission counters — see [Cancellation and rejection](../reference/observability/metrics-catalog.mdx#cancellation-and-rejection) in the Metrics Catalog.

</Step>

</Steps>

## Best Practices

- **Start conservative, then tune.** Begin around `0.75` for decode blocks and raise it if the rejection rate is higher than you want; re-apply the DGD after each change.
- **Monitor before enabling.** Watch worker load patterns (previous step) before committing to thresholds.
- **Use both thresholds for disaggregated serving.** Prefill tokens for prefill workers, decode blocks for decode workers.
- **Coordinate with autoscaling.** If a Kubernetes HPA scales at 70% utilization, set rejection higher (for example `0.85`) so autoscaling gets a chance to add capacity before requests are shed.

## Handling 503 on the client

Clients should retry 503 responses with exponential backoff and jitter rather than hammering a busy cluster:

```python
import time
import random

def send_with_retry(request, max_retries=5):
    for attempt in range(max_retries):
        response = send_request(request)
        if response.status_code != 503:
            return response
        # Exponential backoff with jitter
        wait_time = min(60, (2 ** attempt) + random.uniform(0, 1))
        time.sleep(wait_time)
    raise Exception("Max retries exceeded")
```

The 503 body identifies the condition:

```json
{
  "message": "Service temporarily unavailable: All workers are busy, please retry later",
  "type": "service_unavailable",
  "code": 503
}
```

## Related Documentation

- [Request Rejection Architecture](../design-docs/request-rejection.md) - Busy-detection logic, worker load monitoring, and admission internals
- [Metrics Catalog](../reference/observability/metrics-catalog.mdx#cancellation-and-rejection) - Rejection and admission metrics
- [Request Migration](request-migration.md) - Recovering in-flight requests after a worker failure
- [Health Checks](../observability/health-checks.md) - Worker health monitoring
