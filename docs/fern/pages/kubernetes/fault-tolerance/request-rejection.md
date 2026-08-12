---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Request Rejection
subtitle: Configure independent load thresholds that shed traffic with HTTP 529 before worker latency or memory use becomes unsafe.
---

Request rejection (load shedding) rejects new requests when every eligible worker is overloaded,
rather than accepting work that could exhaust GPU memory or degrade latency for in-flight requests.
Dynamo returns **HTTP 529** for overload by default so clients can distinguish an available but busy
service from **HTTP 503**, which indicates that the service is unavailable.

Rejection is **off by default**. Each threshold is independently opt-in: setting one numeric threshold
enables only that check. You do not need `--admission-control`; that compatibility flag is ignored.

> **How it works:** The busy-detection formulas, data-parallel rank aggregation, worker-load event
> processing, and worker-side overflow queue are documented in
> [Request Rejection Architecture](../../developer-guide/knowledge-base/concepts/fault-tolerance/request-rejection-architecture.md).

<Steps toc={true} tocDepth={2}>

<Step title="Choose the load signals">

Use the signals that match the workers you want to protect:

- Set `--active-decode-blocks-threshold` for decode workers. The value is the fraction of available
  KV cache blocks in use, from `0.0` through `1.0`.
- Set `--active-prefill-tokens-threshold` for an absolute prefill-token limit.
- Set `--active-prefill-tokens-threshold-frac` when the prefill limit should scale with the worker's
  `max_num_batched_tokens` value.

The checks use OR logic. If you configure more than one threshold, exceeding any configured threshold
marks that data-parallel rank as busy. A worker is rejected only after all of its ranks are busy, and a
request is shed only when every eligible worker is busy.

Start conservatively, observe the rejection rate and latency, and then raise the limits. For example,
a decode-block threshold of `0.75` sheds earlier than `0.90`.

</Step>

<Step title="Enable rejection on the Frontend">

Configure thresholds on the **Frontend** component. Decode-block rejection also requires KV router
mode because that mode initializes the worker-load metrics path. For long-output workloads, enable
output-block tracking so generated tokens contribute to the observed KV cache load.

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
            - --router-mode
            - kv
            - --active-decode-blocks-threshold
            - "0.85"
            - --router-track-output-blocks
            - --active-prefill-tokens-threshold
            - "10000"
```

`--router-track-output-blocks` is especially important for long generations. Without it, a workload
can fill KV cache with generated tokens without crossing the load value observed by the router.

You can configure just one signal. For example, a prefill-only deployment can omit the decode-block
threshold and KV-specific options. To disable rejection, remove all three threshold settings or set
their environment-variable values to `None`.

See [Frontend Configuration](../../reference/components/frontend-configuration.mdx#fault-tolerance)
for the complete flag, environment-variable, and validation reference.

</Step>

<Step title="Adjust thresholds at runtime">

Optional. Use the Frontend admin API to change thresholds for a discovered model without redeploying.
The API is available by default. Set `DYN_DISABLE_FRONTEND_ADMIN_API` to a truthy value to turn it
off, which unregisters both `busy_threshold` routes so they return 404.

```bash
kubectl port-forward svc/<deployment-name>-frontend 8000:8000 -n ${NAMESPACE}
```

Set one or more thresholds:

```bash
curl -X POST http://localhost:8000/busy_threshold \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "active_decode_blocks_threshold": 0.85,
    "active_prefill_tokens_threshold": 10000
  }'
```

Read the stored thresholds:

```bash
curl http://localhost:8000/busy_threshold
```

A numeric value enables only its corresponding check. The router applies the new configuration when
the next worker-load or runtime-configuration update arrives, so the change might not affect a routing
decision immediately. This endpoint does not enable `--router-mode kv` or
`--router-track-output-blocks`; set those when the Frontend starts.

</Step>

<Step title="Add a worker-side hard cap">

Optional. A worker can independently cap concurrent engine work and queue only a small burst. Set
`--engine-request-limit N` (or `DYN_ENGINE_REQUEST_LIMIT`) on the **worker** component:

```yaml
- name: VllmWorker
  type: worker
  podTemplate:
    spec:
      containers:
        - name: main
          args:
            - --engine-request-limit
            - "32"
```

When all `N` engine slots and the overflow queue are full, the worker rejects the request and the
Frontend returns HTTP 529 when migration is disabled or its retry attempts do not find capacity.
With a positive `--migration-limit`, an unpinned request using in-process KV routing retries on another
eligible worker first. Split or standalone routing uses global overload and fault state, so its
retry is best-effort and can select the same worker again.
`DYN_DYNAMO_REQUEST_QUEUE_LIMIT` controls the advanced overflow-queue size, defaults to `16`, must be
at least `2`, and has an effect only when the engine limit is set. The effective cap is `N + Q`
in-flight requests per worker.

See [Runtime Configuration](../../reference/components/runtime-configuration.mdx#operations) for the exact
fields and [Worker-Side Request Admission](../../developer-guide/knowledge-base/concepts/fault-tolerance/request-rejection-architecture.md#worker-side-request-admission)
for the queue implementation.

</Step>

<Step title="Verify rejection">

Inspect the configured thresholds and worker-load metrics:

```bash
curl -s http://localhost:8000/busy_threshold
curl -s http://localhost:8000/metrics \
  | grep -E 'dynamo_frontend_worker_active_(decode_blocks|prefill_tokens)'
```

Generate enough load to exceed the configured threshold, then confirm:

- The client receives HTTP 529.
- `dynamo_frontend_model_rejection_total` increases for the affected `model` and `endpoint`.
- Worker-side hard-cap tests increase `dynamo_rejection_request_total` when both the engine and queue
  are full.

For all metric fields and labels, see
[Cancellation and Rejection](../../reference/observability/metrics-catalog.mdx#cancellation-and-rejection).

</Step>

</Steps>

## Troubleshoot Decode-Block Rejection

If decode-block load does not produce HTTP 529 responses:

1. Confirm that `GET /busy_threshold` shows a numeric `active_decode_blocks_threshold` for the model.
2. Confirm that the Frontend started with `--router-mode kv`.
3. For long-output workloads, confirm that the Frontend started with `--router-track-output-blocks`.
4. Check that the Frontend receives worker updates:

   ```bash
   curl -s http://localhost:8000/metrics \
     | grep dynamo_frontend_worker_active_decode_blocks
   ```

5. Check Frontend logs for worker-monitor subscription warnings.
6. Confirm that each worker publishes its total KV block count:

   ```bash
   curl -s http://<worker-system-port>/metrics \
     | grep dynamo_component_total_blocks
   ```

7. Verify the event-plane configuration and connectivity between the Frontend and workers.

`--router-track-active-blocks` is a separate routing option. Busy rejection depends on configured
thresholds and worker-load events; it does not require that internal router tracking flag.

## Configure Client Retries

Retry HTTP 529 responses with exponential backoff and jitter. Do not retry every HTTP 503 as if it
were overload; 503 indicates that Dynamo currently has no available service path.

```python
import random
import time


def send_with_retry(request, max_retries=5):
    for attempt in range(max_retries):
        response = send_request(request)
        if response.status_code != 529:
            return response
        wait_time = min(60, (2**attempt) + random.uniform(0, 1))
        time.sleep(wait_time)
    raise RuntimeError("maximum retries exceeded")
```

If an existing client only understands 503 retry semantics, set
`DYN_HTTP_OVERLOAD_STATUS_CODE=503` on the Frontend. The variable accepts values from 200 through
999. Informational values from 100 through 199, invalid values, and out-of-range values fall back to
529. The value is read and cached on first use.

## Related Documentation

- [Request Rejection Architecture](../../developer-guide/knowledge-base/concepts/fault-tolerance/request-rejection-architecture.md) - Busy detection, event flow, and admission internals
- [Frontend Configuration](../../reference/components/frontend-configuration.mdx#fault-tolerance) - Thresholds, overload status, and admin API
- [Runtime Configuration](../../reference/components/runtime-configuration.mdx#operations) - Worker-side engine and queue limits
- [Metrics Catalog](../../reference/observability/metrics-catalog.mdx#cancellation-and-rejection) - Rejection and admission metrics
- [Request Migration](request-migration.md) - Recovering in-flight requests after worker failure
