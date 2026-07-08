---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Request Rejection Architecture
---

This document describes the internals of how Dynamo implements request rejection (load shedding) to prevent system overload and maintain service stability under high load.

This is an architecture reference. For how to enable, tune, and monitor request rejection, see the [Request Rejection](../fault-tolerance/request-rejection.md) use-case guide. For the metrics rejection emits, see [Cancellation and rejection](../reference/observability/metrics-catalog.mdx#cancellation-and-rejection) in the Metrics Catalog.

## Overview

Request rejection proactively rejects new requests when workers are overloaded. This prevents cascading failures from resource exhaustion, degraded latency for all requests, and out-of-memory conditions on GPU workers. When all workers exceed their configured busy thresholds, new requests receive an HTTP 503 (Service Unavailable) response, signaling clients to retry later.

## Architecture

```
                                    ┌─────────────────┐
                                    │  Worker Monitor │
                                    │  (Background)   │
                                    └────────┬────────┘
                                             │ Updates busy list
                                             ▼
┌──────────┐    ┌──────────┐    ┌─────────────────────┐    ┌──────────┐
│  Client  │───▶│ Frontend │───▶│    Push Router      │───▶│  Worker  │
└──────────┘    └──────────┘    │ (checks busy list)  │    └──────────┘
                                └─────────────────────┘
                                         │
                                         │ If all workers busy
                                         ▼
                                ┌─────────────────────┐
                                │   HTTP 503 Error    │
                                │ "All workers busy"  │
                                └─────────────────────┘
```

## Busy Detection Logic

Workers are marked as "busy" based on a dual-threshold system. A worker is considered busy when **either** threshold is exceeded. The thresholds themselves are configured on the frontend — see [Enable admission control on the Frontend](../fault-tolerance/request-rejection.md#enable-admission-control-on-the-frontend) in the use-case guide.

### KV Cache Block Threshold

Monitors the percentage of KV cache blocks in use:

```
busy = active_decode_blocks / kv_total_blocks > threshold
```

Example: With `active_decode_blocks_threshold=0.85`, a worker using 87% of its KV cache blocks is marked busy.

### Prefill Token Threshold

Monitors the number of tokens currently being prefilled:

```
busy = active_prefill_tokens > threshold
```

Example: With `active_prefill_tokens_threshold=10000`, a worker prefilling 12,000 tokens is marked busy.

### Data-Parallel Rank Aggregation

For workers with multiple data-parallel ranks (tensor parallelism), the worker is only marked busy if **ALL** ranks are busy:

```python
def is_busy(worker):
    return all(rank.is_busy() for rank in worker.dp_ranks)
```

This prevents false positives when only some ranks are temporarily loaded.

## Worker Load Monitoring

The `KvWorkerMonitor` runs as a background task that:

1. Subscribes to KV cache metrics events from workers
2. Maintains load state for each worker instance
3. Recalculates busy instances when metrics change
4. Updates the router with the current busy list

### Metrics Collected

Workers publish these metrics for monitoring:

| Metric | Description |
|--------|-------------|
| `active_decode_blocks` | Number of KV cache blocks currently in use |
| `kv_total_blocks` | Total KV cache blocks available |
| `active_prefill_tokens` | Number of tokens currently being prefilled |

## Rejection Behavior

### Request Flow

1. Request arrives at frontend
2. Push router checks if busy threshold is configured
3. If configured, router retrieves list of free (non-busy) instances
4. If no free instances exist (but instances are registered):
   - Request is rejected with `PipelineError::ServiceOverloaded`
   - HTTP 503 response is returned to client

### Error Response

When requests are rejected, clients receive:

```http
HTTP/1.1 503 Service Unavailable
Content-Type: application/json

{
  "message": "Service temporarily unavailable: All workers are busy, please retry later",
  "type": "service_unavailable",
  "code": 503
}
```

## Worker-Side Request Admission

In addition to the frontend's metric-driven busy detection above, a worker can enforce a hard concurrency cap directly at its request-plane ingress. This is disabled by default — when neither knob is set, the worker behaves exactly as before (a large pool plus a large overflow queue, no rejection). For the configuration knobs (`--engine-request-limit`, `DYN_DYNAMO_REQUEST_QUEUE_LIMIT`), see [Add a worker-side hard cap](../fault-tolerance/request-rejection.md#add-a-worker-side-hard-cap) in the use-case guide.

### Admission Mechanism

When `--engine-request-limit` is set, the worker accepts a request directly into the engine while a slot is free; once all `N` engine slots are busy, further requests go into the small overflow queue of size `Q`; when the engine **and** the queue are both full the worker rejects the request with `Server overloaded: worker at capacity`. The frontend maps this rejection to `ResourceExhausted` → **HTTP 503**, and temporarily marks the worker overloaded so it is skipped on the next routing decision (cleared automatically on the next metric recompute). The effective hard cap is **N + Q** in-flight requests per worker.

### Overflow Channel Sizing

The overflow channel is sized to `Q-1` because the single dispatcher holds one request in transit between the queue and the engine; this makes the cap exact for **Q ≥ 2** (at `Q = 1` the channel floors at 1, so the queued peak is 2 — hence the `Q ≥ 2` requirement). `DYN_DYNAMO_REQUEST_QUEUE_LIMIT` defaults to **16** (hard cap `N + 16`) and only takes effect when the engine limit is set.

## Related Documentation

- [Request Rejection](../fault-tolerance/request-rejection.md) - How to enable, tune, and monitor request rejection (use-case guide)
- [Request Migration Architecture](request-migration.md) - Recovering in-flight requests after worker failure
- [Metrics Catalog](../reference/observability/metrics-catalog.mdx#cancellation-and-rejection) - Rejection and admission metrics
- [Health Checks](../observability/health-checks.md) - Worker health monitoring
