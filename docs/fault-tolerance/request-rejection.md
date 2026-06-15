---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Request Rejection
---

This document describes how Dynamo implements request rejection to prevent system overload and maintain service stability under high load conditions.

## Overview

Request rejection (also known as load shedding) is a fault tolerance mechanism that proactively rejects new requests when workers are overloaded. This prevents:

- Cascading failures from resource exhaustion
- Degraded latency for all requests
- Out-of-memory conditions on GPU workers

When all workers exceed their configured busy thresholds, new requests receive an HTTP 503 (Service Unavailable) response, signaling clients to retry later.

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

## Configuration

### Frontend Arguments

Configure busy thresholds when starting the frontend. `--admission-control token-capacity` is required to activate the thresholds; the default (`none`) leaves them disabled.

```bash
python -m dynamo.frontend \
    --admission-control token-capacity \
    --active-decode-blocks-threshold 0.85 \
    --active-prefill-tokens-threshold 10000
```

| Argument | Type | Description |
|----------|------|-------------|
| `--active-decode-blocks-threshold` | float (0.0-1.0) | KV cache block utilization threshold |
| `--active-prefill-tokens-threshold` | int | Prefill token count threshold |
| `--active-prefill-tokens-threshold-frac` | float | Prefill token threshold as a fraction of `max_num_batched_tokens` |
| `--admission-control` | `token-capacity` \| `none` | Admission control mode. `token-capacity` applies the busy thresholds above; `none` (the default) clears them while leaving router queueing controlled by `--router-queue-threshold`. To enable busy-worker admission, you must pass `--admission-control token-capacity` |

### Dynamic Configuration via API

Thresholds can be adjusted at runtime via the `/busy_threshold` endpoint:

#### Set Thresholds

```bash
curl -X POST http://localhost:8000/busy_threshold \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "active_decode_blocks_threshold": 0.85,
    "active_prefill_tokens_threshold": 10000
  }'
```

#### Get Current Thresholds

```bash
curl http://localhost:8000/busy_threshold
```

Response:
```json
{
  "thresholds": [
    {
      "model": "Qwen/Qwen3-0.6B",
      "active_decode_blocks_threshold": 0.85,
      "active_prefill_tokens_threshold": 10000
    }
  ]
}
```

## Busy Detection Logic

Workers are marked as "busy" based on a dual-threshold system. A worker is considered busy when **either** threshold is exceeded.

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

### Client Retry Strategy

Clients should implement exponential backoff when receiving 503 responses:

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

## Monitoring

### Prometheus Metrics

Track rejection behavior with these metrics:

- `dynamo_frontend_model_rejection_total`: Counter tracking the total number of requests rejected due to resource exhaustion
  - Labels:
    - `model`: The model name being served
    - `endpoint`: The API endpoint that received the request (e.g., `chat_completions`, `completions`, `embeddings`)
  - This metric is incremented when the router returns a `ResourceExhausted` error because all workers are busy. The rejected request is surfaced to the client as an HTTP 503 response.

**Example metrics output:**
```text
dynamo_frontend_model_rejection_total{endpoint="chat_completions",model="Qwen/Qwen3-0.6B"} 32
dynamo_frontend_model_rejection_total{endpoint="completions",model="Qwen/Qwen3-0.6B"} 5
```

**Endpoint:** Available on the frontend HTTP service at `/metrics`.

## Tuning Thresholds

### Conservative Settings (Latency-Focused)

For applications prioritizing low latency:

```bash
--active-decode-blocks-threshold 0.70
--active-prefill-tokens-threshold 5000
```

- Rejects earlier, before workers become fully loaded
- Maintains lower queue depths
- Better tail latencies

### Aggressive Settings (Throughput-Focused)

For applications prioritizing throughput:

```bash
--active-decode-blocks-threshold 0.95
--active-prefill-tokens-threshold 20000
```

- Allows higher worker utilization
- May increase latency variability
- Better overall throughput

### Disabled (No Rejection)

To disable request rejection entirely:

```bash
# Simply don't set the threshold arguments
python -m dynamo.frontend
```

Without thresholds configured, all requests are accepted regardless of worker load.

## Best Practices

### 1. Start Conservative, Then Tune

Begin with conservative thresholds and increase based on observed behavior:

```bash
# Start here
--active-decode-blocks-threshold 0.75

# Increase if rejection rate is too high
--active-decode-blocks-threshold 0.85
```

### 2. Monitor Before Enabling

Observe worker load patterns before setting thresholds:

```bash
# Watch KV cache utilization
watch -n 1 'curl -s localhost:8000/metrics | grep kv_blocks'
```

### 3. Use Both Thresholds for Disaggregated Serving

In disaggregated deployments:
- Use `active_prefill_tokens_threshold` for prefill workers
- Use `active_decode_blocks_threshold` for decode workers

### 4. Coordinate with Autoscaling

If using Kubernetes HPA, ensure rejection thresholds trigger before autoscaling:

```yaml
# HPA triggers at 70% utilization
# Rejection at 85% provides buffer
--active-decode-blocks-threshold 0.85
```

## Worker-Side Request Admission

In addition to the frontend's metric-driven busy detection above, a worker can
enforce a hard concurrency cap directly at its request-plane ingress. This is
disabled by default — when neither knob is set, the worker behaves exactly as
before (a large pool plus a large overflow queue, no rejection).

### Knobs

| Flag | Env var | Meaning |
| --- | --- | --- |
| `--engine-request-limit N` | `DYN_ENGINE_REQUEST_LIMIT` | Max requests handled **concurrently by the engine** (the worker-pool semaphore size). Setting this enables worker-side rejection. |
| _(env-only)_ | `DYN_DYNAMO_REQUEST_QUEUE_LIMIT` | Max requests **waiting in Dynamo** (not yet in the engine) — the overflow queue size. Not a CLI knob; a small fixed burst defaulting to **16** (hard cap `N + 16`). Only takes effect when the engine limit is set. Advanced override only; must be **≥ 2**. |

When `--engine-request-limit` is set, the worker accepts a request directly into
the engine while a slot is free; once all `N` engine slots are busy, further
requests go into the small overflow queue of size `Q`; when the engine **and**
the queue are both full the worker rejects the request with
`Server overloaded: worker at capacity`. The frontend maps this rejection to
`ResourceExhausted` → **HTTP 503**, and temporarily marks the worker overloaded
so it is skipped on the next routing decision (cleared automatically on the next
metric recompute). The effective hard cap is **N + Q** in-flight requests per
worker. The overflow channel is sized to `Q-1` because the single dispatcher
holds one request in transit between the queue and the engine; this makes the
cap exact for **Q ≥ 2** (at `Q = 1` the channel floors at 1, so the queued
peak is 2 — hence the `Q ≥ 2` requirement).

### Metrics

| Metric | Type | Meaning |
| --- | --- | --- |
| `dynamo_rejection_request_total` | counter | Cumulative requests rejected because the worker was at capacity (engine in-flight limit and Dynamo queue both full). |
| `dynamo_engine_request` | gauge | Current requests being handled by the engine. |
| `dynamo_request_queue` | gauge | Current requests queued in Dynamo, not yet in the engine. |

## Related Documentation

- [Health Checks](../observability/health-checks.md) - Worker health monitoring
- [Metrics](../observability/metrics.md) - Available Prometheus metrics
- [Request Migration](request-migration.md) - Handling failed requests
