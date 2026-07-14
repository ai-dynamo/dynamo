---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Request Rejection
subtitle: Configure independent worker-load thresholds that shed traffic with HTTP 529
---

NVIDIA Dynamo rejects new requests when configured worker-load thresholds show that every eligible
worker is overloaded.

## Overview

Request rejection (also known as load shedding) is a fault tolerance mechanism that proactively rejects new requests when workers are overloaded. This prevents:

- Cascading failures from resource exhaustion
- Degraded latency for all requests
- Out-of-memory conditions on GPU workers

When every eligible worker exceeds at least one configured busy threshold, new requests receive an
HTTP 529 response, signaling clients to retry later.

## Architecture

```text
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
                                │   HTTP 529 Error    │
                                │ "All workers busy"  │
                                └─────────────────────┘
```

## Configuration

### Frontend Arguments

Each busy threshold is disabled by default and becomes active when you set a numeric value. Setting
one threshold does not enable either of the other checks. For decode-block rejection, start the
frontend in KV router mode so the worker-load metrics path is active.

```bash
python -m dynamo.frontend \
    --router-mode kv \
    --active-decode-blocks-threshold 0.85 \
    --router-track-output-blocks \
    --active-prefill-tokens-threshold 10000
```

| Argument | Type | Description |
|----------|------|-------------|
| `--router-mode kv` | enum | Required for decode-block rejection. Initializes the KV router worker-load plumbing that receives `active_decode_blocks` updates. Without KV mode, `--active-decode-blocks-threshold` will not produce 529s based on decode-block load. |
| `--active-decode-blocks-threshold` | float (0.0-1.0) | KV cache block utilization threshold. Unset by default |
| `--router-track-output-blocks` | bool | Include generated output tokens in the router's active block count. Enable this for long-output workloads; otherwise only prompt/input blocks are counted and a long generation can fill KV without crossing the threshold seen by the router. |
| `--active-prefill-tokens-threshold` | int | Prefill token count threshold. Unset by default |
| `--active-prefill-tokens-threshold-frac` | float | Prefill token threshold as a fraction of `max_num_batched_tokens`. Unset by default |

### Dynamic Configuration via API

Set thresholds for a discovered model at runtime through the `/busy_threshold` endpoint. A numeric
value enables its corresponding rejection check. The update changes the stored threshold
configuration only; the router re-evaluates which workers are busy when the next worker load or
runtime-config update arrives, so a new threshold may take a short time to affect rejection
decisions. The endpoint does not enable `--router-mode kv` or `--router-track-output-blocks`; set
those options when the frontend starts if your decode-block monitoring path requires them.

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
      "active_prefill_tokens_threshold": 10000,
      "active_prefill_tokens_threshold_frac": null
    }
  ]
}
```

### Migrate from `--admission-control`

The `--admission-control` flag and the `DYN_ADMISSION_CONTROL` environment variable no longer have
any effect. Both are still accepted so that existing launch commands keep starting, but they are
ignored with a startup warning. Configure only the rejection thresholds that you want Dynamo to
enforce:

| Previous configuration | Replacement |
| --- | --- |
| `--admission-control none` or `DYN_ADMISSION_CONTROL=none` | Remove the setting. With no thresholds configured, busy-worker rejection is disabled. |
| `--admission-control token-capacity` with explicit threshold flags | Remove `--admission-control` and keep the explicit threshold flags. |
| `DYN_ADMISSION_CONTROL=token-capacity` with explicit threshold environment variables | Remove `DYN_ADMISSION_CONTROL` and keep the explicit threshold environment variables. |
| `--admission-control token-capacity` without thresholds | Set each threshold you want to enable. The former preset values were `1.0`, `10000000`, and `64.0`, respectively. |

Earlier behavior filled unspecified thresholds with preset values after any one threshold was set.
That implicit coupling has been removed: an unset threshold stays disabled. The independent
`--router-queue-threshold` option also remains disabled until set to a numeric value.

## Busy Detection Logic

Each configured threshold is an independent busy check. A worker is considered busy when any
configured check is exceeded; unset checks are skipped.

### Decode-Block Rejection Requirements

Decode-block based rejection (`active_decode_blocks_threshold`) only works when all of these conditions are true:

1. The frontend is started with `--router-mode kv`.
2. A decode-block threshold is configured (`--active-decode-blocks-threshold` or `/busy_threshold` API).
3. The frontend `KvWorkerMonitor` is receiving worker load events (`ActiveLoad`).
4. Workers are publishing `active_decode_blocks`.
5. Worker runtime config provides `kv_total_blocks` so utilization ratio can be computed.
6. For long-output workloads, `--router-track-output-blocks` is enabled so generated tokens add output blocks to the active block count.

If any prerequisite is missing, decode-block busy detection is effectively disabled for those workers. The most common production symptom is that `--active-decode-blocks-threshold` appears to be accepted but no HTTP 529 is returned when KV fills up.

Examples of missing prerequisites:

- Frontend was launched without `--router-mode kv`; the event subscriber used for worker load metrics is not initialized, so `active_decode_blocks` updates are not consumed and the threshold is ignored.
- Frontend cannot receive events because worker-load subscription is unavailable (for example, event transport not reachable or misconfigured).
- Workers are running in a mode/path that does not publish `active_decode_blocks` (for example, custom integrations without worker metrics publishing).
- Output-heavy traffic is served without `--router-track-output-blocks`; only prompt/input blocks are reflected in the router's active block accounting, so generated output blocks can exhaust KV before triggering decode-block rejection.

### Important: Different from `router_track_active_blocks`

`active_decode_blocks_threshold` and `router_track_active_blocks` are related to load, but they are not the same feature:

- `active_decode_blocks_threshold` drives busy/free worker classification and request rejection (HTTP 529 when all workers are busy).
- `router_track_active_blocks` controls KV router internal block bookkeeping used for routing decisions.

In disaggregated setups, prefill routing intentionally disables `router_track_active_blocks`; this does **not** disable decode-block rejection for decode workers.

### KV Cache Block Threshold

Monitors the percentage of KV cache blocks in use:

```text
busy = active_decode_blocks / kv_total_blocks > threshold
```

Example: With `active_decode_blocks_threshold=0.85`, a worker using 87% of its KV cache blocks is marked busy.

### Prefill Token Threshold

Monitors the number of tokens currently being prefilled:

```text
busy = active_prefill_tokens > threshold
```

Example: With `active_prefill_tokens_threshold=10000`, a worker prefilling 12,000 tokens is marked busy.

### Prefill Token Fraction Threshold

Compares active prefill tokens with a multiple of the worker's `max_num_batched_tokens` value:

```text
busy = active_prefill_tokens > threshold_frac * max_num_batched_tokens
```

The absolute and fractional prefill checks use OR logic when both are configured.

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
   - HTTP 529 response is returned to client

### Error Response

When requests are rejected, clients receive:

```http
HTTP/1.1 529
Content-Type: application/json

{
  "message": "Service temporarily overloaded",
  "type": "Overloaded",
  "code": 529
}
```

#### Configuring the rejection status code

The status code returned for overload rejections is configurable via the
`DYN_HTTP_OVERLOAD_STATUS_CODE` environment variable on the frontend. It
defaults to `529` ("Site is overloaded"). Operators whose proxies or clients
only understand `503` Service Unavailable retry semantics can set
`DYN_HTTP_OVERLOAD_STATUS_CODE=503`. Any valid HTTP status code is accepted; an
unparseable or out-of-range value falls back to `529`.

### Client Retry Strategy

Clients should implement exponential backoff when receiving 529 responses:

```python
import time
import random

def send_with_retry(request, max_retries=5):
    for attempt in range(max_retries):
        response = send_request(request)
        if response.status_code != 529:
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
  - This metric is incremented when the router returns a `ResourceExhausted` error because all workers are busy. The rejected request is surfaced to the client as an HTTP 529 response.

**Example metrics output:**
```text
dynamo_frontend_model_rejection_total{endpoint="chat_completions",model="Qwen/Qwen3-0.6B"} 32
dynamo_frontend_model_rejection_total{endpoint="completions",model="Qwen/Qwen3-0.6B"} 5
```

For decode-block rejection debugging, also inspect:

| Metric | Type | Description |
|--------|------|-------------|
| `dynamo_frontend_worker_active_decode_blocks` | Gauge | Latest active decode blocks per worker and DP rank |
| `dynamo_frontend_worker_active_prefill_tokens` | Gauge | Latest active prefill tokens per worker and DP rank |

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

### 3. Use Workload-Specific Thresholds

In disaggregated deployments:
- Use `active_prefill_tokens_threshold` for prefill workers
- Use `active_prefill_tokens_threshold_frac` instead when the limit should scale with
  `max_num_batched_tokens`
- Use `active_decode_blocks_threshold` for decode workers

### 4. Coordinate with Autoscaling

If using Kubernetes HPA, ensure rejection thresholds trigger before autoscaling:

```yaml
# HPA triggers at 70% utilization
# Rejection at 85% provides buffer
--active-decode-blocks-threshold 0.85
```

## Troubleshooting

### Decode-block rejection not triggering

1. Confirm threshold is actually set:
```bash
curl -s http://localhost:8000/busy_threshold
```
2. Confirm the frontend was started with `--router-mode kv`.
3. For long-output workloads, confirm the frontend was started with `--router-track-output-blocks`.
4. Verify frontend is receiving worker load updates:
```bash
curl -s http://localhost:8000/metrics | grep dynamo_frontend_worker_active_decode_blocks
```
5. Check frontend logs for worker-monitor subscription issues (for example, warnings that KV metrics subscriber is unavailable).
6. Verify worker `kv_total_blocks` is present (runtime config / worker metrics), for example:
```bash
curl -s http://<worker-system-port>/metrics | grep dynamo_component_total_blocks
```
7. Verify event transport configuration between frontend and workers (`--event-plane`, NATS/ZMQ connectivity).

### Common confusion: `router_track_active_blocks`

If `active_decode_blocks_threshold` is configured but you suspect `router_track_active_blocks` is the blocker, treat that as a separate routing knob. Busy rejection depends on worker load events and threshold configuration, not on the router's internal active-block tracking flag.

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
`ResourceExhausted` → **HTTP 529**, and temporarily marks the worker overloaded
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
