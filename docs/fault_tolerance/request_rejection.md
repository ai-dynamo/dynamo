<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Request Rejection (Load Shedding)

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

Configure busy thresholds when starting the frontend:

```bash
python -m dynamo.frontend \
    --active-decode-blocks-threshold 0.85 \
    --active-prefill-tokens-threshold 10000
```

| Argument | Type | Description |
|----------|------|-------------|
| `--active-decode-blocks-threshold` | float (0.0-1.0) | KV cache block utilization threshold |
| `--active-prefill-tokens-threshold` | int | Prefill token count threshold |

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

| Metric | Type | Description |
|--------|------|-------------|
| `dynamo_tasks_rejected_total` | Counter | Total number of rejected tasks |
| `dynamo_queued_requests` | Gauge | Requests waiting in HTTP queue |

### Example Prometheus Queries

```promql
# Rejection rate over 5 minutes
rate(dynamo_tasks_rejected_total[5m])

# Percentage of requests rejected
sum(rate(dynamo_tasks_rejected_total[5m])) /
sum(rate(dynamo_tasks_issued_total[5m])) * 100
```

### Grafana Alerting

Example alert for high rejection rate:

```yaml
alert: HighRequestRejectionRate
expr: |
  sum(rate(dynamo_tasks_rejected_total[5m])) /
  sum(rate(dynamo_tasks_issued_total[5m])) > 0.1
for: 5m
labels:
  severity: warning
annotations:
  summary: "High request rejection rate"
  description: "More than 10% of requests are being rejected"
```

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

## Related Documentation

- [Health Checks](../observability/health-checks.md) - Worker health monitoring
- [Metrics](../observability/metrics.md) - Available Prometheus metrics
- [Request Migration](request_migration.md) - Handling failed requests
