---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Priority Scheduling
subtitle: Request priority across the Dynamo router and backend engines
---

Priority scheduling lets a client mark one request as more important than
another. Dynamo exposes two related request fields:

- `nvext.agent_hints.priority` is a soft priority used by router policy scoring
  and supported backend engines.
- `nvext.agent_hints.strict_priority` is an unsigned router pending-queue tier.
  Higher tiers always precede lower tiers.

```json
{
    "model": "my-model",
    "messages": [
        { "role": "user", "content": "Summarize this incident." }
    ],
    "nvext": {
        "agent_hints": {
            "priority": 10,
            "strict_priority": 1
        }
    }
}
```

## Priority Layers

Priority can affect three different layers. They are configured separately.

| Layer | What It Controls | Required Configuration | Deep Details |
|-------|------------------|------------------------|--------------|
| Frontend API | The user-facing request schema and priority polarity. | Send `priority` for soft router and engine priority, or `strict_priority` for a router-only pending tier. | [NVIDIA Request Extensions](../components/frontend/nvext.md#agent-hints) |
| Router queue | Which waiting request is dispatched first when the router queue is non-empty. | KV routing plus `--router-queue-threshold` set to a value that actually causes queueing. | [`--router-queue-threshold`](../components/router/router-configuration.md#routing-behavior), [`--router-queue-policy`](../components/router/router-configuration.md#routing-behavior) |
| Backend engine | Which admitted request the engine schedules first. | Backend-specific priority scheduling flag, such as vLLM `--scheduling-policy priority` or SGLang `--enable-priority-scheduling`. | [vLLM priority scheduling](../backends/vllm/vllm-reference-guide.md#priority-scheduling), [SGLang priority scheduling](../backends/sglang/agents.md#priority-scheduling) |
| KV cache policy | Which cached blocks are retained or evicted first under memory pressure. | Backend-specific cache priority configuration, such as SGLang `--radix-eviction-policy priority`. | [SGLang priority-based KV cache eviction](../backends/sglang/agents.md#priority-based-kv-cache-eviction) |

These layers are additive. `strict_priority` does not propagate to backend
engine scheduling; use `priority` for that layer.

## Router Queue Priority

The router queue only matters when requests are held before dispatch. If a
request can be routed immediately, there is no pending queue to reorder and the
priority hint will not change TTFT at the router layer.

`--router-queue-threshold` controls when the router starts holding requests. A
request waits in the router queue while every eligible worker is above the
configured threshold. The queue drains when capacity is available, and
higher-priority requests are selected according to the queue key:

```text
(strict_priority, configured_policy_key)
```

The strict tier is compared first. FCFS, LCFS, or Weighted Shortest Processing
Time (WSPT) still computes the secondary key and orders requests within the same
tier.

The default policy is `fcfs`, which uses the priority value as a positive
arrival-time bump. Higher values move the request earlier in the queue. Negative
priority values are clamped to zero for router queueing, so a request cannot be
pushed behind normal first-come, first-served ordering by sending a negative
priority.

For the flag-level semantics, default value, and backend caveats, see
[Router Configuration and Tuning](../components/router/router-configuration.md#routing-behavior).

## Backend Engine Priority

The backend receives the same Dynamo semantic priority, but each engine has its
own native scheduling convention. Dynamo handles that conversion internally.

| Backend | Engine Scheduling Requirement | Dynamo Behavior |
|---------|-------------------------------|-----------------|
| vLLM | Start vLLM with `--scheduling-policy priority`. | Dynamo forwards the user priority with the polarity vLLM expects. |
| SGLang | Start SGLang with `--enable-priority-scheduling`. | Dynamo forwards higher Dynamo values as higher SGLang scheduling priority and rejects the inverted SGLang flag. |
| TensorRT-LLM | Per-request engine scheduling priority is not currently exposed through Dynamo. | Priority can still affect router queueing before dispatch. |

Do not negate `nvext.agent_hints.priority` in client code for vLLM. If a test
shows lower user values receiving better TTFT, first check whether the benchmark
harness or endpoint path inverted the value before it reached Dynamo.

## What Priority Does Not Do

Priority is not Kubernetes `PriorityClass`, GPU preemption, or a hard admission
control policy. It does not reserve capacity for high-priority requests.

Strict priority applies only to requests already parked in one scheduler queue.
It does not preempt admitted work, impose ordering across router replicas or
upstream queues, or guarantee backend engine execution order. An eligible new
arrival can still be admitted directly while other requests are pending.

Priority also does not show an effect unless there is contention at a layer that
uses it:

- Router priority needs a non-empty router queue.
- Engine priority needs backend priority scheduling enabled and engine-side
  queueing or preemption opportunities.
- Cache priority needs memory pressure and a priority-aware eviction policy.

## Verify Priority Is Working

Use a benchmark that can send different `nvext.agent_hints.priority` values on
individual requests. For AIPerf, use a version with per-request `extra` payload
support. Older AIPerf versions may only support global `--extra-inputs`, which
is not enough for mixed-priority tiers in the same run.

For router-priority validation:

- Use a fixed request count or burst-style test so every priority tier gets the
  same number of measured requests.
- Keep the model, input length, output length, streaming mode, and endpoint path
  identical across priority tiers.
- Run at enough load for requests to wait in the router queue. Watch
  [`dynamo_frontend_router_queue_pending_requests`](../observability/metrics.md#router-queue-metrics-dynamo_frontend_router_queue_)
  and confirm it is greater than zero during the measured window.
- Configure the backend priority flag separately if the test is meant to measure
  engine scheduling, not only router queue ordering.

Expected result: higher Dynamo priority values should receive better TTFT under
contention. If lower values win, first check whether the client, benchmark
harness, or gateway path negated the priority before it reached Dynamo.

## Troubleshooting

| Symptom | Checks |
|---------|--------|
| Priority has no visible effect. | Confirm requests actually enter the router queue, and confirm the backend priority flag is enabled if you expect engine-level scheduling. |
| Lower numeric values appear to win. | Do not negate `nvext.agent_hints.priority` for vLLM. Dynamo normalizes backend polarity internally. |
| Router queue never becomes non-empty. | Lower `--router-queue-threshold`, increase offered load, or check the SGLang `max_num_batched_tokens` caveat in [Router Configuration and Tuning](../components/router/router-configuration.md#tuning-guidelines). |
| Priority works through the frontend but not through a Kubernetes gateway path. | Confirm the gateway path preserves `nvext` and use Dynamo v1.2.0 or later. |
| AIPerf cannot assign a different priority per request. | Use an AIPerf build with per-request `extra` payload support. |

## Version Notes

| Capability | Availability |
|------------|--------------|
| Router priority queue and backend priority plumbing | Dynamo v1.0.0 and later. |
| Unified Dynamo API semantics where higher `nvext.agent_hints.priority` means higher priority | Dynamo v1.1.0 and later. |
| EPP / Inference Gateway forwarding fixes for priority hints | Dynamo v1.2.0 and later. |
| AIPerf per-request priority datasets | Required for mixed-priority benchmark runs; use an AIPerf release with per-request `extra` payload support. |

## Related Docs

- [Agent Hints](agent-hints.md)
- [NVIDIA Request Extensions](../components/frontend/nvext.md#agent-hints)
- [Router Configuration and Tuning](../components/router/router-configuration.md)
- [Router Queue Metrics](../observability/metrics.md#router-queue-metrics-dynamo_frontend_router_queue_)
- [vLLM Reference Guide](../backends/vllm/vllm-reference-guide.md#priority-scheduling)
- [SGLang for Agentic Workloads](../backends/sglang/agents.md)
