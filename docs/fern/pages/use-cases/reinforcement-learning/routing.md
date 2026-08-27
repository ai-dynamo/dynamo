---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Route RL Rollouts
subtitle: Match routing and cache reuse to the shape of the rollout workload
---

Dynamo uses the same router for RL and other inference workloads. What changes is the objective: measure serving efficiency together with framework-owned sample freshness and acceptance. Dynamo routes requests; it does not decide whether a trajectory is on-policy or useful for training.

Use the [router configuration reference](../../developer-guide/knowledge-base/modular-components/router/configuration-and-tuning.md) for complete flag definitions and defaults.

## Choose a Starting Strategy

| Rollout shape | Start with | Watch for |
|---|---|---|
| Mostly independent prompts | Round-robin, then load-aware routing | Mixed output lengths can still create queue imbalance. |
| Many samples share a prompt or rubric | KV-aware routing | A cache-rich worker can become overloaded. |
| Sibling samples arrive before KV events | KV-aware routing with a short predicted-placement window | A long prediction window can preserve stale placement. |
| Multi-turn trajectories | KV-aware routing; add session affinity only when strict stickiness is required | Affinity can pin work to an overloaded or replaced worker. |
| Bursty mixed-length prompts | KV-aware or load-aware routing with queueing | Queueing can increase rollout tails or trigger framework timeouts. |

Start with one baseline and one mechanism justified by the workload. Do not enable KV routing, queueing, affinity, priority, offload, and custom policies in the same first experiment.

## Establish the Baseline

Use round-robin to measure the cost of simple distribution:

```bash
python -m dynamo.frontend --router-mode round-robin
```

Record the effective worker set, request errors, generated tokens, time to first token, inter-token latency, end-to-end latency, queue depth, and framework accepted or fresh sample counts. Use the same cold- or warm-cache procedure for every comparison.

Round-robin is a control, not a universal recommendation. Verify that every intended worker is eligible and receives traffic before trusting the baseline.

## Evaluate Prefix Reuse

Enable KV-aware routing when prompts share token prefixes:

```bash
python -m dynamo.frontend --router-mode kv
```

For parallel samples that arrive before the first worker publishes KV events, add a short predicted-placement window:

```bash
python -m dynamo.frontend \
  --router-mode kv \
  --router-predicted-ttl-secs 5
```

Treat `5` seconds as a starting value. Tune it against the observed gap between routing and usable KV events. Confirm that equivalent prompts use the same model, tokenizer, cache salt, LoRA identity, and token sequence; the router cannot recover reuse hidden by the request representation.

If one cache-rich worker receives too much work, compare overlap-credit decay or a lower overlap credit while holding the request schedule fixed. Use `--load-aware` as the control that accounts for active load without crediting prefix reuse:

```bash
python -m dynamo.frontend --load-aware
```

## Map the Setting to the Framework

Configure the frontend that actually receives rollout traffic; do not launch a second frontend just to copy a generic command.

| Framework | Setting | Boundary |
|---|---|---|
| verl native-router path | `actor_rollout_ref.rollout.engine_kwargs.dynamo.router_mode` | Set `thunderagent.enabled=false`; ThunderAgent owns a different scheduling path. |
| NeMo RL managed backend | `policy.generation.dynamo_cfg.frontend_args.router_mode` | NeMo RL launches and owns the frontend. Change this field for the baseline. |

When workers advertise router configuration, verify the effective worker-set values in the frontend logs. Worker settings can replace frontend defaults rather than merge with them.

## Use Affinity and Queueing Deliberately

`X-Dynamo-Session-ID` identifies a session but does not enable affinity. Enable affinity only when related turns must remain on one worker:

```bash
python -m dynamo.frontend \
  --router-mode kv \
  --router-session-affinity-ttl-secs 300
```

Prefer ordinary KV-aware routing when cache state is enough. Affinity does not create a backend conversation, enforce a policy version, or update workers when a session ends.

Add router queueing only after measuring dispatch and engine queue pressure. Compare first-come, first-served (`fcfs`) for tail behavior with weighted shortest processing time (`wspt`) for mixed prompt lengths. Use bounded service classes; never encode rollout IDs, users, or policy versions as queue classes or Prometheus labels.

## Measure Useful Work

Keep model, hardware, prompts, arrival schedule, concurrency, output limits, worker count, parallelism, cache state, and update cadence fixed. Run at least three measured repetitions after warm-up.

Report:

- request success, generated tokens, queue time, latency, and per-worker load
- KV-cache hits and queries when cache reuse is the mechanism
- completed and accepted trajectory groups, not only individual requests
- stale or rejected samples and rollout-phase time
- the causal explanation for the result

Many RL workloads wait for every required sample in a group. Measure the time from first dispatch to the final accepted attempt and identify the slowest request's queue, prefill, and decode contribution. A faster mean request that does not improve group completion or useful training output is not a win.

The router does not filter workers by RL policy version. Gate synchronous updates in the framework, or enforce bounded staleness and sample acceptance there.

## Diagnose Common Problems

| Symptom | Inspect first |
|---|---|
| Low cache reuse for repeated prompts | Tokenized prefix identity, block size, cache events, model/LoRA identity, and worker placement |
| One worker is cache-rich but overloaded | Overlap credit, active prefill/decode load, decay, and affinity |
| Sibling requests scatter | Request burst timing versus KV-event publication; predicted-placement TTL |
| Queue grows while workers appear idle | Eligible worker set, queue threshold, capacity input, and backend prefill limits |
| Priority has no effect | Whether requests enter the router queue under controlled pressure |
| Throughput rises but accepted sample rate falls | Framework update barrier, served policy identity, and acceptance logic |
| Cache reuse drops after policy update | Required cache invalidation and the post-update warm-up window |

Use the [metrics catalog](../../reference/observability/metrics-catalog.mdx#router-metrics) for exact metric names and [Observe and Simulate RL Rollouts](operations-and-simulation.md) to correlate framework, router, and worker behavior.
