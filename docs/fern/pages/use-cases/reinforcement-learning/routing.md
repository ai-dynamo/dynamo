---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Route RL Rollouts
subtitle: Match routing, cache reuse, and queueing to rollout workload shape
---

**Experimental.** RL rollout routing uses the same Dynamo router as other inference workloads, but the optimization objective and failure boundary are different. Measure serving efficiency together with framework-owned sample freshness and acceptance. Dynamo routes requests; it does not decide whether a trajectory is on-policy, accepted, or fresh enough for training.

This guide explains how to choose and validate an RL routing strategy. Use the [router configuration and tuning reference](../../developer-guide/knowledge-base/modular-components/router/configuration-and-tuning.md) for every flag, default, and cost-model detail.

## Start with the Workload, Not the Policy Name

Record the following before selecting a router mode:

- prompt and generated token length distributions, including tails
- samples per prompt and how quickly sibling requests arrive
- repeated system, rubric, few-shot, environment, and conversation prefixes
- single-turn versus multi-turn sessions and child-agent fan-out
- rollout-phase concurrency and its burst/idle schedule around trainer updates
- worker count, tensor/data parallel layout, aggregated versus P/D topology, and cache capacity
- whether requests can be delayed, rejected, retried, or canceled without violating framework semantics
- policy update frequency and how the framework gates requests around updates
- the serving metric and the framework goodput metric used to judge success

A router cannot recover prefix reuse that the request representation hides. Confirm that equivalent prefixes produce compatible token IDs, cache salts, model identity, LoRA identity, and block boundaries before tuning overlap credit.

## Choose an Initial Strategy

| Rollout shape | Start with | Why | Main risk to watch |
|---|---|---|---|
| Independent prompts with little prefix reuse | Round-robin baseline, then `--load-aware` | Establishes distribution cost before adding cache signals; load-aware mode uses active load without assuming reuse | Queue imbalance from mixed output lengths |
| Many samples per prompt or shared rubric/system prefix | `--router-mode kv` | Credits workers that already hold the prompt prefix | One cache-rich busy worker can win too often |
| Sibling samples arrive in one burst before KV events are published | KV routing plus a short `--router-predicted-ttl-secs` | Records recent routing decisions in a short-lived side index so siblings can reuse the first placement | Too-long prediction TTL can preserve stale assumptions |
| Multi-turn agent trajectory | KV routing first; add session affinity only when strict worker stickiness is required | KV routing naturally follows cache state, while affinity pins a session independent of the full cost comparison | Affinity can overload one worker or outlive useful cache state |
| Mixed short and long prompts with queue pressure | KV or load-aware routing plus a queue threshold; compare `fcfs` and `wspt` | Defers dispatch until capacity is available and chooses tail- versus mean-oriented ordering | Queueing can increase end-to-end rollout latency or trigger framework timeouts |
| P/D serving across topology domains | KV routing with validated topology-aware transfer | Avoids slow cross-domain KV movement after prefill selection | Backend metadata and topology constraints must be present and current |
| Different service classes or rollout phases share a frontend | Policy-class queues only after a simple baseline | Isolates queue limits, ordering, and service shares | Misclassification, high-cardinality classes, or starvation from an unvalidated policy file |

The first comparison should normally be a simple distribution baseline versus one mechanism justified by the workload. Adding KV routing, queueing, priority, affinity, offload, custom policies, and autoscaling at the same time makes a causal result impossible.

## Establish a Round-Robin Baseline

Run the frontend with the same model, workers, and request schedule that the variant will use:

```bash
python -m dynamo.frontend --router-mode round-robin
```

Record at least completed requests, generated tokens, request errors, time to first token, inter-token latency, end-to-end request duration, worker utilization, queue depth, and framework accepted/fresh sample counts. Clear or warm caches consistently between repetitions and state which condition you used.

Round-robin is a comparison baseline, not a recommendation for every RL workload. It can duplicate a shared prefix across workers, but it also reveals whether a more complex strategy is paying for itself.

## Map the Router Setting to the Framework

The framework must configure the same frontend that actually receives rollout traffic. Do not launch a second frontend only to copy a generic CLI example.

| Framework path | Router setting | Important boundary |
|---|---|---|
| verl native Dynamo variant | `actor_rollout_ref.rollout.engine_kwargs.dynamo.router_mode` with `thunderagent.enabled=false` | The recipe-owned frontend translates this value into Dynamo routing. When ThunderAgent is enabled, it owns the internal scheduling decision and the same comparison no longer isolates native Dynamo routing. |
| NeMo RL managed backend | `policy.generation.dynamo_cfg.frontend_args.router_mode` | NeMo RL launches its own frontend and forwards the validated value. The pinned smoke uses `kv`; create a `round-robin` control by changing this field, not by starting an external frontend. |
| SLIME and Prime-RL status paths | Integration-specific prototype configuration | Their accepted upstream routing contracts are unresolved. Do not publish a launch or performance recommendation from the status pages. |

NeMo RL also exposes `router_reset_states` under the same `frontend_args` object. Treat it as startup state handling, not policy-update cache invalidation: NeMo RL separately pauses and clears every worker after a refit. The current merged NeMo RL adapter does not forward rollout session IDs, so session-affinity experiments require adapter work and fresh validation rather than only setting a frontend TTL.

## Route for Prefix Reuse

Enable KV-aware routing on the frontend:

```bash
python -m dynamo.frontend --router-mode kv
```

KV routing combines prompt-side load with cache-overlap credit. The selected backend and deployment must publish the KV events or other cache state required by the chosen configuration; otherwise the router can only use the signals it actually observes.

### Parallel samples and best-of-N groups

RL workloads often issue several samples with the same prompt almost simultaneously. The first routing decision can occur before the engine publishes a “block stored” event, causing every sibling to appear to have zero overlap. A short prediction window closes that race:

```bash
python -m dynamo.frontend \
  --router-mode kv \
  --router-predicted-ttl-secs 5
```

Validate the value against the observed gap between routing and usable KV events. Compare per-worker prefix placement, cache hits/queries, request queueing, and generated response lengths. Do not keep the example value solely because it is documented; it is a starting point for bursty sibling arrivals.

### Balance cache reuse against current load

`--router-kv-overlap-score-credit` controls how much device-local prefix overlap reduces prompt-side cost. Higher credit favors reuse; lower credit distributes work more evenly. `--router-kv-overlap-score-credit-decay` reduces the device credit when a cache-rich worker has more active prefill work than the least-loaded candidate.

Change one knob at a time. A useful test sequence is:

1. Default KV configuration.
2. Default overlap credit with nonzero load decay.
3. A lower overlap credit when inter-token latency or worker imbalance remains high.
4. `--load-aware` to measure the value of load modeling without cache reuse.

Keep host, disk, and shared-cache credit separate from device-local credit. A hit in a lower tier still has transfer/materialization cost and should not automatically receive full device-cache credit.

## Use Load-Aware Routing Without Prefix Reuse

When prompts are mostly unique but worker backlog differs, use the load-aware preset:

```bash
python -m dynamo.frontend --load-aware
```

This selects the KV scheduler's active load model while setting prefix-overlap credit to zero and disabling KV reuse assumptions. It is a cleaner comparison than enabling KV mode and leaving stale or incomplete cache signals in the cost model.

Measure prompt-side active load and decode work separately. A worker can look inexpensive at dispatch while carrying long decode requests that later dominate inter-token latency; use the router's active-request and load metrics to explain the result rather than relying on aggregate utilization.

## Decide Whether to Use Session Affinity

Sending `X-Dynamo-Session-ID` identifies a session but does not enable affinity. Enable affinity explicitly with a bounded TTL:

```bash
python -m dynamo.frontend \
  --router-mode kv \
  --router-session-affinity-ttl-secs 300
```

Use affinity when the application requires related turns to return to one worker even when another candidate has a lower current cost. Prefer ordinary KV routing when cache state itself is enough: it can follow reused prefixes without pinning a long-running session to an overloaded or replaced worker.

For strict affinity across multiple frontends, ensure ingress consistently routes a session to one frontend or validate replicated affinity behavior. Session affinity does not create a backend conversation, enforce a policy version, or issue lifecycle RPCs when the session ends.

## Add Queueing and Priority Carefully

`--router-queue-threshold` makes the router defer dispatch while every eligible worker exceeds the configured fraction of prefill-token capacity. Queueing is disabled when no threshold is set.

```bash
python -m dynamo.frontend \
  --router-mode kv \
  --router-queue-threshold 0.8 \
  --router-queue-policy fcfs
```

Use `fcfs` when tail time to first token is the primary concern. Compare `wspt` when minimizing average time to first token for mixed prompt lengths matters more. SGLang deployments require special care because the capacity value used by the threshold can differ from the intended prefill window unless `--max-prefill-tokens` is set; follow the backend caveat in the router tuning reference.

`nvext.agent_hints.strict_priority` selects an absolute router queue tier and `nvext.agent_hints.priority` adjusts ordering within the active policy. These hints affect requests only when they enter the router queue. They do not set backend engine priority, change trainer importance, or guarantee admission.

Do not encode rollout IDs, users, or policy versions as queue classes or Prometheus labels. Keep classes bounded to a small operational taxonomy such as validation, rollout, and evaluation only when the policy is explicitly designed and measured.

## Keep Routing Separate from Policy Freshness

The current typed Dynamo request extension does not contain a stable RL policy version, target trainer step, or maximum lag field. The router therefore cannot guarantee that a request is served by the newest acceptable policy.

Use one of these framework-owned patterns:

| Training/update model | Required framework behavior |
|---|---|
| Synchronous stop-the-world | Gate all new rollout requests, update and verify the entire target worker set, then reopen generation. |
| Rolling replacement | Route new requests only after the deployment mechanism exposes a fully initialized target pool; keep old and new pools distinguishable outside the current router contract. |
| Bounded-staleness asynchronous RL | Maintain policy identity and sample acceptance in the framework, prove which worker version served each sample, and reject or down-weight samples beyond the configured lag. Do not claim Dynamo enforces the bound. |

Routing measurements should report fresh completed trajectories or accepted samples in addition to raw tokens/second. A policy that raises serving throughput while increasing stale or rejected samples can reduce useful training goodput.

## Measure the Group Tail, Not Only the Average Request

Many RL workloads admit or score a group only after all required samples reach an acceptable terminal state. One long decode, overloaded worker, failed stream, or resampled attempt can therefore gate the entire group even when mean request latency improves. Dynamo does not know which requests form a training group, which attempts replace failures, or when the framework considers the group complete.

Keep group identity and attempt disposition in the framework ledger, then report at least:

- time from first group dispatch to the final accepted terminal attempt
- completed and accepted groups per unit time, together with individual request throughput
- within-group latency spread and the slowest request's queue, prefill, and decode contribution
- groups delayed or invalidated by cancellation, retry, worker loss, or a policy-update barrier
- whether a routing change improved the group tail by reducing repeated prefill, queue imbalance, or another observed mechanism

For matched routing runs, preserve the same group composition, arrival pattern, retry policy, output limits, and acceptance rule. Do not replace a failed attempt in only one variant or compare mean request latency when the framework waits on group completion.

## Design a Credible Routing Experiment

Use a matched experiment record:

| Dimension | Record |
|---|---|
| Software | Dynamo commit/release, framework commit, backend version, image digest, CUDA/driver |
| Model | model/tokenizer revision, precision, dense/MoE, parallel layout |
| Hardware | GPU type/count, node topology, network, CPU/memory constraints |
| Workload | prompt/output distributions, samples per prompt, concurrency, schedule, sessions, cache-sharing structure |
| Baseline | router mode and every nondefault routing/queue/cache option |
| Variant | exactly one intended mechanism change where possible |
| Cache state | cold/warm procedure, update/reset schedule, offload tiers |
| Repetitions | warm-up, measured runs, variance or spread |
| Serving outcomes | request success, tokens, TTFT, ITL, end-to-end latency, queue time, KV hits/queries, per-worker load |
| RL outcomes | completed fresh trajectories, accepted samples, policy lag/rejections, rollout phase time, full-step time |

When publishing a result, state the causal mechanism. For example: “the variant reduced repeated prefill work because sibling requests reused a predicted prefix placement,” not “KV routing was X% faster” without cache and workload evidence.

## Record Evidence for a Routing Claim

Before the experiment, record immutable pins, named owners, the headline metric's numerator, denominator, and freshness rule, complete workload shape, fixed controls, and the full router configuration for the baseline and variant. Preserve at least three measured repetitions per variant plus immutable links for the raw requests, configuration, and computed metrics. Label a routing claim as a live measurement only after those artifacts demonstrate the declared mechanism.

Do not publish a recommendation based on a single configuration, fewer than three repetitions, a missing or nonnumeric headline metric, multiple or absent baselines, unmatched controls, missing mechanism evidence, or simulation alone. When a routing result supports a broader RL deployment recommendation, review it together with the claimed weight paths and the [combined observability, replay, and simulation evidence](operations-and-simulation.md#complete-the-cross-cutting-validation-report) so the conclusion refers to one pinned program rather than unrelated demonstrations.

## Diagnose Common Routing Failures

| Symptom | Inspect | Likely experiment |
|---|---|---|
| Low cache hit rate despite repeated prompts | tokenized prefix identity, cache salt, block size, KV events, model/LoRA identity | Compare trace hashes and per-worker cache events for two supposedly identical prompts. |
| One worker is cache-rich but overloaded | overlap credit, active prefill/decode load, decay, affinity | Add overlap-credit decay or lower credit in a matched run. |
| Sibling requests scatter before cache events | request arrival burst versus event delay | Add a short predicted TTL and verify placement. |
| Queue grows while workers appear idle | capacity denominator, worker eligibility, SGLang max prefill tokens, policy class | Remove queueing for a control run, then validate capacity inputs. |
| Priority has no effect | whether requests enter the router queue | Force controlled queue pressure and inspect per-tier queue metrics. |
| Multi-turn sessions overload one worker | affinity TTL and session distribution | Compare affinity with ordinary KV routing and shorten/disable affinity. |
| Throughput rises but accepted sample rate falls | policy-update barrier and framework freshness ledger | Correlate request time with target/served policy identity outside the current typed router schema. |
| Cache hit rate collapses after update | cache invalidation and warm-up | Verify the expected reset, then separate warm-up from steady-state measurements. |

## Observe the Router

The frontend exposes router Prometheus metrics on `/metrics`, including request metrics, routing overhead, and per-worker gauges. Use the [metrics catalog](../../reference/observability/metrics-catalog.mdx#router-metrics) for exact names and the [operations guide](operations-and-simulation.md#diagnose-the-live-run) for an RL correlation workflow.

Use traces for high-cardinality request/session/rollout joins and metrics for bounded aggregate dimensions. A routing dashboard should show request rate, queue depth/time, cache hits and queries, active prompt/decode work, routing overhead, per-worker balance, errors/cancellations, and the framework's accepted/fresh goodput on the same time axis.

## Validation Gate

A routing recommendation is publication-ready only when the workload, topology, baseline, variant, repetitions, serving metrics, RL goodput metric, and causal mechanism are recorded and independently reviewed. Simulated results must be labeled directional and calibrated against a live run before they become performance claims. See [Replay and simulate the request plane](operations-and-simulation.md#replay-and-simulate-the-request-plane) for the fidelity boundary.
