---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Deficit Round Robin Queue Scheduling
subtitle: Weighted arbitration across router policy classes
---

The router uses a Deficit Round Robin (DRR) variant that is work-conserving
across dispatchable physical policy classes. DRR determines which class can
dispatch next; the configured FCFS or WSPT policy determines request order
within that class's shared and exact-worker lanes.

This separation provides:

- Weighted service across policy classes with different request sizes.
- Independent within-class ordering, exact-worker lanes, and strict-priority tiers.
- Progress for requests whose token cost is much larger than their class quantum.
- Bounded arbitration work that does not loop once per token or DRR round.

## Request Cost and Quantum

Each request receives an immutable queue snapshot when it is enqueued:

```text
uncached_tokens = raw_isl_tokens - cached_tokens
scheduling_cost = max(1, uncached_tokens)
```

The router uses exact uncached tokens for cache-bucket classification and uses
the clamped `scheduling_cost` for DRR and WSPT. The snapshot is not recomputed
while the request waits.

Each physical policy class defines a positive `quantum`, measured in uncached
tokens. A class with quantum `4096` earns four times as much DRR credit per
round as a class with quantum `1024`. This weighting controls token service,
not request count: variable-size requests consume correspondingly different
amounts of credit.

## DRR State

The scheduler maintains the following state for each physical class:

- A shared pending heap for requests that can use any eligible worker.
- One pending heap per exact-worker target, ordered by the same strict priority
  and FCFS or WSPT policy.
- A deficit containing earned but unspent credit.
- A quantum controlling how quickly the deficit grows.

The scheduler also maintains a ring cursor identifying the first class to
visit on the next arbitration call. Starting each scan at the cursor prevents
the configured class order from becoming a permanent preference.

## Selecting the Next Request

For each class in cursor order, the scheduler selects a class candidate by
comparing the dispatchable shared-heap head with the dispatchable head of each
exact-worker lane. It then applies DRR to that candidate:

1. If the class is empty, reset its deficit to zero.
2. If none of its lane heads can currently dispatch, retain its deficit but add no credit.
3. If existing deficit covers the selected candidate's cost, dispatch it without adding another quantum.
4. Otherwise, add one quantum and dispatch if the candidate is now affordable.
5. If the candidate remains unaffordable, continue to the next class.

Quantum is granted per ring round, not per request. A class that retained
enough credit can dispatch multiple requests from the same weighted allocation:

```text
quantum = 10
request costs = 3, 3, 3, 3

grant one quantum: deficit = 10
dispatch cost 3:   deficit = 7
dispatch cost 3:   deficit = 4
dispatch cost 3:   deficit = 1
next cost is 3:    advance the cursor
```

Adding another quantum for every request would let small requests accumulate
credit faster than they consume it and would violate the configured weighting.

## Bulk Credit for Oversized Requests

A request may cost many times its class quantum. Repeatedly scanning the ring
once per virtual round would make arbitration time proportional to request
size. Instead, after one complete ring makes no progress, the scheduler
calculates how many additional complete rounds are required for each
dispatchable class:

```text
rounds_needed =
    ceil((head_cost - current_deficit) / quantum)
```

It selects the minimum `rounds_needed`, adds that number of virtual rounds to
every dispatchable class, and scans the ring once more. Each class receives:

```text
added_credit = class_quantum * virtual_rounds
```

Applying the same virtual-round count preserves weighting because every class
still scales credit by its own quantum.

For example:

| Class | Quantum | Head cost | Deficit after normal visit | Additional rounds needed |
|---|---:|---:|---:|---:|
| `standard` | 1000 | 7000 | 1000 | 6 |
| `latency` | 2000 | 9000 | 2000 | 4 |

The scheduler fast-forwards four rounds. `standard` gains `4000` credit and
reaches `5000`, while `latency` gains `8000` and reaches `10000`.
`latency` can then dispatch and retains `1000` credit after paying its cost.

If every class head is blocked, there are no dispatchable classes and the
scheduler adds no bulk credit.

## Charging and Cursor Movement

After selecting a request, the scheduler subtracts its immutable scheduling
cost from the class deficit.

- If the class becomes empty, its deficit resets and the cursor advances.
- If the next head is already affordable, the cursor stays on the class so it
  can spend the remainder of its weighted burst.
- Otherwise, the class retains its remaining deficit and the cursor advances.

Blocked classes retain previously earned credit but do not accumulate more
credit while blocked. This prevents unavailable classes from building an
unbounded burst while preserving work they had already earned.

## Dispatchability and Head-of-Line Behavior

A shared head is dispatchable when its eligible workers are not all above the
class busy threshold. An exact-worker lane head is checked against its target
worker. If no eligible endpoint remains, the candidate proceeds to worker
selection so the router can return the appropriate error instead of parking it
indefinitely. Eligibility continues to enforce exact pins, worker allow-lists,
DP-rank bounds, taints, and overload filtering.

Head-of-line blocking is lane-local. The shared heap does not search past a
blocked shared head, and an exact-worker lane does not search past its own
blocked head. A blocked exact-worker lane does not prevent another exact-worker
lane, or a dispatchable shared head, from competing for the class. FCFS/WSPT
ordering is not bypassed within any individual lane.

New arrivals also join an existing backlog in their resolved class instead of
bypassing queued work. Queue limits, ordering, and DRR charging apply equally
to allow-listed and unconstrained requests.

## Complexity and Progress

One arbitration call performs:

1. At most one ring scan across all classes.
2. One linear calculation for bulk virtual rounds when required.
3. At most one final ring scan.

For `C` configured classes and `L` exact-worker lane heads that must be
inspected or rechecked, arbitration is `O(C + L log L)` in the worst case,
regardless of request cost or quantum. Candidate exact-worker heads are tree
indexed, so ordinary selections do not need to rescan every ready lane; a
worker-state recheck can visit the affected blocked lanes and pay the tree
update factor. The queue actor calls arbitration repeatedly while work remains
dispatchable, but each individual selection is bounded and continuation
draining remains local to the actor.

With only the synthetic no-YAML `default` class, the ring contains one class.
DRR reduces to single-class arbitration, but exact-worker requests still use
their separate lanes.

## Configuration

Set each physical class's `quantum` in the router policy YAML:

```yaml
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: cached
  - min_tokens: 3072
    bucket: uncached

policy_classes:
  - name: cached
    policy_family: standard
    cache_bucket: cached
    queue_policy: wspt
    quantum: 2048

  - name: uncached
    policy_family: standard
    cache_bucket: uncached
    queue_policy: fcfs
    quantum: 512
```

Use larger quantum ratios only when the corresponding classes should receive
larger shares of uncached-token service. For the complete policy-family and
cache-bucket schema, thresholds, and per-worker queue limits, see
[Configuration and Tuning](configuration-and-tuning.md#policy-class-queues). See
the tested [sample policy](https://github.com/ai-dynamo/dynamo/blob/main/examples/router/policy-class-queues.yaml)
for a complete profile.

## Prioritize Premium Requests with Policy Classes

Policy classes help a shared deployment protect premium traffic during demand
spikes. Under sustained prefill pressure, the router gives premium requests a
larger share of queued service while regular requests continue receiving
service. When premium demand subsides, regular traffic can use all available
capacity.

### Configure Premium and Regular Classes

This example mirrors the CPU Mocker regression test: one aggregated worker
serves four concurrent sequences, and every request has 512 uncached input
tokens and generates one output token. Save the following configuration as
`policy-classes.yaml`:

```yaml
default_policy_family: regular
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: premium
    policy_family: premium
    cache_bucket: all
    queue_policy: fcfs
    quantum: 512
    prefill_busy_threshold: 1536
    request_queue_limit_per_worker: 1024
  - name: regular
    policy_family: regular
    cache_bucket: all
    queue_policy: fcfs
    quantum: 128
    prefill_busy_threshold: 1536
    request_queue_limit_per_worker: 1024
```

Start the backend worker, then launch the frontend with load tracking and the
policy configuration:

```bash
python -m dynamo.frontend \
    --router-mode kv \
    --load-aware \
    --router-policy-config ./policy-classes.yaml
```

In this one-output-token, four-wide workload,
`prefill_busy_threshold: 1536` admits four 512-token requests before sustained
prefill pressure causes later requests to enter the policy queues. While both
queues remain backlogged, the `512:128` quantum ratio gives premium four times
the DRR credit. Tune the threshold to the request sizes and prefill pressure at
which queueing should begin in your deployment.

### Select a Class on Each Request

Send the requested policy family in the `x-dynamo-meta-policy-class` header:

```bash
curl http://localhost:8000/v1/completions \
    -H "content-type: application/json" \
    -H "x-dynamo-meta-policy-class: premium" \
    -d '{"model":"YOUR_MODEL","prompt":"YOUR_PROMPT","max_tokens":1}'

curl http://localhost:8000/v1/completions \
    -H "content-type: application/json" \
    -H "x-dynamo-meta-policy-class: regular" \
    -d '{"model":"YOUR_MODEL","prompt":"YOUR_PROMPT","max_tokens":1}'
```

Requests with a missing or unrecognized class header use the `regular`
default family in this configuration. See
[Policy-Class Queues](configuration-and-tuning.md#policy-class-queues) for the
complete class and cache-bucket resolution rules.

### Observe the Premium Share

The regression workload releases 320 distinct 512-token prompts from each
class at the same time. This keeps both queues backlogged with identical
request costs. An equal-share control changes the regular quantum from `128`
to `512`. Across four fresh CPU Mocker runs, the test observed:

| Configuration | First 320 client completions | Regular requests left when premium finishes |
|---|---|---:|
| Equal `512:512` | 159-161 premium and 159-161 regular | 0-3 |
| Weighted `512:128` | 254-257 premium and 63-66 regular | 237-240 |

The weighted configuration shifts the first 320 completions from an even
split to approximately 4:1. Premium receives substantially more service, and
regular still completes 63-66 requests during the same window. Measuring a
completion prefix captures the service share while both queues are active.

Each request costs 512 uncached tokens in this workload. Premium earns 512
tokens of DRR credit per round and regular earns 128, producing the ideal
`256:64` split. Equal request costs make the queued token-service ratio visible
directly in request completions; with varied request sizes, `quantum` continues
to control the share of uncached-token service.

The CPU Mocker regression test at
`tests/router/test_policy_class.py`
asserts both the larger premium share and continued regular progress.
