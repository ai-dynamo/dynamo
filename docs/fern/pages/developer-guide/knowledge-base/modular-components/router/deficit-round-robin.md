---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Deficit Round Robin Queue Scheduling
subtitle: Weighted arbitration across router policy classes
---

The router uses a Deficit Round Robin (DRR) variant that is work-conserving
across dispatchable policy classes. DRR determines which class can dispatch
next; each class's own queue determines request order within that class.

This separation provides:

- Weighted service across policy classes with different request sizes.
- Independent within-class ordering.
- Progress for requests whose token cost is much larger than their class quantum.
- Bounded arbitration work that does not loop once per token or DRR round.

## Request Cost and Quantum

Each request receives an immutable queue snapshot when it is enqueued:

```text
uncached_tokens = raw_isl_tokens - cached_tokens
scheduling_cost = max(1, uncached_tokens)
```

The router uses the clamped `scheduling_cost` for DRR. The snapshot is not
recomputed while the request waits.

Each policy class defines a positive `quantum`, measured in uncached tokens. A
class with quantum `4096` earns four times as much DRR credit per round as a
class with quantum `1024`. This weighting controls token service, not request
count: variable-size requests consume correspondingly different amounts of
credit.

## DRR State

The scheduler maintains the following state for each class:

- One runnable queue, ordered by the class's SLO deadline.
- A deficit containing earned but unspent credit.
- A quantum controlling how quickly the deficit grows.

The scheduler also maintains a ring cursor identifying the first class to
visit on the next arbitration call. Starting each scan at the cursor prevents
the configured class order from becoming a permanent preference.

## Selecting the Next Request

For each class in cursor order, the scheduler first rejects every head whose
class deadline has already passed, then takes the remaining head as that class's
candidate. It then applies DRR to that candidate:

1. If the class is empty, reset its deficit to zero.
2. If its head cannot currently dispatch, retain its deficit but add no credit.
3. If existing deficit covers the candidate's cost, dispatch it without adding another quantum.
4. Otherwise, add one quantum and dispatch if the candidate is now affordable.
5. If the candidate remains unaffordable, continue to the next class.

Rejecting an expired head spends no deficit and does not advance the ring
cursor, so shedding late work never consumes a class's service share.

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

A class head is dispatchable when the workers it may use are not all above the
class busy threshold. A head pinned to one worker is checked against that
worker. If no eligible endpoint remains, the candidate proceeds to worker
selection so the router can return the appropriate error instead of parking it
indefinitely. Eligibility continues to enforce exact pins, worker allow-lists,
DP-rank bounds, taints, and overload filtering.

Head-of-line blocking is class-local. A class holds one queue and only its head
is tested, so a head that cannot run yet, such as one pinned to a busy worker,
holds that class's line until it can. Other classes keep making progress through
DRR, and deadline ordering is never bypassed inside a class.

New arrivals also join an existing backlog in their resolved class instead of
bypassing queued work. Queue limits, ordering, and DRR charging apply equally
to allow-listed and unconstrained requests.

## Complexity and Progress

One arbitration call performs:

1. At most one ring scan across all classes.
2. One linear calculation for bulk virtual rounds when required.
3. At most one final ring scan.

For `C` configured classes, arbitration is `O(C)` head reads plus `O(log n)` per
request removed, regardless of request cost or quantum. Reading a class head is
constant time, and rejecting `k` expired heads costs `O(k log n)` on that class.
The queue actor calls arbitration repeatedly while work remains dispatchable,
but each individual selection is bounded and continuation draining remains local
to the actor.

With only the synthetic no-YAML `default` class, the ring contains one class and
DRR reduces to single-class arbitration.

Each class holds exactly one queue, ordered by due time. Before a class offers a
dispatch candidate, the poll removes every head whose class deadline has already
passed and rejects it. Expired work never spends class deficit and never
advances the ring cursor.

## Configuration

Set each class's `quantum` and `slo_ms` in the router policy YAML:

```yaml
default_policy_class: cached

policy_classes:
  - name: cached
    slo_ms: 600000
    quantum: 2048

  - name: uncached
    slo_ms: 600000
    quantum: 512
```

Use larger quantum ratios only when the corresponding classes should receive
larger shares of uncached-token service. `quantum` and `slo_ms` are independent:
`quantum` sets a class's share of service while both queues are backlogged, and
`slo_ms` sets how long a request in that class may wait for service before the
router sheds it. Both classes above carry the same generous objective, so this
profile weights service and no expiry is expected under it; see
[Reject Late Work with a Class SLO](#reject-late-work-with-a-class-slo) for a
profile where `slo_ms` decides an outcome. For the complete flat-class schema,
thresholds, and per-worker queue limits, see
[Configuration and Tuning](configuration-and-tuning.md#policy-class-queues).
See the tested [sample policy](https://github.com/ai-dynamo/dynamo/blob/main/examples/router/policy-class-queues.yaml)
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
default_policy_class: regular
policy_classes:
  - name: premium
    slo_ms: 600000
    quantum: 512
    prefill_busy_threshold: 1536
    request_queue_limit_per_worker: 1024
  - name: regular
    slo_ms: 600000
    quantum: 128
    prefill_busy_threshold: 1536
    request_queue_limit_per_worker: 1024
```

The two class configurations differ only in `quantum`, and both carry the same
generous `slo_ms`, so this example isolates DRR weighting: no expiry is expected
during the run, and each queue dispatches in arrival order. Lower `slo_ms` when
a class should shed work that cannot begin service before its queue deadline
rather than dispatching it late.

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

Send the policy class name in the `x-dynamo-meta-policy-class` header:

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

Requests that send no class header use `default_policy_class`, which is
`regular` in this configuration. A request that names a class this profile does
not configure is rejected. See
[Policy-Class Queues](configuration-and-tuning.md#policy-class-queues) for the
complete class resolution rules.

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

## Reject Late Work with a Class SLO

The previous walkthrough gave both classes the same generous `slo_ms` so that
`quantum` alone decided the outcome. This one holds `quantum` fixed and lets
`slo_ms` decide instead: when a queued request cannot begin service before its
queue deadline, the router rejects it rather than dispatching it late.

### Configure a Short and a Long Objective

Save the following configuration as `slo-classes.yaml`:

```yaml
default_policy_class: standard
policy_classes:
  - name: standard
    slo_ms: 30000
    quantum: 512
    prefill_busy_threshold: 0
    request_queue_limit_per_worker: 32
  - name: latency
    slo_ms: 3000
    quantum: 512
    prefill_busy_threshold: 0
    request_queue_limit_per_worker: 32
```

The two class configurations differ only in `slo_ms`; no other class
configuration field differs, so `quantum`, the busy threshold, and the queue
limit are identical. `prefill_busy_threshold: 0` uses a strict comparison, so
the worker counts as busy as soon as it holds any active prefill token: one long
request is enough to push every later arrival into its class queue. The queue
limits sit far above the offered load, so a rejection here can only be a
deadline expiry.

Start the backend worker, then launch the frontend:

```bash
python -m dynamo.frontend \
    --router-mode kv \
    --load-aware \
    --router-policy-config ./slo-classes.yaml
```

### Send a Long Request, Then a Short-SLO One

Send one large request in the `standard` class. It arrives while the worker is
idle, so the router dispatches it immediately without queueing. Make the prompt
large enough to keep the worker busy for well over the 3000 ms `latency`
objective. If it finishes sooner, the second request begins service before its
deadline and is served normally instead of being shed:

```bash
curl http://localhost:8000/v1/completions \
    -H "content-type: application/json" \
    -H "x-dynamo-meta-policy-class: standard" \
    -d '{"model":"YOUR_MODEL","prompt":"YOUR_LONG_PROMPT","max_tokens":1}' &
```

While it runs, send a small request in the `latency` class:

```bash
curl -i http://localhost:8000/v1/completions \
    -H "content-type: application/json" \
    -H "x-dynamo-meta-policy-class: latency" \
    -d '{"model":"YOUR_MODEL","prompt":"YOUR_SHORT_PROMPT","max_tokens":1}'
```

### Observe the Rejection

The worker is busy, so the second request enters the `latency` queue and cannot
begin service before its queue deadline. Three seconds after it arrived that
deadline passes, and the next deficit-round-robin poll of the class rejects it
at the head instead of dispatching it. The client receives HTTP 529 with a
structured body:

```json
{
  "message": "router policy class \"latency\" deadline exceeded at dispatch (slo=3000ms, overdue=1200ms)",
  "type": "Overloaded",
  "code": 529,
  "details": {
    "policy_class": "latency",
    "stage": "dispatch",
    "slo_ms": 3000,
    "overdue_ms": 1200
  }
}
```

`overdue_ms` is how far past the deadline the poll that shed the request ran, so
the value above is illustrative: it depends on when the next poll happens, not
on the configuration. `stage` names where the deadline was found to have passed:
`dispatch` means the request had already been queued and was shed at the class
head; `admission` means it was already late when it reached queue storage, and
`deferred_wake` means it expired while a custom queue-admission policy held it.
The same rejection increments
`dynamo_frontend_router_queue_deadline_expired_total{policy_class="latency",stage="dispatch"}`,
and the class's pending-request gauge returns to its previous value because
shedding reverses the queue accounting the request applied on the way in.

The `standard` request keeps running and returns HTTP 200 because it was
dispatched immediately: `slo_ms` bounds how long a request may wait for service,
not how long it may take once it is running, so a class SLO is never an
in-flight completion timeout. Its own 30000 ms objective is not what let it
finish; never being queued is.

The CPU Mocker regression test at `tests/router/test_policy_queue.py` runs this
scenario against a slowed-down worker and asserts the rejection status, the
class, stage, and SLO carried in the structured detail, the dispatch-stage
counter, and that the blocking request still completes.
