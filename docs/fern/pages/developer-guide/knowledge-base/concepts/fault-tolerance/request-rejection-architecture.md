---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Request Rejection Architecture
subtitle: Worker-load event processing, busy-state aggregation, overload errors, and hard worker admission limits.
---

Dynamo implements request rejection (load shedding) at two layers: Frontend routing can avoid workers
reported as busy, and each worker can enforce a hard request-plane concurrency cap.

For deployment steps, see [Request Rejection](../../../../kubernetes/fault-tolerance/request-rejection.md). For exact
configuration fields, see [Frontend Configuration](../../../../reference/components/frontend-configuration.mdx#fault-tolerance)
and [Runtime Configuration](../../../../reference/components/runtime-configuration.mdx#operations).

## Request Flow

```text
                                    ┌─────────────────┐
                                    │ Worker Monitor  │
                                    │  (background)   │
                                    └────────┬────────┘
                                             │ worker-load updates
                                             ▼
┌──────────┐    ┌──────────┐    ┌─────────────────────┐    ┌──────────┐
│  Client  │───▶│ Frontend │───▶│     Push Router     │───▶│  Worker  │
└──────────┘    └──────────┘    │ excludes busy set   │    └──────────┘
                                └──────────┬──────────┘
                                           │ every eligible worker busy
                                           ▼
                                ┌─────────────────────┐
                                │ HTTP 529 Overloaded │
                                └─────────────────────┘
```

The router distinguishes two failure classes:

- **Overloaded** means workers are registered but every eligible worker is busy. The Frontend returns
  HTTP 529 by default.
- **Unavailable** means no usable service path exists. The Frontend returns HTTP 503.

`DYN_HTTP_OVERLOAD_STATUS_CODE` can change the overload response code for client compatibility. Values
from 200 through 999 are accepted. Informational values from 100 through 199, invalid values, and
out-of-range values fall back to 529. The value is read and cached on first use.

## Independently Enabled Signals

All three busy thresholds are `None` by default. Setting a numeric value activates only that signal;
there is no master admission-control switch.

For each data-parallel rank, the monitor evaluates the configured checks with OR logic:

```text
decode_busy = active_decode_blocks / kv_total_blocks > decode_threshold
absolute_prefill_busy = active_prefill_tokens > absolute_prefill_threshold
fractional_prefill_busy =
    active_prefill_tokens > fractional_prefill_threshold * max_num_batched_tokens

rank_busy = any(configured check is true)
worker_busy = all(data-parallel ranks are busy)
```

The fractional and absolute prefill checks can be enabled separately or together. A worker is not
excluded until all of its data-parallel ranks are busy, which avoids discarding capacity on ranks that
can still admit work.

Decode-block rejection depends on the KV router worker-load path. `--router-mode kv` initializes that
path. `--router-track-output-blocks` adds generated output tokens to the router's observed active-block
count; without it, long outputs can consume KV cache without appearing in the tracked load. The
separate `--router-track-active-blocks` option affects the router cost model and is not a prerequisite
for busy rejection.

## Worker Load Monitoring

`KvWorkerMonitor`:

1. Subscribes to worker KV and prefill load events.
2. Stores per-worker, per-rank values such as `active_decode_blocks`, `kv_total_blocks`,
   `active_prefill_tokens`, and `max_num_batched_tokens`.
3. Recalculates the busy set when load or runtime configuration changes.
4. Publishes the current busy set to the router.

A `POST /busy_threshold` update changes the stored threshold configuration. It does not synchronously
recompute every worker. The next worker-load or runtime-configuration update triggers reevaluation, so
a new threshold can take a short time to change routing decisions.

## Rejection Path

When a request arrives:

1. The push router resolves the registered workers for the model.
2. If at least one busy threshold is configured, the router removes workers in the current busy set.
3. If registered workers exist but no eligible worker remains, the router returns
   `PipelineError::ServiceOverloaded`.
4. The HTTP layer maps overload to the configured overload status, 529 by default.
5. The Frontend increments `dynamo_frontend_model_rejection_total`.

The Frontend also exports the latest observed worker values through
`dynamo_frontend_worker_active_decode_blocks` and
`dynamo_frontend_worker_active_prefill_tokens`, which help distinguish missing telemetry from a
threshold that is simply too high.

## Worker-Side Request Admission

A worker process admits through one process-global admission gate. There is a single admission point
in the runtime, on the shared ingress path both request planes already funnel through, so requests
arriving over TCP and over NATS compete for the same `N` engine slots and the same FIFO queue of size
`Q`, whether the backend is implemented in Rust or Python:

```text
TCP request plane  ─┐
                    ├─> shared ingress handler ─> gate ─> Rust or Python backend worker
NATS request plane ─┘
```

The gate is a property of the process, not of an individual endpoint: every endpoint served over the
shared ingress path shares one limit and one queue. Generation traffic dominates that budget in a
worker process, but the same process's control, status, indexer, LoRA-management and KV-management
endpoints draw on it too, so size `N` with a little headroom above the concurrency you expect the
engine to sustain. The in-process health-check canary is issued through the local endpoint registry
and never passes through the gate.

A request takes a free engine slot when one is available, otherwise it joins the FIFO queue, and
otherwise the worker refuses it with `Server overloaded: worker at capacity`. A slot is released when
the admitted request finishes, errors, or is cancelled, and passes to the oldest queued request. New
arrivals never bypass an older waiter, and a request cancelled while queued frees its queue slot
immediately.

The refusal happens before the backend is asked to do any work, so the request never reaches the
engine. It is reported through the same pre-stream error path a failed `generate` uses, which today
carries an opaque message rather than a category; the Frontend therefore treats it as it treats any
pre-stream worker failure. Preserving the worker-scoped classification across that boundary is
separate transport work.

`N` resolves in this order:

1. `DYN_ENGINE_REQUEST_LIMIT` (`--engine-request-limit`), when set to a positive integer.
2. `ceil(3/2 x max_num_seqs x data_parallel_size)` in integer arithmetic, from a capacity reported by
   model-card registration.
3. Exactly `10000`, when neither is available. This is the final limit; the `3/2` factor is never
   applied to it.

`Q` defaults to `40000` and is overridden independently by `DYN_DYNAMO_REQUEST_QUEUE_LIMIT`. The
queue holds exactly `Q` requests: no dispatcher holds a hidden `Q + 1`th.

The worker refusal does not add a failed-worker exclusion to the routing request or change the
standalone router protocol.

The effective maximum is `N + Q` requests. `DYN_DYNAMO_REQUEST_QUEUE_LIMIT` is an advanced override
read independently of the engine limit, and defaults to `40000`.

### Controlled Delay

Queueing is bounded in time as well as in length. Every request that joins the FIFO is stamped, at
enqueue, with a deadline of `enqueue time + queue delay`. The delay is one process-wide budget: it
defaults to `5000` milliseconds and is overridden by a positive `DYN_DYNAMO_REQUEST_QUEUE_TIMEOUT_MS`,
in whole milliseconds, down to `1`. That override is environment-only; there is no command-line flag.
An unset, unparseable or non-positive value leaves the default in place.

The deadline bounds queue residence only. It does not limit how long an admitted request may run,
it is not a per-request SLO, and a request that takes a free engine slot immediately never carries
one at all. When a queued request passes its deadline it is removed from the queue at once, so it
stops consuming queue capacity, and the worker refuses it with
`Server overloaded: request rejected after exceeding the backend admission queue delay` — the same
refusal path as a full queue, distinguished only by the message.

Because every entry gets the same delay budget at enqueue, FIFO order is also nondecreasing deadline
order, so expiry only ever removes a prefix of the queue and never inspects the requests behind it.
The gate owns exactly one timer for the oldest live deadline, re-armed whenever the head changes,
rather than a periodic sweep or a timer per queued request. Every path that hands a freed slot to a
queued request also re-checks the head deadline first, so a slot released before that timer fires is
still never given to a request the delay budget has already given up on. Queue order among requests
that have not expired is unchanged. This is transport-independent: the same gate, timer and expiry
result apply over TCP and NATS.

### Where The Capacity Hint Comes From

The implemented rule is exactly this: **the first usable capacity report from any non-LoRA base model
card registered in the process wins.** A report is usable when `max_num_seqs` and
`data_parallel_size` are both present and non-zero and their scaled product is in range. A later
conflicting report from any other base card in the same process is logged and ignored, and the
environment override always wins over any report. LoRA cards report no capacity.

The gate does not test which kind of component published the card, and nothing in the runtime records
that distinction. `register_model` is reached by routers as well as engines — `global_router`,
`vllm.omni.stage_router` and `thunderagent_router` all call it — so a router card carrying a usable
`max_num_seqs` would size the gate for its own process. In tree those router cards leave
`max_num_seqs` unset today, so they report nothing usable and the limit falls through to rule 3. That
is a property of the current router configurations, not a guarantee the runtime enforces: if you add
a `max_num_seqs` to a router's `ModelRuntimeConfig`, that value becomes that router process's
admission limit.

Registration can complete after the gate is already admitting requests, so the limit stays adjustable
rather than being frozen at startup and a component whose metadata resolves late is still sized
correctly. Raising the limit releases queued work immediately; lowering it never revokes a slot that
is already held.

### Relationship To The TCP Request Plane

The TCP request plane keeps its own worker pool, sized by `DYN_TCP_WORKER_POOL_SIZE` and
`DYN_TCP_WORK_QUEUE_SIZE`. That pool bounds TCP-side task execution only; it is independent of the
gate and no longer changes with the engine-admission settings. The two use the same numeric defaults
by coincidence, not by sharing constants.

The TCP worker pool exports:

- `dynamo_rejection_request_total`
- `dynamo_engine_request`
- `dynamo_request_queue`

These count TCP request-plane pool activity only. The backend admission gate does not yet export
metrics of its own; its resolved limit, queue length and each shed request appear in the worker log.

See [Cancellation and Rejection](../../../../reference/observability/metrics-catalog.mdx#cancellation-and-rejection)
for metric types and labels.

## Related Documentation

- [Request Rejection](../../../../kubernetes/fault-tolerance/request-rejection.md) - Enable, tune, verify, and troubleshoot load shedding
- [Frontend Configuration](../../../../reference/components/frontend-configuration.mdx#fault-tolerance) - Threshold and overload response fields
- [Runtime Configuration](../../../../reference/components/runtime-configuration.mdx#operations) - Worker hard-cap fields
- [Observability Architecture](../observability-architecture.md#active-worker-health-checks) - Worker health monitoring
