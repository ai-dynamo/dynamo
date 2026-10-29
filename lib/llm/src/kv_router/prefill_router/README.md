<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Prefill Router

`PrefillRouter` chooses prefill workers for disaggregated serving. It has three
users, and each one relies on a different part of its interface:

1. **The frontend pipeline** runs it as an operator between the encoder and the
   decode `RoutingHost`. Each request runs on a prefill worker first, then goes
   to decode with the KV handoff attached. With conditional disaggregation, a
   request can skip the prefill hop and run on its decode worker.
2. **Discovery** creates one router per WorkerSet and moves its prefill target
   as prefill workers come and go.
3. **The inference-gateway EPP** reserves a prefill worker without dispatching
   to it, then hands the choice to the frontend in a request header.

This file records the router's side of each contract and why it behaves the
way it does. The step order of one frontend request lives in the doc comment on
`PrefillRouter::generate`. Behavior users see is documented in the Fern docs
(`docs/fern/pages/developer-guide/advanced-customizations/conditional-disaggregation.md`
and the router knowledge-base pages).

## Files

- `mod.rs`: the `PrefillRouter` struct, the `Operator` impl (`generate`),
  `PrefillRouterLifecycle`, and decode-request construction
  (`build_decode_request`, `build_decode_router_override`).
- `activation.rs`: constructors, target tracking, binding construction, and the
  `ActivationSnapshot`.
- `conditional_bypass.rs`: the conditional-disaggregation decision, built on a
  non-admitting decode route preview.
- `admission.rs`: drains the prefill stream into a handoff, a terminal output,
  or an error.
- `handoff.rs`: keeps consuming prefill while decode runs and merges prefill
  failures into the decode stream.
- `query.rs`: `reserve_prefill_worker` and `PrefillReservation`.
- `stage.rs`: `PrefillRouter` as a `StageRouter` (`schedule`, `preview`,
  `dispatch`) and `generate_v2`, the request path composed from them. Not
  wired into the pipeline yet.

## Core: Target, Binding, and Activation

A **target** names the prefill endpoint the router should serve. A **binding**
is the router built for that target: the prefill endpoint, its router mode, and
its worker selection. Bindings are built asynchronously, so a target can exist
before its binding does.

| State | Meaning |
|---|---|
| `Active` | A target is set and its binding is built. |
| `Pending` | A target is set but its binding is still building. |
| `Untargeted` | No target was ever set, or the target was cleared. |

Each user handles the states as follows:

| State | Frontend `generate` | Discovery `available_worker_ids_for` | EPP `reserve_prefill_worker` |
|---|---|---|---|
| `Active` | Runs the prefill hop. | The binding's available workers. | Reserves a prefill worker. |
| `Pending` | Fails closed: migratable `WorkerUnavailable` (HTTP 503). | Empty set. | `NotActivated`. The EPP falls back to aggregated. |
| `Untargeted` | Forwards to decode unchanged. | `None`. | `NotActivated`. The EPP falls back to aggregated. |

Known gap: the EPP column does not fail closed during `Pending`. See
[Use 3](#use-3-the-inference-gateway-epp).

The hot path reads the lifecycle state with one atomic load. It takes the
target lock only when the state is not `Active`.

## Use 1: The Frontend Pipeline

### Interface

| Item | Caller | Purpose |
|---|---|---|
| `Operator::generate`, through `into_operator()` | `entrypoint/input/common.rs` | The pipeline stage. |
| `disabled_with_session_affinity_mode` | `common.rs` | A pass-through router for models without prefill. |
| `conditional_disagg_enabled`, `set_decode_routing_host` | `common.rs` | Wire the decode `RoutingHost`. Required when conditional disaggregation is on. |

`set_decode_routing_host` accepts one host for the router's lifetime.
`generate` expects it whenever the conditional plan keeps a request local.

### The activation gate runs before conditional disaggregation

This order is inherited, not designed. The gate predates conditional
disaggregation, and ai-dynamo/dynamo#11725 added the conditional branch below
it. No PR or review discussed the alternative of letting conditional
disaggregation decide first.

The current order still has reasons to stay:

- The gate is one atomic load. Conditional planning runs a KV index preview,
  which requests that are rejected or passed through would otherwise pay for.
- In `Untargeted` there is no prefill pool, so a local-or-remote decision has
  nothing to compare against.
- "Fail closed while `Pending`" stays unconditional.

Running conditional disaggregation first would let requests it keeps local
succeed during `Pending`. That changes behavior: it needs a decision on the
`Untargeted` path, a three-state prefill-load signal (see below), and updated
activation tests. Raise it under DEP ai-dynamo/dynamo#15457 rather than in a
refactor.

### `Pending` fails closed instead of passing through

Model admission counts a prefill worker as live when it registers, before this
router has built its binding. Passing through in that window sent aggregated
requests to decode-only workers, which fail with `bootstrap_info is required
for disaggregated decode`. ai-dynamo/dynamo#16736 (issue
ai-dynamo/dynamo#16529) made the window return a retryable 503 instead.

### The binding is checked again after the Prefill phase barrier

A request that passed the gate can reach the binding load after the target was
replaced, while it awaited conditional planning or the phase barrier. The
second check resolves from a fresh snapshot with the same rules as the gate.
It was added in review of ai-dynamo/dynamo#16736.

Known gap: if that second check finds `Untargeted`, the request is forwarded
with its tracker still in the Prefill phase.

### Query-only requests go first

A `query_instance_id` request is an advisory worker lookup. It goes straight
to the decode `RoutingHost` in every activation state, so it never triggers a
conditional bypass or a prefill hop.

### The bypass marker is router-owned

`generate` strips any client-supplied `x-bypass-remote-prefill` before
anything else, so a client cannot skip remote prefill. Only the conditional
branch adds the marker, and it also prepends an `x-bypass-remote-prefill`
annotation item to the response stream.

### Conditional planning errors fall back to remote prefill

Cancellation and invalid-argument errors fail the request. Any other planning
error logs a warning and continues through remote prefill, which is the
ordinary route.

### A missing prefill-load signal means "do not bypass"

The `prefill_load` and `isl_or_load` policies read whether the prefill worker
the router would pick is busy. `None` means the signal is unavailable, and the
load policy treats it as not busy. Three causes produce `None`:

- No threshold resolves: neither `conditional_disagg_prefill_busy_threshold`
  nor `router_queue_threshold` is set. The load policy then never fires, and
  nothing warns about it.
- No binding. After the gate this happens only in the target-replacement
  window, where the second check fails the request anyway.
- The probe returned an error.

### Prefill survives client cancellation

The prefill request runs on an independent cancellation context, and decode
routing proceeds even when the client context is stopped or killed. Stopping
either would orphan an in-flight KV transfer and leak KV blocks
(NVBugs 5969206).

### The prefill hop does not start the classifier lifecycle

KV selection returns early for `RequestPhase::Prefill`
(`routing_host/kv.rs`). Only the decode leg starts the request-classifier
lifecycle.

## Use 2: Discovery

### Interface

| Item | Caller | Purpose |
|---|---|---|
| `new_with_selection_policy` | `discovery/watcher.rs` | Construct a router for a WorkerSet. |
| `PrefillRouterLifecycle::set_target` | `discovery/model_manager.rs` | Move the target as prefill workers change. |
| `PrefillRouterLifecycle::available_worker_ids_for` | `discovery/model_manager.rs` | Report which prefill workers this router can serve. |

### Target changes

`set_target` is synchronous and updates the lifecycle under the target lock:

| Call | Binding | Lifecycle |
|---|---|---|
| Same target as now | Kept | Unchanged |
| A different target | Cleared, rebuilt asynchronously | `Pending` |
| The target its binding was built for | Kept | `Active` |
| `None` | Kept | `Unavailable`, seen as `Untargeted` |

Clearing is synchronous so that requests holding an older catalog snapshot stop
using a removed endpoint before the replacement catalog is published.

### Model admission ignores this lifecycle

WorkerSet selection is gated on registered worker topology, not on this
router's state (`test_is_displayable_ignores_prefill_router_lifecycle` in
`discovery/model.rs`). ai-dynamo/dynamo#16736 considered gating admission on
the lifecycle and rejected it: it would also change the intentional
pass-through when every prefill worker leaves an aggregated-capable
deployment. The router handles the activation window itself.

## Use 3: The Inference-Gateway EPP

### Interface

| Item | Caller | Purpose |
|---|---|---|
| `new` | `deploy/inference-gateway/ext-proc/src/epp.rs`, sidecar testkit | Public constructor. Activates once from `activation_rx`. |
| `reserve_prefill_worker` | EPP, sidecar testkit | Select and book a prefill worker without dispatching to it. |
| `PrefillReservation::{worker_id, dp_rank, release}` | EPP | The booking. |

### Activation is one-shot

`new` takes a one-shot `activation_rx`. When the EPP discovers a prefill
worker, it sends the endpoint and the router sets a `Legacy` target once.
Nothing clears that target later, so an EPP router never returns to
`Untargeted`. When prefill workers leave, reservations fail with "No workers
available for prefill" instead.

### Reservations own their cleanup

A KV-routed reservation holds the exact scheduler booking it admitted. Dropping
it enqueues cleanup, and `release` waits for the scheduler to acknowledge it.
Cleanup is tied to that booking, not to the current binding, so a later rebind
cannot release a newer booking that reused the same request ID and worker.
Built-in router modes return an untracked, advisory reservation whose release
does nothing. Reservations do not advance the round-robin dispatch cursor.

The EPP releases each booking once, from whichever response lifecycle callback
runs first.

### Handoff to the frontend

The EPP sends the reserved worker as `x-dynamo-prefill-instance-id` (and
`x-dynamo-prefill-dp-rank`). In Direct routing mode, the frontend's `generate`
requires that preselected worker and returns an error without it. A request
with a preselected prefill worker also skips conditional disaggregation.

### Known gap: errors fall back to aggregated

`reserve_prefill_worker` returns `PrefillError::NotActivated` for both
`Pending` and `Untargeted`, and every error is an untyped `anyhow::Error`. The
EPP falls back to aggregated routing on any error, including `Pending`, a
policy-class queue rejection, and "No workers available". During `Pending` this
may send aggregated requests to decode-only workers, which is the failure
ai-dynamo/dynamo#16736 fixed for the frontend. A typed error that separates
these cases would let the EPP fail closed where the frontend does.

## Cross-Use Invariants

### The decode router override is the same in both paths

`build_decode_router_override` (frontend) and `decode_router_config_override`
in `epp.rs` (EPP) both set `assume_kv_reuse` and `track_prefill_tokens` to
`false` and, for ordinary disaggregation, `overlap_score_credit` to `0.0`.
Change them together.

With conditional disaggregation enabled, the frontend leaves
`overlap_score_credit` unset so the base router setting applies. This affects
every request, including those that take remote prefill.

### Every user defines each activation state

A new public entry point adds a column to the state table above and its own
section here, with the tests that guard it.
