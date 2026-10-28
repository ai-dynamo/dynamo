<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Prefill Router

`PrefillRouter` is the frontend pipeline stage between the encoder and the
decode `RoutingHost`. For disaggregated serving it runs each request on a
prefill worker first, then forwards it to decode with the KV handoff attached.
With conditional disaggregation enabled, it may instead keep a request on its
decode worker and skip the prefill hop.

The step-by-step order of one request lives in the doc comment on
`PrefillRouter::generate` in `mod.rs`. This file records the module layout and
why the order and the fallbacks are what they are.

## Files

- `mod.rs`: the `PrefillRouter` operator, `generate`, and decode-request
  construction (`build_decode_request`, `build_decode_router_override`).
- `activation.rs`: target tracking, binding construction, and the
  `ActivationSnapshot` (`Active`, `Pending`, `Untargeted`).
- `conditional_bypass.rs`: the conditional-disaggregation decision, built on a
  non-admitting decode route preview.
- `admission.rs`: drains the prefill stream into a handoff, a terminal output,
  or an error.
- `handoff.rs`: keeps consuming prefill while decode runs and merges prefill
  failures into the decode stream.
- `query.rs`: advisory prefill lookups and reservations that do not dispatch.

## Activation States

| State | Meaning | `generate` does |
|---|---|---|
| `Active` | A prefill target is set and its binding is built. | Runs the prefill hop. |
| `Pending` | A target is set but the binding is still building. | Fails closed with a migratable `WorkerUnavailable` (HTTP 503). |
| `Untargeted` | No target was ever set, or every prefill worker left. | Forwards to decode unchanged. |

The hot path reads the lifecycle state with one atomic load and takes the
target lock only when the state is not `Active`.

## Design Decisions

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

It deliberately did not gate model admission on the prefill lifecycle: worker
topology gates selection independently of this router, and the `Untargeted`
pass-through is intentional for aggregated-capable backends.

### The binding is checked again after the Prefill phase barrier

`set_target` with a different target clears the binding and moves the router
to `Pending`. A request that passed the gate can reach the binding load after
that, while it awaited conditional planning or the phase barrier. The second
check resolves from a fresh snapshot with the same rules as the gate. It was
added in review of ai-dynamo/dynamo#16736.

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

### The decode router override mirrors the EPP

`build_decode_router_override` sets `assume_kv_reuse` and
`track_prefill_tokens` to `false` and, for ordinary disaggregation,
`overlap_score_credit` to `0.0`. The EPP sets the same values in
`decode_router_config_override` in `deploy/inference-gateway/ext-proc/src/epp.rs`.

With conditional disaggregation enabled, `overlap_score_credit` is left unset
so the base router setting applies. This affects every request, including
those that take remote prefill.

### The prefill hop does not start the classifier lifecycle

KV selection returns early for `RequestPhase::Prefill`
(`routing_host/kv.rs`). Only the decode leg starts the request-classifier
lifecycle.
