<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Prefill routing and conditional disaggregation

`PrefillRouter` coordinates an optional remote prefill hop before decode. P denotes a prefill worker; D denotes a decode worker. In aggregated execution, one worker performs both phases.

## How a discovered endpoint becomes a router

Runtime EPP creates `PrefillRouter` with a one-shot receiver. Its discovery task finds a prefill-serving endpoint in the selected namespace and sends that endpoint through the channel.

The activation task consumes it:

```rust
if let (Ok(endpoint), Some(router)) = (result, router.upgrade()) {
    router.set_target(Some(WorkerSetTarget::Legacy(endpoint)));
}
```

A target-driving task builds and publishes a `PrefillBinding`. It creates an endpoint client, starts the runtime-config watcher through `ModelManager`, reads the prefill model cards, resolves routing mode and configuration, starts load tracking, and constructs a `RoutingHost`. Only after publishing that binding does the router become active.

The endpoint denotes a namespace/component/endpoint worker pool, not one selected worker. Its client follows workers joining and leaving that pool. EPP's one-shot endpoint discovery chooses the pool once; moving to a different endpoint or namespace requires rebinding. Continuous worker membership and continuous pool discovery are different responsibilities.

Sources: [activation receiver](activation.rs), [binding construction](activation.rs), [EPP prefill discovery](../../../../../deploy/inference-gateway/ext-proc/src/epp.rs), and [live membership/configuration watch](../../discovery/runtime_configs.rs).

## Reserving P without executing P

`reserve_prefill_worker` requires an active binding and takes the reservation ID,
prompt tokens, and routing hints. For a KV-routed pool, it calls
`find_best_match_details_with_policy_class_admitted` with `update_states=true`.
The result contains the selected worker/rank and an armed `BookingHandle`, which
is retained inside `PrefillReservation`.

Dropping the reservation enqueues cleanup of that exact scheduler attempt;
explicit release waits for acknowledgement. This is the runtime EPP path.
The reservation call does not execute prefill.

For built-in non-KV routing modes, this method peeks a worker and returns an untracked reservation. Do not claim every mode waits for KV admission or owns a booking.

Source: [prefill reservation API](query.rs).

## Conditional disaggregation in the integrated frontend

The integrated frontend can decide that a cache-hot D should perform prefill locally:

1. Preview one D route with `preview_kv_route(..., RequestPhase::Decode)`. This is advisory and holds no capacity.
2. Give its cache signals to the conditional-disaggregation policy. When configured, inspect the chosen P's load and apply the D-load gate too.
3. If the policy chooses local prefill on D, call `plan_kv_route_from_preview`. This performs admission and booking for the previewed D target.
4. Dispatch that admitted plan and mark the request as bypassing remote prefill.
5. If the policy declines bypass, proceed through the remote P path and the subsequent D routing path.

A successful preview does not mean “remote prefill is unnecessary” or “D is admitted.” The policy decision and admitted plan are separate steps. Preview returns one selected worker/rank, not all candidates, and selection follows configured policy rather than a maximum-cache-hit guarantee.

Admission can still wait or fail because preview held no reservation. The selected D can disappear, capacity can change, or the request can be cancelled. The outer prefill coordinator falls back to remote prefill for decision/planning errors except cancellation and invalid arguments, which propagate. This is caller policy, not a universal rule that all preview failures imply remote prefill.

An explicit P pin skips this conditional decision. Non-KV decode routing, missing decode host, and empty routing tokens also bypass the conditional path.

Sources: [preview, topology policy, and admitted plan](conditional_bypass.rs) and [dispatch and error fallback](mod.rs).

## Runtime EPP uses a different orchestration order

Runtime EPP first attempts to reserve P, then queries D through admission without booking. After resolving the HTTP destination, it separately calls `add_request` to register D load. P can remain booked while D waits. Neither the D query/booking pair nor the full P/D operation is an atomic reservation transaction.

If P selection fails, this EPP path falls back to aggregated routing. If D selection fails, the locally held P reservation is dropped and cleaned up. EPP does not invoke the conditional D-preview policy in this pick path. When it succeeds with P, it forwards a P pin; the worker-side prefill router sees that pin and skips the conditional decision.

The downstream Dynamo worker frontend performs the prefill/bootstrap/KV-transfer workflow. EPP selects workers and tracks their load; the downstream frontend runs the selected topology. See the [runtime EPP lifecycle guide](../../../../../deploy/inference-gateway/ext-proc/README.md) for callback ownership.

D registration errors are logged and do not fail the pick, so returning a destination does not guarantee that decode bookkeeping was installed.

Sources: [P-then-D selection](../../../../../deploy/inference-gateway/ext-proc/src/epp.rs) and [pinned P and prefill execution](mod.rs).

## What prefill completion on D means

It means an update to D's router-side request accounting. `mark_prefill_completed` marks the phase complete, decrements the prefill-request count, and removes tracked prompt load. It retains the request's decode booking.

For aggregated execution, D performed the prefill, so removing that load is meaningful. For disaggregated runtime EPP, D is booked with `track_prefill_tokens=false`; there is no prompt-token load to remove, but phase accounting still advances. The `KvRouter` wrapper also touches the request lease. P's reservation is released separately.

This call does not send a command to the D engine or prove that a backend completed prefill. Runtime EPP infers the transition from its first nonempty response-body chunk.

Sources: [phase/load mutation](../../../../kv-router/src/sequences/single.rs), [lease touch](../../kv_router.rs), and [PD decode accounting override](../../../../../deploy/inference-gateway/ext-proc/src/epp.rs).
