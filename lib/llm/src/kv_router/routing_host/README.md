<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Routing Host

`RoutingHost` routes requests to one worker set. It selects a worker, holds what that
selection reserved, sends the request, and tracks the response until everything is released.
One type serves two routing planes:

- **KV plane:** `RouterMode::KV`. A `KvRouter` scores workers by cached KV blocks and load,
  and books the chosen worker in its scheduler.
- **Built-in plane:** every other mode: `RoundRobin`, `Random`, `PowerOfTwoChoices`,
  `LeastLoaded`, `DeviceAwareWeighted`, and `Direct`. A first-party selector picks the worker
  without KV state.

This file explains how a caller drives each plane and what the two share. For where hosts sit
in the frontend pipeline, the KV booking lifecycle, and cancellation, see the
[router host layer README](../README.md). The rules for changing this code are in
[`AGENTS.md`](AGENTS.md).

## Files

- `../routing_host.rs`: `RoutingHost`, `RoutePlan`, `KvRoutePlan`, `RoutePreview`, the
  `AsyncEngine` impl (`generate`), and the shared dispatch helpers (`accept_dispatch`,
  response monitoring).
- `builder.rs`: `RoutingHostBuilder`, which checks the inputs against the routing mode.
- `kv.rs`, `kv_selection.rs`: the KV plane: preview, plan, track, and dispatch.
- `builtin.rs`, `occupancy.rs`: the built-in plane: selectors, `BuiltinRoutePlan`, its plan and
  dispatch, and in-flight occupancy for the load-based modes.
- `request_guard.rs`: `RequestGuard`, which tracks a sent request and releases what it holds.
- `cancellation.rs`: the cleanup budget and what a client disconnect may cancel.
- `tests.rs`: tests for both planes.

## Building a host

`RoutingHost::builder(client, router_mode)`, then `build()`. The builder rejects inputs that
do not fit the mode.

| Input | KV plane | Built-in plane |
|---|---|---|
| `kv_router` | Required | Rejected |
| `load_context` | Optional (the `KvRouter` owns the client) | Required |
| `affinity_coordinator` | Optional | Optional |
| `lora` | Rejected. The `KvRouter` takes the request's LoRA name as a selection input instead | `RoundRobin` and `Random` only, and not together with session affinity |
| `multimodal_cache` | Not used | `DeviceAwareWeighted` |
| `dispatch` | Optional: sends through the given hop instead of the request plane, for a host that does not send requests itself, such as a gateway endpoint picker, or a test | Same |

## Driving a KV host

| Entry point | Use it for | What it holds and releases |
|---|---|---|
| `generate` | The pipeline's terminal engine: select, book, send. | The response stream releases the booking when it ends. |
| `generate` with `query_instance_id` | An advisory lookup. | Nothing: it selects with `Query` admission and returns the selection as `routing_data` without sending. |
| `plan_route`, then `dispatch_route` | A stage router that books before it sends, such as the prefill hop. | The `RoutePlan` owns the booking until `dispatch_route` moves it into the request guard, or until `abort`. |
| `preview_kv_route`, then `plan_kv_route_from_preview`, then `dispatch_kv_plan` | Conditional disaggregation: decide from a decode preview before booking anything. | A `RoutePreview` holds nothing. The plan made from it books the previewed worker. |
| `prefill_worker_busy` | The conditional-disaggregation prefill-load probe. | Nothing. |
| `kv_router()` | Direct `KvRouter` queries, such as the EPP's prefill reservation. | Whatever the caller books there. |

How a KV route runs:

1. **Plan.** Validate request-supplied worker pins. Acquire the session-affinity hold. For a
   Decode or Aggregated request, start or resume the request's classifier lifecycle. Then
   select and book the worker with `Lease` admission. The `KvRoutePlan` now owns the booking,
   the affinity hold, and the classifier lifecycle.
2. **Prepare** (through `dispatch_route` only). `prepare` runs once, with the booked worker. If
   it fails, the plan is aborted and the request never counts as started.
3. **Dispatch.** The request guard takes over the booking and records the routing decision. The
   request is sent to the booked worker exactly; there is no transport fallback.
4. **Release.** Stream end, an error, or a drop releases the booking. `attempt_release()`
   gives an early-release handle, which Migration uses to free a prefill booking before it
   retries.

## Driving a built-in host

| Entry point | Use it for | What it holds and releases |
|---|---|---|
| `generate` | The pipeline's terminal engine: select, send. | The response stream releases the reservation when it ends. |
| `generate` with `query_instance_id` | Nothing different: the built-in plane has no query path. The request is selected and **sent**. The session-affinity lookup does not acquire a hold, so the session is not bound. | As above. |
| `plan_route`, then `dispatch_route` | A stage router that selects before it sends, such as the prefill hop. | The `RoutePlan` owns the guard and the affinity hold until dispatch, or until `abort`. |
| `peek_next_worker` | An advisory pick, such as the EPP's built-in prefill reservation. | Nothing: it does not advance the round-robin cursor or reserve occupancy. |

How a built-in route runs:

1. **Plan.** In Direct mode, require an explicit worker or a session-affinity target. Pick a
   LoRA-filtered target when LoRA routing applies. Otherwise validate request-supplied worker
   pins, acquire the session-affinity hold, and select: round-robin advances its cursor, and
   the load-based modes reserve occupancy on the chosen worker. The request guard is created
   here, so the request counts as started at plan time. The `BuiltinRoutePlan` owns the guard,
   with its occupancy reservation and LoRA load, and the affinity hold.
2. **Dispatch.** One of four send arms runs:
   - Direct mode with no affinity session sends to the explicit target, with fallback.
   - A pinned target is sent exactly.
   - A load-based mode sends to its preselected worker, with fallback.
   - Any other mode sends to the chosen worker, with fallback limited to the LoRA candidates
     when LoRA routing applies.

   With fallback, `PushRouter` moves the request to another instance if the chosen one is gone.
   It then calls the prepare callback with the new worker, which moves the guard and its
   occupancy reservation there (`retarget_worker`), and drops the planned DP rank.
3. **Prepare.** `prepare` runs inside the send, with the worker the request is actually sent
   to. For a pinned target it runs just before the send.
4. **Release.** Stream end, an error, or a drop releases the reservation and the LoRA load.
   There is no scheduler booking, so `attempt_release()` returns `None`.

## What is the same

- **One interface.** Both planes are the same `RoutingHost` type and the same `AsyncEngine`.
  `plan_route`, `dispatch_route`, and `RoutePlan` (`target`, `attempt_release`, `abort`) work
  for either, so a stage router does not branch on the routing mode.
- **Plan, then dispatch.** Every route is selected first and sent second. Between the two, the
  `RoutePlan` is the only owner of what the selection holds. At dispatch, a `RequestGuard`
  becomes the owner.
- **The `prepare` contract.** `prepare` runs exactly once, with the worker the request is sent
  to, and a `prepare` failure gives the route up.
- **Worker pins.** Request-supplied worker IDs are validated at selection only. A pinned worker
  that disappears after selection is a service failure, not a client error.
- **Session affinity.** Hard mode pins the session's worker; soft mode prefers it. The session
  is bound to the worker once the response stream starts. If dispatch fails under hard affinity
  and the target is no longer valid, the binding is invalidated.
- **Sending and tracking.** Both planes use the same cleanup budget and staged-KV rule for a
  client that disconnects mid-route, and the same response monitoring. Both record the failed
  worker for Migration, and accept the dispatch the same way.
- **Giving a plan up.** `abort` releases what the plan holds. Dropping a plan also releases
  it, without waiting for acknowledgement.

## What is different

| | KV plane | Built-in plane |
|---|---|---|
| Selection reads | KV block overlap, scheduler load, and the request's selection profile | Discovery membership, and occupancy for the load-based modes |
| Held between plan and dispatch | A scheduler booking (`KvRoutePlan`) | A request guard with any occupancy reservation and LoRA load (`BuiltinRoutePlan`) |
| Request counts as started | At dispatch | At plan |
| Send | Exactly to the booked worker | May move to another instance; `prepare` sees the final worker |
| `RoutePlan::target()` | The worker the request will reach | The planned worker, which the send may change |
| `query_instance_id` | Returns the selection without booking or sending | No effect on sending |
| Non-booking preview | `preview_kv_route` | None. `peek_next_worker` is the advisory pick |
| Classifier lifecycle | Started for Decode and Aggregated requests | None |
| Early release for Migration | `attempt_release()` | None |
| `abort` compared with drop | `abort` waits for the scheduler to acknowledge the release and passes the cause to the classifier. Drop queues the release | Same release either way |
