<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KV route planning and request lifetime

`RoutingHost` owns worker selection, backend dispatch, and response cleanup.
The shared selection algorithm lives in `lib/kv-router`, reached through
`KvRouter` and `EmbeddedSelection`. Runtime EPP also uses `KvRouter`, but its
decode path queries and books separately rather than dispatching through a host.

## Choose the operation by its effects

| Operation | Admission | Booking | Dispatch | Result |
|---|---|---|---|---|
| `preview_kv_route` | Skipped | None | None | One `RoutePreview`: worker/rank and cache/load signals |
| `plan_kv_route_from_preview` | Yes, for the previewed target | Owned by plan | None | `RoutePlan`, or failure |
| `dispatch_kv_plan` | Uses the existing plan | Transferred into request cleanup | Yes | Backend response stream |
| KV `RoutingHost::generate`, ordinary request | Yes | Owned by request cleanup | Yes | Backend response stream |
| KV `RoutingHost::generate`, `query_instance_id` | Yes | None | None | One routing-data response |

“Select” names an operation, not one ownership contract. Standalone `/select`
does not book; the admitted selection used by ordinary generation does.
Read the exact API and admission mode before assuming either behavior.

Sources: [preview, plan, and dispatch](routing_host/kv.rs),
[generate](routing_host.rs), and the
[selection-service modes](../../../kv-router/src/services/selection/README.md).

## RequestPhase identifies the routing hop

The routing `RequestPhase` has `Prefill`, `Decode`, and `Aggregated` variants.
It determines phase-specific routing and timing records and helps select the
relevant worker pin and lifecycle handling. It does not reserve capacity or
prove that backend computation has finished. A decode route can execute
prefill locally when remote prefill is bypassed.

Decode and aggregated selection own the logical request's classifier lifecycle.
The prefill hop leaves that lifecycle to the decode owner. Another module's
classifier state enum with the same name is a different type.

Sources: [RequestPhase](../protocols/common/timing.rs) and
[classifier lifecycle ownership](routing_host/kv.rs).

## Preview is an observation

A successful preview returns one chosen worker and data-parallel rank, not a
candidate list. Signals include overlap blocks, cached tokens, potential decode
blocks, and optional total KV capacity. The configured policy, worker constraints,
and affinity handling determine the choice; it is not a maximum-raw-cache-hit
contract.

Preview bypasses queue admission. Worker eligibility still matters, but preview
does not establish that the request is admittable or reserve capacity. A worker
can leave, another request can consume capacity, or the request can be cancelled
before the later admission completes.

`plan_kv_route_from_preview` checks the request ID and supplies the previewed
worker as the planned target to admitted selection. It carries forward the same
cleanup budget. A successful plan is the commitment point.

Making successful preview guarantee a booking would change its ownership and
cancellation contract. Such an API would need to return an owned plan or booking.
The current [conditional-prefill caller](prefill_router/README.md) deliberately
makes its topology decision before that commitment.

## Booking ownership

There are three relevant owners:

- A scheduler `BookingHandle` guards an exact admitted attempt. Its caller must
  retain, release, or explicitly transfer that responsibility.
- A `RoutePlan` owns `KvRequestCleanup` while conditional decode awaits dispatch.
  Dispatch transfers cleanup into the request guard. Dropping an unused plan
  releases its booking.
- Public request-ID admissions and `KvRouter::add_request` can install detached
  request leases. `KvRouter::free(id)` or expiry releases that state.

A detached request lease does not provide the dispatching host's response
progress tracking. It needs an external caller to report phase transitions and
completion. Runtime EPP uses its [callback path](../../../../deploy/inference-gateway/ext-proc/README.md)
for that purpose.

The selection core's `Book` mode instead retains a reservation indexed by
`selection_id`; HTTP callers send lifecycle operations. That service-owned
reservation and a caller-owned `Lease` are explained in the
[selection guide](../../../kv-router/src/services/selection/README.md).

Sources: [RoutePlan](routing_host.rs), [request cleanup](routing_host/request_guard.rs),
[request leases](request_lease.rs), and [public KvRouter admissions](../kv_router.rs).

## Ordinary generation and conditional planning

Ordinary KV generation calls `select_with_affinity`, installs tracking with
`track_selection`, and calls `dispatch_selection`. Conditional disaggregation
uses the explicit preview → plan-from-preview → dispatch-plan sequence. The
prefill execution hop has its own `select_and_dispatch_kv_prefill` helper.

The dispatched path installs `kv_hint`, handles affinity, tracks generated output
blocks and approximate cache state, advances classifier state, and cleans up the
stream. Selection alone does not perform these effects.

The current branch does not expose a general `plan_kv_route`, `RouteRequest`,
`RouteBooking`, `RoutePlan::into_booking`, or `RoutePlan::detach_booking` API.
A generalized schedule-only design could share an owned plan between frontend
execution and EPP, but it must define those ownership transfers rather than
assuming the narrower conditional plan already serves every caller. See the
[frontend integration discussion](../README.md).

## PushRouter construction

`PushRouter::from_client_with_dispatch` accepts an injected dispatch implementation.
Its repository call sites are tests, including routing-host, prefill, and runtime
response-stream tests. Production construction uses `from_client` or
`from_client_with_monitor`. Public visibility alone does not establish production
use, and repository search cannot rule out external consumers.

Source: [PushRouter constructors](../../../runtime/src/pipeline/network/egress/push_router.rs).

## Module map

- The sibling [kv_router.rs](../kv_router.rs) prepares cache and scheduling inputs;
  it does not forward model requests.
- [embedded.rs](embedded.rs) connects discovery and load to the shared selection core.
- `indexer/` and `publisher/` connect cache lookup, KV events, and worker metrics.
- [kv_selection.rs](routing_host/kv_selection.rs) prepares selection inputs;
  [kv.rs](routing_host/kv.rs) implements KV preview, planning, tracking, and dispatch.
- [request_guard.rs](routing_host/request_guard.rs) tracks response progress and cleanup;
  [cancellation.rs](routing_host/cancellation.rs) handles unfinished work on cancellation.
- [prefill_router](prefill_router/README.md) coordinates remote prefill;
  [encoder_router.rs](encoder_router.rs) handles optional multimodal encoding.
