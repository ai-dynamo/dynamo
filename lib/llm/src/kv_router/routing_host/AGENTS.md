<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Routing Host Rules

Read `README.md` in this directory before adding a `RoutingHost` entry point, a routing mode,
or a caller that plans routes. It describes how each plane is driven and what they share. The
parent [`AGENTS.md`](../AGENTS.md) guardrails apply here too.

## Invariants

- **Plan, then dispatch, through one owner.** Every route is selected with a plan
  (`KvRoutePlan` or `BuiltinRoutePlan`, wrapped in `RoutePlan`) before it is sent. Until
  dispatch, the plan is the only owner of the booking, reservation, and affinity hold; at
  dispatch, the `RequestGuard` takes over. Do not add a path that selects and sends without
  a plan. Guarded by `builtin_route_plan_holds_its_reservation_until_aborted` and
  `planned_dispatch_transfers_the_reservation_to_request_cleanup`.
- **A KV send is exact.** Never retarget a KV guard. Only the built-in plane uses transport
  fallback.
- **`prepare` runs once, with the final worker.** A built-in send can move to another
  instance, so do not prepare a request from `RoutePlan::target()` ahead of the send. Guarded
  by `builtin_route_plan_prepares_for_the_worker_it_is_sent_to`.
- **A failed `prepare` gives the route up.** For KV, that happens before the request counts as
  started. Guarded by `prefill_prepare_failure_releases_the_booking_before_the_request_counts`.
- **Release an undispatched plan with `abort`** when the caller can await. It waits for the
  scheduler's acknowledgement and records the cause. Dropping the plan is the fallback, not
  the normal path.
- **Validate request-supplied worker pins at selection only.** A worker that disappears later
  is a service failure.
- **A preview never books.** `preview_kv_route` and `peek_next_worker` hold nothing.
- **Only Decode and Aggregated KV routes start the classifier lifecycle.** A prefill route
  must leave a parked decode lifecycle alone.

## Changing Behavior

- When you add a routing mode, decide its plane and add it to the README's tables: builder
  inputs, what its plan holds, and how it sends.
- When you add an entry point, add it to that plane's table in the README, with what it holds
  and when it releases it.
- Keep the README's "same" and "different" sections in step with any change to either plane.

## Testing

Tests for both planes live in `tests.rs`. On macOS, run them without the default features:

```bash
cargo test -p dynamo-llm --no-default-features --lib kv_router::routing_host
```
