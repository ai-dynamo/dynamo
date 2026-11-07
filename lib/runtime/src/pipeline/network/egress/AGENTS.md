<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# lib/runtime/src/pipeline/network/egress

This module sends requests to worker instances on the request plane. Read
[`README.md`](README.md) before you change `PushRouter`. It explains the two uses of
`PushRouter`, which component calls each public function, and what the direct engine lacks
compared with `RoutingHost`.

## Rules

- When `RoutingHost` (`lib/llm/src/kv_router/routing_host.rs`) uses `PushRouter`, `PushRouter`
  only dispatches: address lookup, transport fallback and fault detection. `RoutingHost` owns
  worker selection, occupancy booking, affinity, LoRA filtering, Direct mode and the request
  lifecycle (`RequestGuard`). Do not move any of these into `PushRouter`.
- Add new selection logic as a worker-selection policy that `RoutingHost` runs; the README
  links the guide for writing one. Add a variant to `RoutingHost`'s `RoutingPolicy` enum
  (`routing_host.rs`) only when the request lifecycle itself differs. Do not add a `RouterMode` selection path to `PushRouter`,
  and do not add selection logic to the direct engine's mode methods.
- The direct engine (`AsyncEngine::generate`, `round_robin`, `random`, `power_of_two_choices`,
  `least_loaded`, `device_aware_weighted`) is for callers with generic `T`/`U` types. Keep its
  behavior stable. The Python bindings call these methods by name.
- A `*_prepared` callback receives the worker that will receive the request. Move everything
  the caller tied to the selected worker (occupancy booking, DP rank, request fields) in that
  callback. The README's "The `prepare` callback" section gives the full contract.
- When you add, remove or repurpose a public function, update the use-case tables in
  `README.md`.

## Transport fallback

`resolve_transport` applies the caller's `TransportFallback` when the selected worker has left
discovery:

- `Allow`: send to any other free worker. Used by the direct-engine modes, the bidirectional
  engine, and `RoutingHost` RoundRobin, Random and Direct (`direct_within_prepared(.., None)`)
  and the occupancy modes (`dispatch_preselected_prepared`). The occupancy modes are
  PowerOfTwoChoices, LeastLoaded and DeviceAwareWeighted, which count each worker's in-flight
  requests; see the README.
- `Within(set)`: send only to another free worker in `set`. Used by `RoutingHost` LoRA
  routing (`direct_within_prepared(.., Some(set))`).
- `Deny`: fail with `CannotConnect`. Used by `direct`, `dispatch_exact` and
  `dispatch_kv_admitted`.

With no permitted worker free, fallback fails with `Unavailable`. Keep the `CannotConnect`
and `Unavailable` error types distinct: migration and HTTP status mapping depend on them.

## Testing

Unit tests are at the end of `push_router.rs`:

```bash
cargo test -p dynamo-runtime --lib pipeline::network::egress::push_router
```

End-to-end tests are in `lib/runtime/tests/` (for example `bidirectional_e2e.rs` and
`response_stream_establish_timeout.rs`). For changes that affect `RoutingHost` dispatch, also
run `cargo test -p dynamo-llm --no-default-features --lib kv_router::routing_host`.
