<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo Offline Replay Adapters

The deterministic event loop, Generalized Mocker Engine driver, logical-worker
lifecycle, and report collector live in `aisim_core::replay`. This directory
contains only Dynamo-owned compatibility entrypoints and Router/Planner
composition:

- `entrypoints.rs` converts canonical `MockerConfig` and workload inputs into
  a canonical `ReplaySpec`, then calls `aisim_core::replay::Replayer`.
- `extensions/kv_router` adapts Dynamo's existing `PlacementPolicy`-based
  Router implementation to the Replay composition boundary.
- `extensions/kv_events` converts neutral engine KV observations into the
  event batch consumed by the Dynamo Router policy.

See the [`aisim-core` crate](https://crates.io/crates/aisim-core) for the
virtual-time runtime and its liveness contract.

## Worker-selection policies

Each worker pool runs the worker-selection policy the router config selects for its role,
resolved from `ReplayScope::router_plugins` (the policies Dynamo ships when unset). The policy
is built by its registered factory, so the replay seed does not reach it and a policy that
reads the clock reads wall time. Conversation affinity pins a session's worker under every
policy. `request_classifier` plugins are not supported.

## Reproducibility

`arrival_seed` in the Python replay API controls request arrival generation only.
For Poisson traffic, AISimulate's YAML `traffic.load.seed` is forwarded as this
arrival seed. It does not seed KV router worker selection.

With KV routing enabled, the default router still breaks ties between equal-cost
workers randomly at temperature zero. Identical inputs and arrival seeds can
therefore produce different routing, prefix reuse, and simulated latencies,
despite the deterministic event loop.
