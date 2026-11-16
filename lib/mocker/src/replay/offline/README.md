<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo Offline Replay Adapters

The deterministic event loop, Generalized Mocker Engine driver, logical-worker
lifecycle, and report collector live in `aisimulate_core::replay`. This directory
contains only Dynamo-owned compatibility entrypoints and Router/Planner
composition:

- `entrypoints.rs` converts canonical `MockerConfig` and workload inputs into
  a canonical `ReplaySpec`, then calls `aisimulate_core::replay::Replayer`.
- `extensions/kv_router` adapts Dynamo's existing `PlacementPolicy`-based
  Router implementation to the Replay composition boundary.
- `extensions/kv_events` converts neutral engine KV observations into the
  event batch consumed by the Dynamo Router policy.

See the [`aisimulate-core` crate](https://crates.io/crates/aisimulate-core) for the
virtual-time runtime and its liveness contract.

## Worker-selection policies

A router config that selects a worker-selection policy (`router_policy_config`, or
`DYN_ROUTER_WORKER_SELECTION_POLICY`) runs that policy in replay. Each worker pool resolves the
policy selected for its role (`aggregated`, `prefill` or `decode`) from the registry in
`ReplayScope::router_plugins`. The Python bindings pass the registry their live router uses, so
a linked catalog is available too; without one, replay offers the policies Dynamo ships.

A configured policy is built by its registered factory, as for a live router. The replay seed
does not reach it, and a policy that reads the clock reads wall time, not replay time.
`request_classifier` plugins are not supported.

## Reproducibility

`arrival_seed` in the Python replay API controls request arrival generation only.
For Poisson traffic, AISimulate's YAML `traffic.load.seed` is forwarded as this
arrival seed. It does not seed KV router worker selection.

With KV routing enabled, the default router still breaks ties between equal-cost
workers randomly at temperature zero. Identical inputs and arrival seeds can
therefore produce different routing, prefix reuse, and simulated latencies,
despite the deterministic event loop.
