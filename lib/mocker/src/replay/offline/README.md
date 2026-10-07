<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo Offline Replay Adapters

The deterministic event loop, Generalized Mocker Engine driver, logical-worker
lifecycle, and report collector live in `aisimulate_core::replay`. This directory
contains only Dynamo-owned compatibility entrypoints and Router/Planner
composition:

- `entrypoints.rs` converts existing `MockEngineArgs` and workload inputs into
  a canonical `ReplaySpec`, then calls `aisimulate_core::replay::Replayer`.
- `extensions/kv_router` adapts Dynamo's existing `PlacementPolicy`-based
  Router implementation to the Replay composition boundary.
- `extensions/kv_events` converts neutral engine KV observations into the
  event batch consumed by the Dynamo Router policy.

See the [`aisimulate-core` crate](https://crates.io/crates/aisimulate-core) for the
virtual-time runtime and its liveness contract.

## Reproducibility

`arrival_seed` in the Python replay API controls request arrival generation only.
For Poisson traffic, AISimulate's YAML `traffic.load.seed` is forwarded as this
arrival seed. It does not seed KV router worker selection.

With KV routing enabled, the default router still breaks ties between equal-cost
workers randomly at temperature zero. Identical inputs and arrival seeds can
therefore produce different routing, prefix reuse, and simulated latencies,
despite the deterministic event loop.
