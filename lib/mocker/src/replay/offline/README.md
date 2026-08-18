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

See [`aisimulate-core`](https://github.com/ai-dynamo/aisimulate/tree/main/crates/core) for
the virtual-time runtime and its liveness contract.
