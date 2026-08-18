<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AISimulate Core

`aisimulate-core` is the public Rust package for engine-neutral inference
simulation. It preserves two explicit implementation and API layers:

- `aisimulate_core::engine` owns scheduling, native GPU KV accounting,
  preemption, timing, and attention-DP composition.
- `aisimulate_core::replay` owns deterministic virtual time, logical-worker
  lifecycle, placement and scaling composition, and replay reports.

The crate root promotes the common `EngineConfig`, `ReplaySpec`, `Replayer`,
canonical `ReplayReport`, and timing-provider contracts. Dynamo-specific Router, Planner,
transport, and live-runtime adapters remain outside this crate.
