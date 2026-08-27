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

## Replica-local router views

Offline KV-router replay normally uses one router view. Set
`MockEngineArgs.router_replicas` above one to model requests arriving through
multiple frontend/router replicas. Requests are assigned round-robin from the
configured `router_replica_seed`; every replica owns independent cache and
active-load state.

`KvRouterConfig.router_replica_sync` gates only active-sequence lifecycle copies
to peer views. Independently, `router_replica_sync_delivery_rate` controls
deterministic best-effort delivery to each peer and to each router's KV-event
view, including a single replica. A value of `1.0` converges the views; lower
values intentionally leave measurable stale state. A stored child is withheld
when its parent was missed, so simulated loss cannot create an impossible cache
chain. Replay logs cumulative attempted, delivered, and dropped event counters
at debug level. The delivery model does not infer an unobservable production
transport delay; it is a sensitivity control for inconsistent router state.

See the [`aisimulate-core` crate](https://crates.io/crates/aisimulate-core) for the
virtual-time runtime and its liveness contract.
