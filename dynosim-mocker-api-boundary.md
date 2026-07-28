<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AISim/Dynamo API Boundaries

Status: Draft for review

## Boundary map

Only the following contracts cross component boundaries:

| Components | Contract |
| --- | --- |
| `Replayer` ↔ `Generalized Mocker Engine` | engine request, control, and pass effects |
| `Replayer` ↔ `Router Adapter` | `PlacementPolicy<Request>` |
| `Live Mocker` ↔ `Generalized Mocker Engine` | `LiveBoundaryCore` and `LivePassExecution` |
| `Replayer` ↔ `Planner Adapter` | TODO |

The contracts are owned by AISim. Dynamo implements the adapters and converts
between Dynamo-specific and AISim-neutral types.

## Generalized Mocker Engine API

Both `Replayer` and `Live Mocker` drive the same `Generalized Mocker Engine`.
The target component boundary reuses the existing engine operations while
keeping Replayer-owned state, such as `TraceCollector`, out of their
signatures:

```rust
receive(DirectRequest) -> Uuid
apply_command_effects(SchedulerCommand, ...) -> SchedulerCommandEffects
retry_pending_destinations() -> Vec<SchedulerLifecycleEvent>
execute_pass(now_ms) -> EnginePassResult
```

Inputs crossing into the engine are:

- engine-relevant configuration;
- `DirectRequest`;
- `SchedulerCommand`;
- caller-provided simulation time.

`EnginePassResult` returns:

- modeled pass completion time;
- output signals and request completions;
- admissions and lifecycle effects;
- neutral KV observations;
- forward-pass and engine metrics.

The caller owns the clock and execution loop. The engine does not sleep,
advance a global event queue, write a replay report, or publish Dynamo events.
`TraceCollector`, Dynamo `RouterEvent`, runtime handles, and event publishers
do not cross this boundary.

Native G1 management and framework-native G2 offload are internal to the
`Generalized Mocker Engine`. There is no KVBM or KVBM Adapter API in this
design.

## Replayer API

`Replayer` accepts a resolved workload, deployment topology, and engine
configuration, and returns `TraceSimulationReport`.

`Replayer` owns:

- workload admission;
- the virtual clock and event queue;
- worker topology and lifecycle;
- request completion and report collection;
- composition with optional placement and Planner policies.

Engine observations needed by an extension cross the Replayer boundary through
`EngineEventBatch`, `ReplayEngineObservation`, and `Observation::Batch`.

## Router Adapter API

The boundary between `Replayer` and Dynamo `Router Adapter` is:

```rust
PlacementPolicy<Request>
    type Metadata
    type Observation

    place(...) -> PlacementEffects
    observe(...) -> Vec<Placement>

    cancel_pending(...)
    request_terminal(...)
    prefill_completed(...)
    pending_count()

    worker_ready(...)
    worker_draining(...)
    worker_removed(...)
    topology_settled(...)
```

The contract exchanges only these neutral placement types:

- `Placement`;
- `PlacementDecision`;
- `PlacementEffects`;
- `WorkerTopology`.

`PlacementEffects` may decide the current request and release requests
previously queued by a stateful policy.

`AggregatedRoundRobinPlacement`, `PoolRoundRobinPlacement`, and
`KvRouterPlacement` implement the same contract. `KvRouterPlacement` is the
Dynamo `Router Adapter`; it converts production `Router` results and event
batches to the neutral placement and observation types.

## Live Mocker API

The boundary follows the existing `LiveBoundaryCore` and
`LivePassExecution` split. `Live Mocker` submits `DirectRequest`, scheduler
control commands, and the current time to the `Generalized Mocker Engine`. It
receives pass outputs, lifecycle effects, KV observations, FPM, and metrics.

`Live Mocker` owns:

- `DistributedRuntime` and `AsyncEngine` integration;
- Tokio tasks, channels, cancellation, and wall-clock waiting;
- streaming response assembly;
- handoff orchestration;
- conversion and publication of Dynamo events and production metrics.

The dependency is one-way:

```text
Dynamo Live Mocker -> AISim Generalized Mocker Engine
```

The engine does not depend on Dynamo runtime, NATS, ZMQ, handoff sessions, live
event publishers, or the offline `Replayer`.

## Planner Adapter API

TODO: Hongkuan will add this section.
