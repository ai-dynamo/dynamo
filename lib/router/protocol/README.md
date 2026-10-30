<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-router-protocol

The identities and worker-facing vocabulary every Dynamo router crate names.

## Dependency contract

`[dependencies]` is exactly `serde`, `serde_json`, and `thiserror`, and nothing else: no `dynamo-*` crate, no async runtime, no logging, no builder macros. Test-only crates go under `[dev-dependencies]`. The crate is data: struct and enum definitions, their `serde` derives, `Display`/`FromStr`/`From` conversions, and the error types those conversions return.

`scripts/router_layering_check.sh` enforces this. It asserts the direct dependency set by equality, forbids `dynamo-runtime`, `dynamo-kv-router`, `tokio`, `anyhow`, `tracing`, `async-trait`, `derive_builder`, `prometheus`, `axum`, `hyper`, `reqwest`, and `zmq` anywhere in the `--all-features` closure, and greps `src/` for `dyn `, `async `, `std::fs`, `anyhow`, `tracing`, `derive_builder`, and `#[doc(hidden)] pub`. A line that must carry one of those tokens as prose ends with `// layering: allow`; the gate prints every exempt line.

## Entry rule

A type enters protocol only if it is serialized across a process boundary to an engine, a worker, or a sidecar, or is an identity every crate names.

Everything else belongs to the crate that produces it. When a second crate needs such a type, it depends on the producer; it does not lift the type here.

## What is deliberately not here

| Concept | Lives in |
|---|---|
| KV cache events, `RouterEvent` and its builders, the engine codec, the replay cursor, placement helpers | `dynamo-kv-events` |
| Block and sequence hashing, the tracking-hash context | `dynamo-kv-hashing` |
| The router request and response, routing constraints, config overrides, eligibility, the hook traits the kernel calls | `dynamo-scheduler` |
| Candidate rows and views, the policy plugin API | `dynamo-worker-selection` |
| `QueueRejection`, `QueueLimitKind`, `WorkerPlacement`, the classifier API | `dynamo-admission` |
| Replica events and booking descriptors (the slot-tracker replication wire) | `dynamo-slot-tracker` |
| Indexer service DTOs and indexer identity configuration | `dynamo-kv-indexer` |

## `AttemptId` and `AttemptIdMinter`

`AttemptId` has no public constructor. `AttemptIdMinter::mint` is the only source of a value and yields fresh ids only; the slot tracker holds the one instance and mints at booking. Every other crate receives an `AttemptId` on a replica event, a booking descriptor, or an indexer lease and carries it unchanged. There is no `AttemptId::new`, no `From<u64>`, and no `#[doc(hidden)]` escape. The chosen-value constructors `AttemptId::for_testing` and `AttemptId::for_benchmark` exist only behind the `testing` and `bench` features; no normal dependency may enable either, and the gate fails if one does.

## Checking a change

```bash
scripts/router_layering_check.sh          # dependency set, closure, behavior grep
scripts/router_layering_check.sh --host   # plus host-resolved graphs, as CI runs it
```
