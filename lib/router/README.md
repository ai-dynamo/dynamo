<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# lib/router

The Dynamo router as a set of small crates. Every arrow into a stage starts at the scheduler kernel: stages never message each other, the kernel carries one stage's output to the next as that stage's own input type, and `dynamo-kv-router` composes the result behind features.

## The thirteen crates

| Crate | One sentence | Depends on |
|---|---|---|
| `dynamo-error` | The error contract: `DynamoError`, `ErrorClass::http_status`, reason catalog, `PublicDetails` | serde, serde_json |
| `dynamo-kv-hashing` | The hash contract engines and router share: the block hash identities, block hashes, sequence hashes, tracking-hash context; its `multimodal` module is the multimodal identity both ends of the wire compute (per-block and per-request metadata, the pad and identifier rules, the extra-key marker) | dynamo-tokens, xxhash-rust |
| `dynamo-router-protocol` | The identities and worker-facing vocabulary every crate names | serde, serde_json, thiserror |
| `dynamo-kv-events` | Everything about KV events: types, engine codec, replay cursor, the residency vocabulary (tiers, domains, owners) | protocol, kv-hashing, serde, rmp-serde |
| `dynamo-kv-hints` | KV transfer hints the router hands engines: the wire, the source choice, the worker transfer keys | protocol, kv-hashing, kv-indexer, serde, rustc-hash |
| `dynamo-kv-indexer` | KV events in, prefix knowledge out; owns its service DTOs, identity config, the transfer candidate vocabulary, and the residency owner key | protocol, kv-events, kv-hashing |
| `dynamo-slot-tracker` | Bookings per rank in, load and fullness out; owns its replication wire | protocol, kv-hashing |
| `dynamo-session-affinity` | Session bindings in, constraints out; owns the affinity target and its replication wire | protocol |
| `dynamo-worker-selection` | Pick one rank from a candidate table under a policy; owns the rows and the plugin API | protocol, slot-tracker, session-affinity |
| `dynamo-admission` | Admit, hold, or reject now; owns the queue, the classifier API, `QueueRejection` | protocol, slot-tracker, session-affinity |
| `dynamo-scheduler` | Run the stages atomically for one request; owns the request, eligibility, the hook traits, the conversions, validated config, the one error conversion | protocol, kv-hashing, kv-hints, the five stages, dynamo-error |
| `dynamo-metrics` | Names, factory, collector helper with monotonic baselines | prometheus |
| `dynamo-kv-router` | Compose a router and be the one crate plugins name; flat config; every adapter behind a feature | scheduler, metrics; optional axum, reqwest, zmq, dynamo-kv-events |

On the request path but not router crates: `dynamo-llm` (the frontend host) and `dynamo-runtime` (transport and route pickers; depends on no router crate).
