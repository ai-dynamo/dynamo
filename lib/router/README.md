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
| `dynamo-kv-hashing` | The hash contract engines and router share: block hashes, sequence hashes, tracking-hash context | dynamo-tokens, xxhash-rust, blake3 |
| `dynamo-router-protocol` | The identities and worker-facing vocabulary every crate names | serde, serde_json, thiserror |
| `dynamo-kv-events` | Everything about KV events: types, engine codec, replay cursor | protocol, serde, rmp-serde |
| `dynamo-zmq` | One ZMQ socket helper | zmq, tokio net |
| `dynamo-kv-indexer` | KV events in, prefix knowledge out; owns its service DTOs and identity config | protocol, kv-events, kv-hashing |
| `dynamo-slot-tracker` | Bookings per rank in, load and fullness out; owns its replication wire | protocol, kv-hashing |
| `dynamo-session-affinity` | Session bindings in, constraints out; owns its replication wire | protocol |
| `dynamo-worker-selection` | Pick one rank from a candidate table under a policy; owns the rows and the plugin API | protocol |
| `dynamo-admission` | Admit, hold, or reject now; owns the queue, the classifier API, `QueueRejection` | protocol |
| `dynamo-scheduler` | Run the stages atomically for one request; owns the request, eligibility, the hook traits, the conversions, validated config, the one error conversion | protocol, kv-hashing, the five stages, dynamo-error |
| `dynamo-metrics` | Names, factory, collector helper with monotonic baselines | prometheus |
| `dynamo-kv-router` | Compose a router and be the one crate plugins name; flat config; every adapter behind a feature | scheduler, metrics; optional axum, reqwest, dynamo-zmq, dynamo-kv-events |

On the request path but not router crates: `dynamo-llm` (the frontend host) and `dynamo-runtime` (transport and route pickers; depends on no router crate).

## Status

So far `protocol` exists here, `dynamo-kv-hashing` (`lib/kv-hashing`) holds the router's hash contract in its `routing` module, `dynamo-kv-events` (`lib/kv-events`) holds the event types, the engine ZMQ codec, and the replay cursor, and `dynamo-zmq` (`lib/zmq`) holds the socket helper. The rest arrive in the order of the split plan: the five stages (`kv-indexer`, `slot-tracker`, `session-affinity`, `worker-selection`, `admission`), `dynamo-scheduler`, then `dynamo-kv-router` thinned to composition and the host import rewrite; `dynamo-error` and `dynamo-metrics` land on their own tracks. Until a crate arrives, `lib/kv-router` is the monolith and keeps `pub use` shims for every type that has moved.

## Gate

`scripts/router_layering_check.sh` checks every crate in this directory against the table above: router dependencies by exact set on default features, a forbidden list on `--all-features`, for `protocol` the direct dependency set and a behavior grep, and on the workspace graph that no normal dependency enables a `testing`, `bench`, or `profile` feature of a router crate. CI runs it with `--host`, which also resolves `dynamo-llm`, `dynamo-ext-proc`, and `dynamo-mocker` and fails if any router crate reaches `prometheus`, `axum`, `hyper`, `reqwest`, or `zmq` through a host's feature unification. `scripts/router_move_diff.py` proves a move PR moved text without changing it.
