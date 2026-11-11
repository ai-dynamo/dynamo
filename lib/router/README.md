<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# lib/router

The Dynamo router, split into small crates. Each stage (indexer, slot tracker, session affinity, worker selection, admission) is its own crate and knows nothing about the others. The scheduler kernel runs them in order for one request and hands each stage's output to the next. `dynamo-kv-router` composes the kernel and the stages into a running router, with every transport adapter behind a feature, and is the crate plugins and most hosts link.

| Crate | What it does | Depends on |
|---|---|---|
| `dynamo-error` | `DynamoError`, `ErrorClass`, the reason catalog, `PublicDetails` | none |
| `dynamo-kv-hashing` | Block, sequence, and tracking hashes shared by engines and the router, plus the multimodal identity both sides compute | dynamo-tokens |
| `dynamo-router-protocol` | Worker and attempt identities and the wire types every router crate uses | none |
| `dynamo-kv-events` | KV event types, the engine ZMQ codec, the replay cursor, residency tiers and owners | protocol, kv-hashing |
| `dynamo-kv-hints` | KV transfer hints the router hands engines: wire format, source choice, worker transfer keys | protocol, kv-hashing, kv-indexer |
| `dynamo-kv-indexer` | KV events in, prefix matches out. Radix trees, the indexer wrappers, LRU retention and pruning, lower-tier indexes, the cuckoo-filter relay index, the service DTOs | protocol, kv-events, kv-hashing, dynamo-tokens |
| `dynamo-slot-tracker` | Bookings per rank in, load and fullness out. Includes its replication wire | protocol, dynamo-tokens |
| `dynamo-session-affinity` | Session binding table, leases, and the replication wire | protocol |
| `dynamo-worker-selection` | Picks one rank from the candidate rows under a policy. Owns the rows and the plugin API | protocol, slot-tracker, session-affinity |
| `dynamo-admission` | Admit, hold, or reject a request now. The per-class queue, the classifier API, `QueueRejection` | protocol, slot-tracker, session-affinity, dynamo-tokens |
| `dynamo-scheduler` | The kernel. Runs the stages for one request, owns the request and response types, eligibility, the queue actor, the selection core, the validated config, the tracking hash | protocol, kv-hashing, kv-events, kv-hints, the five stages, dynamo-tokens |
| `dynamo-kv-router` | Composes a router. Flat config, the YAML policy file, the plugin registry, and the HTTP, ZMQ, replica-mesh, peer-recovery, and Prometheus adapters behind features | scheduler, the five stages, protocol, kv-events, kv-hashing, kv-hints, dynamo-tokens, dynamo-truthy |

`scripts/router_layering_check.sh` enforces the dependency rules and runs in CI. Each crate's `AGENTS.md` states its boundary.
