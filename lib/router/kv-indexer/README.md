<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-kv-indexer

KV events in, prefix knowledge out. The router's indexer stage, now complete. It holds the radix trees (`RadixTree`, `PositionalIndexer`, `ConcurrentRadixTreeCompressed`), the `ThreadPoolIndexer` and `BranchShardedIndexer` wrappers, `KvIndexer` and `LocalKvIndexer`, approximate LRU retention and pruning, the lower-tier continuation index and per-tier registry, the cuckoo-filter pool index, the hash lifecycle delegate, the indexer counters, the residency projection, overlap scores, KV transfer candidates, the ingest backend (`backend`, `lookup`, `recording`, `session_updates`), the session prefix index, the request-plane DTOs (`api`), explicit identity material (`identity`), its configuration slice (`config::IndexerConfig`), and the shared test helpers (`testing`, behind `testing` or `bench`). `lib/kv-router` re-exports every item at its old path and keeps the adapters: the HTTP indexer server, the ZMQ listener and `WorkerRegistry`, the peer recovery client, and Prometheus exposition. Index internals are described in [src/README.md](src/README.md).

Dependencies: `dynamo-router-protocol`, `dynamo-kv-events`, `dynamo-kv-hashing`, and nothing router-side beyond them; no axum, reqwest, zmq, or `dynamo-runtime` under any feature. Tokio stays: the indexers are actors with channels and a single-threaded runtime, and the stage rule forbids sockets and HTTP, not async. The `export` feature pulls Prometheus because the counters are `IntCounterVec` fields today; it is transitional and goes away with the atomics rewrite. Hosts that build counters from a runtime `Component` use the `KvIndexerMetricsExt` extension trait in `dynamo-kv-router`.

Part of the router split (see `lib/router/README.md`).
