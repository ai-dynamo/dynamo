<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-kv-indexer

KV events in, prefix knowledge out. The router's indexer stage, landing in slices. It now holds the radix trees (`RadixTree`, `PositionalIndexer`, `ConcurrentRadixTreeCompressed`), the `ThreadPoolIndexer` and `BranchShardedIndexer` wrappers, approximate LRU retention and pruning, the cuckoo-filter pool index, the hash lifecycle delegate, the indexer counters, the residency projection, overlap scores, KV transfer candidates, and the shared test helpers (`testing`, behind `testing` or `bench`). It also holds the lower-tier continuation index and the per-tier registry (`LowerTierIndexer`, `LowerTierIndexers`, `TieredMatchProvider`). The request-plane DTOs, `LocalKvIndexer`, and the ingest backend follow. `lib/kv-router` re-exports every item at its old path until the stage is complete. Index internals are described in [src/README.md](src/README.md).

Dependencies: `dynamo-router-protocol`, `dynamo-kv-events`, `dynamo-kv-hashing`, and nothing router-side beyond them; no axum, reqwest, zmq, or `dynamo-runtime` under any feature. Tokio stays: the indexers are actors with channels and a single-threaded runtime, and the stage rule forbids sockets and HTTP, not async. The `export` feature pulls Prometheus because the counters are `IntCounterVec` fields today; it is transitional and goes away with the atomics rewrite. Hosts that build counters from a runtime `Component` use the `KvIndexerMetricsExt` extension trait in `dynamo-kv-router`.

Part of the router split (see `lib/router/README.md`).
