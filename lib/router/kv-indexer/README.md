<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-kv-indexer

KV events in, prefix knowledge out. The router's indexer stage, landing in slices: this slice holds the cuckoo-filter pool index, the hash lifecycle delegate, and the indexer counters; the radix trees, the concurrent compressed tree, the lower tiers, and the ingest backend follow. `lib/kv-router` re-exports every item at its old path until the stage is complete.

Dependencies: `dynamo-router-protocol`, `dynamo-kv-events`, `dynamo-kv-hashing`, and nothing router-side beyond them; no axum, reqwest, zmq, or `dynamo-runtime` under any feature. The `export` feature pulls Prometheus because the counters are `IntCounterVec` fields today; it is transitional and goes away with the atomics rewrite. Hosts that build counters from a runtime `Component` use the `KvIndexerMetricsExt` extension trait in `dynamo-kv-router`.

Part of the router split (see `lib/router/README.md`).
