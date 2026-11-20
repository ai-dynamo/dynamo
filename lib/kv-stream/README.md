<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-kv-stream

`dynamo-kv-stream` keeps one rank's KV cache state for the process that reads the engine's KV events: an engine's Rust frontend (vLLM, SGLang) or a Dynamo sidecar. It folds the engine's ordered event batches into an exact residency ledger and serves the three APIs a router needs from it: the live stream, replay from a cursor, and a snapshot cut at an exact sequence.

It is synchronous, single-owner and transport-neutral. The host owns gRPC and its wire types and picks the thread that owns each stream; the stream keeps the host's own batch objects for replay. The crate depends on `rustc-hash` and `thiserror` only, and builds on Rust 1.92 and later.

```toml
dynamo-kv-stream = "=<Dynamo release>"
```

The crate is versioned with Dynamo, and a Dynamo minor release may change its API, so pin the exact release.

| API the host serves | What it calls |
|---|---|
| live stream | `RankStream::apply_batch`, then fan out the batch it just passed in |
| replay from a cursor | `RankStream::resume` and `RankStream::backlog` |
| snapshot | `RankStream::snapshot`, then `SnapshotImage::runs` on any thread |
| subscribe, server picks | `RankStream::admit` |

Both engines fit one ledger: each tier is either a set (SGLang: a block is on a tier or not) or counted (vLLM: duplicate blocks under one hash, one remove each). The ledger is keyed by the engine's own block hash, so the engine computes nothing for Dynamo.

A prefix index is optional. `StreamIndex` is the seam, and `dynamo_kv_indexer::stream` implements it for Dynamo's radix tree and concurrent tree.
