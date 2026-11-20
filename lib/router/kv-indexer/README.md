<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-kv-indexer

`dynamo-kv-indexer` tracks which KV cache blocks each worker holds and, for a request, how much of its prefix each worker already has cached. It applies KV events to one of several index structures (radix tree, positional index, concurrent compressed radix tree, and thread-pool and branch-sharded wrappers around them), and also holds the lower-tier (CPU and disk) continuation index, approximate LRU retention, and the cuckoo-filter index the DC relay uses. Design notes: [src/README.md](src/README.md).

The intended public API is three modules: `host`, what in-tree hosts (the frontend, `dynamo-kv-router`) import; `cuckoo`, the DC relay contract; and `stream`, for a process that follows an engine's KV events. Everything else the crate root exposes exists so `dynamo-kv-router`'s `compat.rs` can keep the pre-split `dynamo_kv_router` paths compiling through v1.8.0 and v1.9.0; those exports are removed with that shim in v1.10.0 (see the deprecations page), and the module visibility is tightened in the next PR of this stack.

`stream` re-exports [`dynamo-kv-stream`](../../kv-stream/README.md), the crate engines embed: one rank's KV event batches folded into a residency ledger, with cursor replay and snapshots. On top of it, this crate adds what needs Dynamo's types: `RadixIndex` and `LaneIndex` feed a rank's stream into the radix tree and, through the thread pool's lanes, the concurrent tree, and `image_to_router_events` turns a snapshot into the `RouterEvent`s a router installs.
