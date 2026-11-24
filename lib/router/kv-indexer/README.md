<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-kv-indexer

`dynamo-kv-indexer` tracks which KV cache blocks each worker holds and, for a request, how much of its prefix each worker already has cached. It applies KV events to one of several index structures (radix tree, positional index, concurrent compressed radix tree, and thread-pool and branch-sharded wrappers around them), and also holds the lower-tier (CPU and disk) continuation index, approximate LRU retention, and the cuckoo-filter index the DC relay uses. Design notes: [src/README.md](src/README.md).

The intended public API is two modules: `host`, what in-tree hosts (the frontend, `dynamo-kv-router`) import, and `cuckoo`, the DC relay contract. Everything else the crate root exposes exists so `dynamo-kv-router`'s `compat.rs` can keep the pre-split `dynamo_kv_router` paths compiling through v1.8.0 and v1.9.0; those exports are removed with that shim in v1.10.0 (see the deprecations page), and the module visibility is tightened in the next PR of this stack.
