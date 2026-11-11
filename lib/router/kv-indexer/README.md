<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-kv-indexer

`dynamo-kv-indexer` tracks which KV cache blocks each worker holds and, for a request, how much of its prefix each worker already has cached. It applies KV events to one of several index structures (radix tree, positional index, concurrent compressed radix tree, and thread-pool and branch-sharded wrappers around them), and also holds the lower-tier (CPU and disk) continuation index, approximate LRU retention, and the cuckoo-filter index the DC relay uses. Design notes: [src/README.md](src/README.md).
