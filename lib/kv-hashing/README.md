<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-kv-hashing

`dynamo-kv-hashing` is the hash contract shared by engines and the router. Engines publish the `ExternalSequenceBlockHash` values they compute for the blocks they cache, and the router recomputes the same identities from a request's tokens: `routing::compute_block_hash_for_seq` gives each block's `LocalBlockHash` and `routing::compute_seq_hash_for_block` the sequence hashes. Its `multimodal` module defines how images and other non-text inputs change those hashes. It is pure computation with no async or I/O, so any crate can depend on it.
