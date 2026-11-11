<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-kv-hashing

`dynamo-kv-hashing` is the hash contract shared by engines and the router: it turns blocks of tokens into the `LocalBlockHash` and `ExternalSequenceBlockHash` values both sides compare (`routing::compute_block_hash_for_seq`), and its `multimodal` module defines how images and other non-text inputs change those hashes. It is pure computation with no async or I/O, so any crate can depend on it.
