<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-kv-hashing

`dynamo-kv-hashing` defines the block hashes engines and the router compare. `routing::compute_block_hash_for_seq` turns a request's token blocks into `LocalBlockHash` values, the same content hash an engine reports for each block it stores, and `routing::compute_seq_hash_for_block` chains them into sequence hashes. `ExternalSequenceBlockHash` is the engine's own sequence-aware block id, produced by an algorithm the router does not know. Its `multimodal` module defines how images and other non-text inputs change these hashes. It is pure computation with no async or I/O, so any crate can depend on it.
