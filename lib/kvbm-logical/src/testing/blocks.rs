// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test helpers that mint `(BlockId, SequenceHash)` pairs for low-level
//! tests against the inactive backends and the unified [`BlockStore`].
//!
//! Higher-level tests construct guards (`MutableBlock`, `CompleteBlock`,
//! `ImmutableBlock`) by going through a [`BlockManager`](crate::BlockManager).

#![allow(dead_code)]

use crate::BlockId;
use crate::KvbmSequenceHashProvider;
use crate::pools::SequenceHash;

use super::token_blocks::create_test_token_block;

/// Compute the [`SequenceHash`] for a token sequence.
pub(crate) fn hash_for_tokens(tokens: &[u32]) -> SequenceHash {
    create_test_token_block(tokens, tokens.len() as u32).kvbm_sequence_hash()
}

/// Mint a `(BlockId, SequenceHash)` pair for backend-level tests.
pub(crate) fn block_id_and_hash(id: BlockId, tokens: &[u32]) -> (BlockId, SequenceHash) {
    (id, hash_for_tokens(tokens))
}
