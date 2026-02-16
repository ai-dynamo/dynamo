// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![doc = include_str!("../../docs/sequence.md")]

mod assignments;
mod store;

pub use assignments::{
    zip_assigned, zip_assigned_pending, ExternalBlockAssignments,
    LogicalBlockAssignmentError, LogicalBlockAssignments,
};

use std::ops::Range;

use dynamo_tokens::{Token, TokenBlock, TokenBlockError, TokenBlockSequence, Tokens};

use crate::{BlockId, KvbmSequenceHashProvider, SequenceHash};

/// Errors that can occur in block sequence operations.
#[derive(Debug, thiserror::Error)]
pub enum BlockSequenceError {
    /// A known block_id appeared after an unknown block_id in `extend_block_ids`.
    #[error(
        "ordering violation: known block_id {known_id} at index {known_index} \
         appeared after new block_id {new_id} at index {first_new_index}"
    )]
    OrderingViolation {
        known_id: BlockId,
        new_id: BlockId,
        known_index: usize,
        first_new_index: usize,
    },

    /// The position embedded in a sequence hash didn't match the expected position.
    #[error(
        "position mismatch for block_id {block_id}: expected {expected}, actual {actual}"
    )]
    PositionMismatch {
        expected: usize,
        actual: u64,
        block_id: BlockId,
    },

    /// A block_id already exists in one of the collections.
    #[error("duplicate block_id {block_id} already present")]
    DuplicateBlockId {
        block_id: BlockId,
    },

    /// Error from underlying token block operations.
    #[error("token extension error: {0}")]
    TokenExtension(#[from] TokenBlockError),
}

/// Owns a `TokenBlockSequence` and provides sequence access and token extension methods.
///
/// This is a thin wrapper around `TokenBlockSequence` that provides a convenient API
/// for the block assignment workflow. It does NOT embed assignments — those are managed
/// separately by [`ExternalBlockAssignments`].
#[derive(Debug)]
pub struct BlockSequence {
    sequence: TokenBlockSequence,
}

impl BlockSequence {
    /// Creates a new `BlockSequence` from tokens, block size, and optional salt hash.
    pub fn new(tokens: Vec<Token>, block_size: u32, salt_hash: Option<u64>) -> Self {
        let tokens = Tokens::from(tokens);
        Self {
            sequence: TokenBlockSequence::new(tokens, block_size, salt_hash),
        }
    }

    /// Returns the completed token blocks.
    pub fn blocks(&self) -> &[TokenBlock] {
        self.sequence.blocks()
    }

    /// Returns the block size.
    pub fn block_size(&self) -> usize {
        self.sequence.block_size()
    }

    /// Returns the total number of tokens (including partial block).
    pub fn total_tokens(&self) -> usize {
        self.sequence.total_tokens()
    }

    /// Returns a reference to the underlying `TokenBlockSequence`.
    pub fn sequence(&self) -> &TokenBlockSequence {
        &self.sequence
    }

    /// Returns all sequence hashes from completed blocks.
    pub fn all_sequence_hashes(&self) -> Vec<SequenceHash> {
        self.sequence
            .blocks()
            .iter()
            .map(|b| b.kvbm_sequence_hash())
            .collect()
    }

    /// Extends the sequence with tokens, potentially completing blocks.
    ///
    /// Returns the range of newly completed block indices, or `None` if no blocks completed.
    pub fn extend_tokens(
        &mut self,
        tokens: Vec<Token>,
    ) -> Result<Option<Range<usize>>, BlockSequenceError> {
        let tokens = Tokens::from(tokens);
        self.sequence.extend(tokens).map_err(Into::into)
    }

    /// Appends a single token to the sequence.
    ///
    /// Returns the index of the completed block if the token completed a block.
    pub fn append_token(
        &mut self,
        token: Token,
    ) -> Result<Option<usize>, BlockSequenceError> {
        self.sequence.append(token).map_err(Into::into)
    }
}
