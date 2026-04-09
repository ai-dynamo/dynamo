// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Sequence Module
//!
//! Block assignment tracking for token sequences.
//!
//! This module provides two assignment trackers that share a three-phase
//! lifecycle: **unassigned** (queued), **staged** (paired with token data),
//! and **assigned** (committed).
//!
//! - [`ExternalBlockAssignments`] tracks at the **identity level** — mapping
//!   `BlockId` to `SequenceHash`.
//! - [`LogicalBlockAssignments`] tracks at the **guard level** — managing
//!   RAII block guards through `MutableBlock` to `CompleteBlock` to
//!   `ImmutableBlock` transitions.
//!
//! Both types are backed by the same ordered-collection machinery and expose
//! similar query, iteration, and mutation APIs.
//!
//! ## `ExternalBlockAssignments`
//!
//! Identity-level tracking of block IDs paired with sequence hashes.
//!
//! ### Basic flow
//!
//! Create a sequence from tokens, register block IDs, and assign them
//! against the completed token blocks:
//!
//! ```rust
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use kvbm_logical::{ExternalBlockAssignments, BlockSequence};
//!
//! // Create a sequence with 3 complete blocks (4 tokens each).
//! let tokens: Vec<u32> = (0..12).collect();
//! let seq = BlockSequence::new(tokens, 4, None);
//! assert_eq!(seq.blocks().len(), 3);
//!
//! // Create assignments starting at offset 0.
//! let mut assignments = ExternalBlockAssignments::new(0);
//!
//! // Register block IDs (e.g., allocated by the scheduler).
//! assignments.extend_block_ids(vec![10, 20, 30])?;
//! assert_eq!(assignments.unassigned_count(), 3);
//!
//! // Assign: pairs each pending ID with its token block's hash.
//! let range = assignments.assign_pending(seq.blocks())?;
//! assert_eq!(range, 0..3);
//! assert_eq!(assignments.assigned_count(), 3);
//! assert_eq!(assignments.unassigned_count(), 0);
//!
//! // Query by index.
//! let (id, hash) = assignments.get_assigned(0).unwrap();
//! assert_eq!(id, 10);
//! assert_eq!(hash, seq.all_sequence_hashes()[0]);
//! # Ok(())
//! # }
//! ```
//!
//! ### Two-step staging
//!
//! Use `stage_pending` and `commit_staged` for explicit control over the
//! staging phase (useful when validation must happen between staging and
//! committing):
//!
//! ```rust
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use kvbm_logical::{ExternalBlockAssignments, BlockSequence};
//!
//! let tokens: Vec<u32> = (0..8).collect();
//! let seq = BlockSequence::new(tokens, 4, None);
//!
//! let mut assignments = ExternalBlockAssignments::new(0);
//! assignments.extend_block_ids(vec![1, 2])?;
//!
//! // Stage: pairs IDs with hashes but does not commit.
//! let staged_range = assignments.stage_pending(seq.blocks())?;
//! assert_eq!(staged_range, 0..2);
//! assert_eq!(assignments.staged_count(), 2);
//! assert_eq!(assignments.assigned_count(), 0);
//!
//! // Commit: moves staged into assigned.
//! let assigned_range = assignments.commit_staged();
//! assert_eq!(assigned_range, 0..2);
//! assert_eq!(assignments.assigned_count(), 2);
//! assert_eq!(assignments.staged_count(), 0);
//! # Ok(())
//! # }
//! ```
//!
//!
//! ## `LogicalBlockAssignments`
//!
//! Guard-level tracking through the full block lifecycle. Blocks flow
//! through `MutableBlock` (unassigned) to `CompleteBlock` (staged) to
//! `ImmutableBlock` (assigned/registered).
//!
//! ### Full pipeline
//!
//! Allocate physical blocks from a `BlockManager`, stage them against token
//! data, register them, and query the result:
//!
//! ```rust
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use kvbm_logical::{
//!     BlockManager, BlockRegistry, BlockSequence,
//!     LogicalBlockAssignments,
//!     manager::FrequencyTrackingCapacity,
//! };
//!
//! // Build a manager with 10 blocks of size 4.
//! let tracker = FrequencyTrackingCapacity::Small.create_tracker();
//! let registry = BlockRegistry::builder()
//!     .frequency_tracker(tracker)
//!     .build();
//! let manager = BlockManager::<()>::builder()
//!     .block_count(10)
//!     .block_size(4)
//!     .registry(registry)
//!     .with_lru_backend()
//!     .build()?;
//!
//! // Create a token sequence with 3 complete blocks.
//! let tokens: Vec<u32> = (0..12).collect();
//! let seq = BlockSequence::new(tokens, 4, None);
//!
//! // Allocate 3 mutable blocks from the manager.
//! let blocks = manager.allocate_blocks(3).unwrap();
//! let ids: Vec<usize> = blocks.iter().map(|b| b.block_id()).collect();
//!
//! let mut la = LogicalBlockAssignments::new();
//!
//! // Extend: adds mutable blocks to the unassigned queue.
//! la.extend_blocks(blocks)?;
//! assert_eq!(la.unassigned_count(), 3);
//!
//! // Stage: completes each mutable block with its token data.
//! la.stage(seq.blocks())?;
//! assert_eq!(la.staged_count(), 3);
//! assert_eq!(la.unassigned_count(), 0);
//!
//! // Register: finalizes staged blocks through the manager.
//! la.register(&manager);
//! assert_eq!(la.assigned_count(), 3);
//! assert_eq!(la.staged_count(), 0);
//!
//! // Query assigned blocks.
//! for i in 0..3 {
//!     let (id, immutable) = la.get_assigned(i).unwrap();
//!     assert_eq!(*id, ids[i]);
//!     assert_eq!(immutable.sequence_hash(), seq.all_sequence_hashes()[i]);
//! }
//! # Ok(())
//! # }
//! ```

mod assignments;
mod store;

pub use assignments::{
    ExternalBlockAssignments, LogicalBlockAssignmentError, LogicalBlockAssignments, zip_assigned,
    zip_assigned_pending,
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
    #[error("position mismatch for block_id {block_id}: expected {expected}, actual {actual}")]
    PositionMismatch {
        expected: usize,
        actual: u64,
        block_id: BlockId,
    },

    /// A block_id already exists in one of the collections.
    #[error("duplicate block_id {block_id} already present")]
    DuplicateBlockId { block_id: BlockId },

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
    pub fn append_token(&mut self, token: Token) -> Result<Option<usize>, BlockSequenceError> {
        self.sequence.append(token).map_err(Into::into)
    }
}
