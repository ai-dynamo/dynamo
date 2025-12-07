// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Source block types for the offload engine.
//!
//! Blocks can be provided to the offload engine in three forms:
//! - External: BlockId + SequenceHash, block is held elsewhere
//! - Strong: RAII ImmutableBlock reference
//! - Weak: WeakBlock that may have been evicted

use std::marker::PhantomData;

use crate::v2::logical::blocks::{BlockMetadata, ImmutableBlock, WeakBlock};
use crate::v2::{BlockId, SequenceHash};

/// External block reference with sequence hash for registration.
///
/// Used when the caller holds the actual block but wants to provide
/// the offload engine with enough information to register blocks
/// in the destination tier after transfer.
#[derive(Debug, Clone, Copy)]
pub struct ExternalBlock<T: BlockMetadata> {
    /// The block ID in the source tier
    pub block_id: BlockId,
    /// The sequence hash for registration in destination tier
    pub sequence_hash: SequenceHash,
    _marker: PhantomData<T>,
}

impl<T: BlockMetadata> ExternalBlock<T> {
    /// Create a new external block reference.
    pub fn new(block_id: BlockId, sequence_hash: SequenceHash) -> Self {
        Self {
            block_id,
            sequence_hash,
            _marker: PhantomData,
        }
    }
}

/// Represents a single block source for offloading.
///
/// The source type determines how the block is resolved:
/// - `External`: Caller holds the block, we have ID + SequenceHash for registration
/// - `Strong`: We hold a strong RAII reference
/// - `Weak`: We hold a weak reference that may need upgrading
#[derive(Debug)]
pub enum SourceBlock<T: BlockMetadata> {
    /// External block reference with ID and sequence hash
    External(ExternalBlock<T>),
    /// Strong RAII reference to an immutable block
    Strong(ImmutableBlock<T>),
    /// Weak reference that may have been evicted
    Weak(WeakBlock<T>),
}

impl<T: BlockMetadata> SourceBlock<T> {
    /// Get the block ID if available without upgrading.
    ///
    /// For External and Strong variants, returns Some(id).
    /// For Weak variant, returns None (would need upgrade to get ID).
    pub fn block_id(&self) -> Option<BlockId> {
        match self {
            SourceBlock::External(ext) => Some(ext.block_id),
            SourceBlock::Strong(block) => Some(block.block_id()),
            SourceBlock::Weak(_) => None,
        }
    }

    /// Get the sequence hash if available without upgrading.
    ///
    /// All variants can provide sequence_hash without upgrading:
    /// - External: stored in ExternalBlock
    /// - Strong: from ImmutableBlock
    /// - Weak: WeakBlock stores sequence_hash directly
    pub fn sequence_hash(&self) -> Option<SequenceHash> {
        match self {
            SourceBlock::External(ext) => Some(ext.sequence_hash),
            SourceBlock::Strong(block) => Some(block.sequence_hash()),
            SourceBlock::Weak(weak) => Some(weak.sequence_hash()),
        }
    }

    /// Check if this is an external block reference.
    pub fn is_external(&self) -> bool {
        matches!(self, SourceBlock::External(_))
    }

    /// Check if this is a strong reference.
    pub fn is_strong(&self) -> bool {
        matches!(self, SourceBlock::Strong(_))
    }

    /// Check if this is a weak reference.
    pub fn is_weak(&self) -> bool {
        matches!(self, SourceBlock::Weak(_))
    }
}

impl<T: BlockMetadata> From<ExternalBlock<T>> for SourceBlock<T> {
    fn from(ext: ExternalBlock<T>) -> Self {
        SourceBlock::External(ext)
    }
}

impl<T: BlockMetadata> From<ImmutableBlock<T>> for SourceBlock<T> {
    fn from(block: ImmutableBlock<T>) -> Self {
        SourceBlock::Strong(block)
    }
}

impl<T: BlockMetadata> From<WeakBlock<T>> for SourceBlock<T> {
    fn from(block: WeakBlock<T>) -> Self {
        SourceBlock::Weak(block)
    }
}

/// Collection of source blocks for batch operations.
///
/// Blocks are grouped by their source type for efficient processing.
/// All blocks in a SourceBlocks must be of the same type.
#[derive(Debug)]
pub enum SourceBlocks<T: BlockMetadata> {
    /// External block references with IDs and sequence hashes
    External(Vec<ExternalBlock<T>>),
    /// Strong RAII references
    Strong(Vec<ImmutableBlock<T>>),
    /// Weak references that may need upgrading
    Weak(Vec<WeakBlock<T>>),
}

impl<T: BlockMetadata> SourceBlocks<T> {
    /// Create an empty collection of external blocks.
    pub fn empty_external() -> Self {
        SourceBlocks::External(Vec::new())
    }

    /// Create an empty collection of strong blocks.
    pub fn empty_strong() -> Self {
        SourceBlocks::Strong(Vec::new())
    }

    /// Create an empty collection of weak blocks.
    pub fn empty_weak() -> Self {
        SourceBlocks::Weak(Vec::new())
    }

    /// Get the number of blocks in this collection.
    pub fn len(&self) -> usize {
        match self {
            SourceBlocks::External(blocks) => blocks.len(),
            SourceBlocks::Strong(blocks) => blocks.len(),
            SourceBlocks::Weak(blocks) => blocks.len(),
        }
    }

    /// Check if the collection is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get external blocks, or None for other types.
    pub fn external_blocks(&self) -> Option<&[ExternalBlock<T>]> {
        match self {
            SourceBlocks::External(blocks) => Some(blocks),
            _ => None,
        }
    }

    /// Get strong blocks, or None for other types.
    pub fn strong_blocks(&self) -> Option<&[ImmutableBlock<T>]> {
        match self {
            SourceBlocks::Strong(blocks) => Some(blocks),
            _ => None,
        }
    }

    /// Get weak blocks, or None for other types.
    pub fn weak_blocks(&self) -> Option<&[WeakBlock<T>]> {
        match self {
            SourceBlocks::Weak(blocks) => Some(blocks),
            _ => None,
        }
    }

    /// Check if this is external blocks.
    pub fn is_external(&self) -> bool {
        matches!(self, SourceBlocks::External(_))
    }

    /// Check if this is strong blocks.
    pub fn is_strong(&self) -> bool {
        matches!(self, SourceBlocks::Strong(_))
    }

    /// Check if this is weak blocks.
    pub fn is_weak(&self) -> bool {
        matches!(self, SourceBlocks::Weak(_))
    }
}

impl<T: BlockMetadata> From<Vec<ExternalBlock<T>>> for SourceBlocks<T> {
    fn from(blocks: Vec<ExternalBlock<T>>) -> Self {
        SourceBlocks::External(blocks)
    }
}

impl<T: BlockMetadata> From<Vec<ImmutableBlock<T>>> for SourceBlocks<T> {
    fn from(blocks: Vec<ImmutableBlock<T>>) -> Self {
        SourceBlocks::Strong(blocks)
    }
}

impl<T: BlockMetadata> From<Vec<WeakBlock<T>>> for SourceBlocks<T> {
    fn from(blocks: Vec<WeakBlock<T>>) -> Self {
        SourceBlocks::Weak(blocks)
    }
}

// Allow converting a single SourceBlock into SourceBlocks
impl<T: BlockMetadata> From<SourceBlock<T>> for SourceBlocks<T> {
    fn from(block: SourceBlock<T>) -> Self {
        match block {
            SourceBlock::External(ext) => SourceBlocks::External(vec![ext]),
            SourceBlock::Strong(b) => SourceBlocks::Strong(vec![b]),
            SourceBlock::Weak(b) => SourceBlocks::Weak(vec![b]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_tokens::TokenBlockSequence;

    /// Create a test sequence hash at a given position.
    fn test_seq_hash(position: usize) -> SequenceHash {
        let tokens_per_block = 4;
        let total_tokens = (position + 1) * tokens_per_block;
        let tokens: Vec<u32> = (0..total_tokens as u32).collect();
        let seq = TokenBlockSequence::from_slice(&tokens, tokens_per_block as u32, Some(1337));
        seq.blocks()[position].positional_sequence_hash()
    }

    #[test]
    fn test_external_block_creation() {
        let hash = test_seq_hash(0);
        let ext: ExternalBlock<()> = ExternalBlock::new(42, hash);
        assert_eq!(ext.block_id, 42);
        assert_eq!(ext.sequence_hash, hash);
    }

    #[test]
    fn test_source_blocks_from_vec_external() {
        let ext1: ExternalBlock<()> = ExternalBlock::new(1, test_seq_hash(0));
        let ext2: ExternalBlock<()> = ExternalBlock::new(2, test_seq_hash(1));
        let ext3: ExternalBlock<()> = ExternalBlock::new(3, test_seq_hash(2));
        let blocks: SourceBlocks<()> = vec![ext1, ext2, ext3].into();
        assert!(blocks.is_external());
        assert_eq!(blocks.len(), 3);
        let external = blocks.external_blocks().unwrap();
        assert_eq!(external[0].block_id, 1);
        assert_eq!(external[1].block_id, 2);
        assert_eq!(external[2].block_id, 3);
    }

    #[test]
    fn test_source_blocks_empty() {
        let blocks: SourceBlocks<()> = SourceBlocks::empty_external();
        assert!(blocks.is_empty());
        assert!(blocks.is_external());
    }

    #[test]
    fn test_source_block_accessors() {
        let hash = test_seq_hash(5);
        let ext: ExternalBlock<()> = ExternalBlock::new(42, hash);
        let block: SourceBlock<()> = ext.into();
        assert_eq!(block.block_id(), Some(42));
        assert_eq!(block.sequence_hash(), Some(hash));
        assert!(block.is_external());
    }
}
