// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Source block types for the offload engine.
//!
//! Blocks can be provided to the offload engine in three forms:
//! - External: Just a BlockId, block is held elsewhere
//! - Strong: RAII ImmutableBlock reference
//! - Weak: WeakBlock that may have been evicted

use crate::v2::BlockId;
use crate::v2::logical::blocks::{BlockMetadata, ImmutableBlock, WeakBlock};

/// Represents a single block source for offloading.
///
/// The source type determines how the block is resolved:
/// - `External`: Caller holds the block, we just have the ID
/// - `Strong`: We hold a strong RAII reference
/// - `Weak`: We hold a weak reference that may need upgrading
#[derive(Debug)]
pub enum SourceBlock<T: BlockMetadata> {
    /// Just a BlockId - block is held elsewhere by caller
    External(BlockId),
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
            SourceBlock::External(id) => Some(*id),
            SourceBlock::Strong(block) => Some(block.block_id()),
            SourceBlock::Weak(_) => None,
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

impl<T: BlockMetadata> From<BlockId> for SourceBlock<T> {
    fn from(id: BlockId) -> Self {
        SourceBlock::External(id)
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
    /// External block IDs - blocks held elsewhere
    External(Vec<BlockId>),
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
            SourceBlocks::External(ids) => ids.len(),
            SourceBlocks::Strong(blocks) => blocks.len(),
            SourceBlocks::Weak(blocks) => blocks.len(),
        }
    }

    /// Check if the collection is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get block IDs for external blocks, or None for other types.
    pub fn external_ids(&self) -> Option<&[BlockId]> {
        match self {
            SourceBlocks::External(ids) => Some(ids),
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

impl<T: BlockMetadata> From<Vec<BlockId>> for SourceBlocks<T> {
    fn from(ids: Vec<BlockId>) -> Self {
        SourceBlocks::External(ids)
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
            SourceBlock::External(id) => SourceBlocks::External(vec![id]),
            SourceBlock::Strong(b) => SourceBlocks::Strong(vec![b]),
            SourceBlock::Weak(b) => SourceBlocks::Weak(vec![b]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_blocks_from_vec_block_ids() {
        let ids = vec![1, 2, 3];
        let blocks: SourceBlocks<()> = ids.into();
        assert!(blocks.is_external());
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks.external_ids(), Some(&[1, 2, 3][..]));
    }

    #[test]
    fn test_source_blocks_empty() {
        let blocks: SourceBlocks<()> = SourceBlocks::empty_external();
        assert!(blocks.is_empty());
        assert!(blocks.is_external());
    }
}
