// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod local;
pub mod logical;

pub use local::LocalBlockDataFactory;

use crate::block_manager::{LayoutConfig, OffloadFilter};

use super::*;

use derive_getters::Dissolve;

/// Core trait for block factories that can create blocks with specific locality and storage
///
/// This trait provides the foundation for creating blocks with different locality providers
/// (Local, Logical, etc.) and storage types.
pub trait BlockFactory<S: Storage, L: LocalityProvider> {
    /// Create block data for a specific block ID
    /// This does not consume the factory and can be called multiple times
    fn create_block_data(&self, block_id: BlockId) -> BlockResult<L::BlockData<S>>;

    /// Create a single block with default metadata
    /// This does not consume the factory and can be called multiple times
    fn create_block<M: BlockMetadata + Default>(
        &self,
        block_id: BlockId,
    ) -> BlockResult<Block<S, L, M>> {
        let block_data = self.create_block_data(block_id)?;
        Block::new(block_data, M::default())
    }

    /// Create a single block with the given metadata
    /// This does not consume the factory and can be called multiple times
    fn create_block_with_metadata<M: BlockMetadata>(
        &self,
        block_id: BlockId,
        metadata: M,
    ) -> BlockResult<Block<S, L, M>> {
        let block_data = self.create_block_data(block_id)?;
        Block::new(block_data, metadata)
    }

    /// Get the number of blocks this factory can create
    fn num_blocks(&self) -> usize;

    /// Get the layout configuration information
    fn layout_config(&self) -> &LayoutConfig;

    /// Get the offload filter for this factory
    fn offload_filter(&self) -> Option<Arc<dyn OffloadFilter>>;
}

/// Extension trait for factories that can produce all blocks at once
pub trait IntoBlocks<S: Storage, L: LocalityProvider>: BlockFactory<S, L> + Sized {
    /// Consume the factory and create all blocks with default metadata
    fn into_blocks<M: BlockMetadata + Default>(self) -> BlockResult<Vec<Block<S, L, M>>> {
        let num_blocks = self.num_blocks();
        let mut blocks = Vec::with_capacity(num_blocks);
        for block_idx in 0..num_blocks {
            let block = self.create_block(block_idx)?;
            blocks.push(block);
        }
        Ok(blocks)
    }

    /// Consume the factory and create all blocks with the given metadata value
    fn into_blocks_with_metadata<M: BlockMetadata + Clone>(
        self,
        metadata: M,
    ) -> BlockResult<Vec<Block<S, L, M>>> {
        let num_blocks = self.num_blocks();
        let mut blocks = Vec::with_capacity(num_blocks);
        for block_idx in 0..num_blocks {
            let block = self.create_block_with_metadata(block_idx, metadata.clone())?;
            blocks.push(block);
        }
        Ok(blocks)
    }
}
