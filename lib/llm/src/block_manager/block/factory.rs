use crate::block_manager::LayoutConfig;

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
}

/// Extension trait for factories that can produce all blocks at once
pub trait IntoBlocks<S: Storage, L: LocalityProvider> {
    /// Consume the factory and create all blocks with default metadata
    fn into_blocks<M: BlockMetadata + Default>(self) -> BlockResult<Vec<Block<S, L, M>>>;

    /// Consume the factory and create all blocks with the given metadata value
    fn into_blocks_with_metadata<M: BlockMetadata + Clone>(
        self,
        metadata: M,
    ) -> BlockResult<Vec<Block<S, L, M>>>;
}

/// Factory for creating LocalBlockData (DEPRECATED - use LocalBlockFactory instead)
#[derive(Debug, Clone, Dissolve)]
pub struct LocalBlockDataFactory<S: Storage> {
    layout: Arc<dyn BlockLayout<StorageType = S>>,
    block_set_idx: usize,
    worker_id: WorkerID,
}

impl<S: Storage> LocalBlockDataFactory<S> {
    pub fn new(
        layout: Arc<dyn BlockLayout<StorageType = S>>,
        block_set_idx: usize,
        worker_id: WorkerID,
    ) -> Self {
        Self {
            layout,
            block_set_idx,
            worker_id,
        }
    }
}

impl<S: Storage> BlockFactory<S, locality::Local> for LocalBlockDataFactory<S> {
    fn create_block_data(&self, block_idx: BlockId) -> BlockResult<BlockData<S>> {
        if block_idx >= self.layout.num_blocks() {
            return Err(BlockError::InvalidBlockID(block_idx));
        }

        let data = BlockData::new(
            self.layout.clone(),
            block_idx,
            self.block_set_idx,
            self.worker_id,
        );
        Ok(data)
    }

    fn num_blocks(&self) -> usize {
        self.layout.num_blocks()
    }

    fn layout_config(&self) -> &LayoutConfig {
        self.layout.config()
    }
}
