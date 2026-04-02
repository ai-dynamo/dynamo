// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::block_manager::{
    OffloadFilter,
    locality::{Logical, LogicalBlockData, LogicalResources},
};

#[derive(Debug)]
pub struct LogicalBlockFactory<S: Storage, R: LogicalResources> {
    layout_config: Arc<LayoutConfig>,
    block_set_idx: usize,
    worker_id: WorkerID,
    resources: Arc<R>,
    storage_type: StorageType,
    storage: std::marker::PhantomData<S>,
    offload_filter: Option<Arc<dyn OffloadFilter>>,
}

impl<S: Storage, R: LogicalResources> LogicalBlockFactory<S, R> {
    pub fn new(
        layout_config: Arc<LayoutConfig>,
        block_set_idx: usize,
        worker_id: WorkerID,
        resources: Arc<R>,
        storage_type: StorageType,
        offload_filter: Option<Arc<dyn OffloadFilter>>,
    ) -> Self {
        Self {
            layout_config,
            block_set_idx,
            worker_id,
            resources,
            storage_type,
            storage: std::marker::PhantomData,
            offload_filter,
        }
    }
}

impl<S: Storage, R: LogicalResources> BlockFactory<S, Logical<R>> for LogicalBlockFactory<S, R> {
    fn create_block_data(&self, block_idx: BlockId) -> BlockResult<LogicalBlockData<S, R>> {
        if block_idx >= self.num_blocks() {
            return Err(BlockError::InvalidBlockID(block_idx));
        }

        let data = LogicalBlockData::new(
            block_idx,
            self.block_set_idx,
            self.worker_id,
            self.resources.clone(),
            self.storage_type,
            self.layout_config.page_size,
        );
        Ok(data)
    }

    fn num_blocks(&self) -> usize {
        self.layout_config.num_blocks
    }

    fn layout_config(&self) -> &LayoutConfig {
        &self.layout_config
    }

    fn offload_filter(&self) -> Option<Arc<dyn OffloadFilter>> {
        self.offload_filter.clone()
    }
}

impl<S: Storage, R: LogicalResources> IntoBlocks<S, Logical<R>> for LogicalBlockFactory<S, R> {}

#[cfg(test)]
mod tests {
    use crate::block_manager::block::data::logical::null::NullResources;
    use crate::block_manager::{ManagedBlockPool, PinnedStorage};

    use super::*;

    const TEST_BLOCK_SET_ID: usize = 42;
    const TEST_WORKER_ID: WorkerID = 1337;

    #[tokio::test]
    async fn test_logical_block_factory() {
        let layout_config = LayoutConfig::builder()
            .num_blocks(10)
            .page_size(16)
            .num_layers(3)
            .outer_dim(2)
            .inner_dim(8192)
            .dtype_width_bytes(2)
            .build()
            .unwrap();

        let factory = LogicalBlockFactory::<PinnedStorage, NullResources>::new(
            Arc::new(layout_config),
            TEST_BLOCK_SET_ID,
            TEST_WORKER_ID,
            Arc::new(NullResources),
            StorageType::Pinned,
            None,
        );

        let block_data = factory.create_block_data(0).unwrap();
        assert_eq!(block_data.block_id(), 0);
        assert_eq!(block_data.block_set_id(), TEST_BLOCK_SET_ID);
        assert_eq!(block_data.worker_id(), TEST_WORKER_ID);
        assert_eq!(block_data.storage_type(), &StorageType::Pinned);

        let _resources = block_data.resources();

        let blocks = factory
            .into_blocks_with_metadata(BasicMetadata::default())
            .unwrap();

        ManagedBlockPool::builder().blocks(blocks).build().unwrap();
    }
}
