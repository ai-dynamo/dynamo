// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

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

impl<S: Storage> IntoBlocks<S, locality::Local> for LocalBlockDataFactory<S> {}
