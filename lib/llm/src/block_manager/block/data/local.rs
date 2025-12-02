// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use smallvec::SmallVec;

/// Individual block storage
#[derive(Debug)]
pub struct LocalBlockData<S: Storage> {
    layout: Arc<dyn BlockLayout<StorageType = S>>,
    // The special case of multiple block IDx will only happen for small KV cache blocks on device.
    // offloaded blocks will be large and `block_idxs.len() == 1` will hold for them.
    block_idxs: SmallVec<[usize; 1]>,
    block_set_idx: usize,
    worker_id: WorkerID,
}

impl<S: Storage> Clone for LocalBlockData<S> {
    fn clone(&self) -> Self {
        Self {
            layout: self.layout.clone(),
            block_idxs: self.block_idxs.clone(),
            block_set_idx: self.block_set_idx,
            worker_id: self.worker_id,
        }
    }
}

pub struct LocalBlockDataBase<S: Storage> {
    layout: Arc<dyn BlockLayout<StorageType = S>>,
    block_set_idx: usize,
    worker_id: WorkerID,
}

impl<S: Storage> Clone for LocalBlockDataBase<S> {
    fn clone(&self) -> Self {
        Self {
            layout: self.layout.clone(),
            block_set_idx: self.block_set_idx,
            worker_id: self.worker_id,
        }
    }
}

impl<S: Storage> LocalBlockDataBase<S> {
    pub(crate) fn get_data(&self, block_idxs: SmallVec<[usize; 1]>) -> LocalBlockData<S> {
        LocalBlockData {
            layout: self.layout.clone(),
            block_idxs,
            block_set_idx: self.block_set_idx,
            worker_id: self.worker_id,
        }
    }
}

impl<S> LocalBlockData<S>
where
    S: Storage,
{
    /// Create a new block storage
    pub(crate) fn new(
        layout: Arc<dyn BlockLayout<StorageType = S>>,
        block_idx: usize,
        block_set_idx: usize,
        worker_id: WorkerID,
    ) -> Self {
        Self {
            layout,
            block_idxs: [block_idx].iter().map(|x| *x).collect(),
            block_set_idx,
            worker_id,
        }
    }

    pub(crate) fn base(&self) -> LocalBlockDataBase<S> {
        LocalBlockDataBase {
            layout: self.layout.clone(),
            block_set_idx: self.block_set_idx,
            worker_id: self.worker_id.clone(),
        }
    }
}

impl<S: Storage> BlockDataExt<S> for LocalBlockData<S>
where
    S: Storage,
{
    #[inline(always)]
    fn block_id(&self) -> BlockId {
        if self.block_idxs.len() == 1 {
            self.block_idxs[0]
        } else {
            tracing::error!("Backtrace: {}", std::backtrace::Backtrace::force_capture());
            panic!("used LocalBlockData::block_id() for fragmented block");
        }
    }

    #[inline(always)]
    fn fragment_block_id(&self, idx: usize) -> BlockId {
        if self.block_idxs.len() != 1 {
            self.block_idxs[idx]
        } else {
            tracing::error!("Backtrace: {}", std::backtrace::Backtrace::force_capture());
            panic!("used LocalBlockData::fragment_block_id() for non--fragmented block");
        }
    }

    #[inline(always)]
    fn block_set_id(&self) -> usize {
        self.block_set_idx
    }

    #[inline(always)]
    fn worker_id(&self) -> WorkerID {
        self.worker_id
    }

    #[inline(always)]
    fn storage_type(&self) -> &StorageType {
        self.layout.storage_type()
    }

    fn is_fully_contiguous(&self) -> bool {
        self.layout.layout_type() == LayoutType::FullyContiguous
    }

    fn is_fragmented(&self) -> bool {
        self.block_idxs.len() != 1
    }

    fn num_fragments(&self) -> usize {
        self.block_idxs.len()
    }

    fn num_layers(&self) -> usize {
        self.layout.num_layers()
    }

    fn num_outer_dims(&self) -> usize {
        self.layout.outer_dim()
    }

    fn num_inner_dims(&self) -> usize {
        self.layout.inner_dim()
    }

    fn page_size(&self) -> usize {
        self.layout.page_size()
    }

    fn is_local(&self) -> Option<&dyn BlockDataViews<S>> {
        Some(self)
    }

    fn is_local_mut(&mut self) -> Option<&mut dyn BlockDataViews<S>> {
        Some(self)
    }
}

impl<S: Storage> BlockDataViews<S> for LocalBlockData<S> {
    fn local_layer_view(
        &self,
        layer_idx: usize,
        outer_idx: usize,
    ) -> BlockResult<view::LayerView<'_, S>> {
        let mr = self
            .layout
            .memory_region(self.block_id(), layer_idx, outer_idx)?;
        let storage_type = mr.storage_type();
        unsafe { view::LayerView::new(self, mr.addr(), mr.size(), storage_type) }
    }

    fn local_layer_view_fragment(
        &self,
        fragment_idx: usize,
        layer_idx: usize,
        outer_idx: usize,
    ) -> BlockResult<view::LayerView<'_, S>> {
        let mr = self.layout.memory_region(
            self.fragment_block_id(fragment_idx),
            layer_idx,
            outer_idx,
        )?;
        let storage_type = mr.storage_type();
        unsafe { view::LayerView::new(self, mr.addr(), mr.size(), storage_type) }
    }

    fn local_layer_view_mut(
        &mut self,
        layer_idx: usize,
        outer_idx: usize,
    ) -> BlockResult<view::LayerViewMut<'_, S>> {
        let mr = self
            .layout
            .memory_region(self.block_id(), layer_idx, outer_idx)?;
        unsafe { view::LayerViewMut::new(self, mr.addr(), mr.size(), mr.storage_type()) }
    }

    fn local_layer_view_fragment_mut(
        &mut self,
        fragment_idx: usize,
        layer_idx: usize,
        outer_idx: usize,
    ) -> BlockResult<view::LayerViewMut<'_, S>> {
        let mr = self.layout.memory_region(
            self.fragment_block_id(fragment_idx),
            layer_idx,
            outer_idx,
        )?;
        unsafe { view::LayerViewMut::new(self, mr.addr(), mr.size(), mr.storage_type()) }
    }

    fn local_block_view(&self) -> BlockResult<view::BlockView<'_, S>> {
        if self.is_fully_contiguous() {
            let mr = self.layout.memory_region(self.block_id(), 0, 0)?;
            let offset = mr.addr();
            let size = mr.size()
                .checked_mul(self.num_layers())
                .and_then(|intermediate| intermediate.checked_mul(self.num_outer_dims()))
                .ok_or_else(|| {
                    BlockError::InvalidState(format!(
                        "Block size calculation overflow: region_size={} * layers={} * outer_dims={}",
                        mr.size(), self.num_layers(), self.num_outer_dims()
                    ))
                })?;
            let storage_type = mr.storage_type();
            unsafe { view::BlockView::new(self, offset, size, storage_type) }
        } else {
            Err(BlockError::InvalidState(
                "Block is not fully contiguous".to_string(),
            ))
        }
    }

    fn local_block_view_mut(&mut self) -> BlockResult<view::BlockViewMut<'_, S>> {
        if self.is_fully_contiguous() {
            let mr = self.layout.memory_region(self.block_id(), 0, 0)?;
            let offset = mr.addr();
            let size = mr.size()
                .checked_mul(self.num_layers())
                .and_then(|intermediate| intermediate.checked_mul(self.num_outer_dims()))
                .ok_or_else(|| {
                    BlockError::InvalidState(format!(
                        "Block size calculation overflow: region_size={} * layers={} * outer_dims={}",
                        mr.size(), self.num_layers(), self.num_outer_dims()
                    ))
                })?;
            let storage_type = mr.storage_type();
            unsafe { view::BlockViewMut::new(self, offset, size, storage_type) }
        } else {
            Err(BlockError::InvalidState(
                "Block is not fully contiguous".to_string(),
            ))
        }
    }
}

impl<S: Storage> StorageTypeProvider for LocalBlockData<S> {
    type StorageType = S;
}

impl<S: Storage> BlockDataProvider for LocalBlockData<S> {
    type Locality = locality::Local;

    fn block_data(&self) -> &impl BlockDataExt<Self::StorageType> {
        self
    }
}

impl<S: Storage> BlockDataProviderMut for LocalBlockData<S> {
    type Locality = locality::Local;

    fn block_data_mut(&mut self) -> &mut impl BlockDataExt<Self::StorageType> {
        self
    }
}

impl<S: Storage> Local for LocalBlockData<S> {}
