// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

pub mod local;
pub mod logical;
pub mod view;

pub use local::LocalBlockData as BlockData;

pub trait BlockDataExt<S: Storage>: Send + Sync + 'static + std::fmt::Debug {
    /// The index of the block in the block set
    fn block_id(&self) -> BlockId;

    /// The index of the block in the block set
    fn fragment_block_id(&self, idx: usize) -> BlockId;

    /// The identifier of the block set within the worker
    fn block_set_id(&self) -> usize;

    /// The identifier of the worker that owns the block
    /// Note: If the block is a logical block, this will be the worker id of the worker
    /// that owns the logical block, not the worker id of the worker that owns the physical block
    /// because their could be multiple workers contributing to the same logical block.
    fn worker_id(&self) -> WorkerID;

    /// The storage type of the block
    fn storage_type(&self) -> &StorageType;

    /// Whether the block is fully contiguous
    fn is_fully_contiguous(&self) -> bool;

    /// Is the block fragmented
    fn is_fragmented(&self) -> bool;

    /// Returns the number of fragments
    fn num_fragments(&self) -> usize;

    /// Returns the number of layers in the block
    fn num_layers(&self) -> usize;

    /// The size of the page in the block
    fn page_size(&self) -> usize;

    /// Returns the number of outer dimensions in the block
    fn num_outer_dims(&self) -> usize;

    fn num_inner_dims(&self) -> usize;

    /// Whether or not one can acquire read-only views to the block's storage
    fn is_local(&self) -> Option<&dyn BlockDataViews<S>>;

    /// Whether or not one can acquire mutable views to the block's storage
    fn is_local_mut(&mut self) -> Option<&mut dyn BlockDataViews<S>>;

    /// Get a read-only view of this block's storage for a layer
    fn layer_view(
        &self,
        layer_idx: usize,
        outer_idx: usize,
    ) -> BlockResult<view::LayerView<'_, S>> {
        match self.is_local() {
            Some(views) => views.local_layer_view(layer_idx, outer_idx),
            None => Err(BlockError::ViewsNotAvailableOnLogicalBlocks),
        }
    }

    /// Get a read-only view of this block's storage for a layer
    fn layer_view_fragment(
        &self,
        fragment_idx: usize,
        layer_idx: usize,
        outer_idx: usize,
    ) -> BlockResult<view::LayerView<'_, S>> {
        match self.is_local() {
            Some(views) => views.local_layer_view_fragment(fragment_idx, layer_idx, outer_idx),
            None => Err(BlockError::ViewsNotAvailableOnLogicalBlocks),
        }
    }

    /// Get a mutable view of this block's storage for a layer
    fn layer_view_mut(
        &mut self,
        layer_idx: usize,
        outer_idx: usize,
    ) -> BlockResult<view::LayerViewMut<'_, S>> {
        match self.is_local_mut() {
            Some(views) => views.local_layer_view_mut(layer_idx, outer_idx),
            None => Err(BlockError::ViewsNotAvailableOnLogicalBlocks),
        }
    }

    /// Get a mutable view of this block's storage for a layer
    fn layer_view_fragment_mut(
        &mut self,
        fragment_idx: usize,
        layer_idx: usize,
        outer_idx: usize,
    ) -> BlockResult<view::LayerViewMut<'_, S>> {
        match self.is_local_mut() {
            Some(views) => views.local_layer_view_fragment_mut(fragment_idx, layer_idx, outer_idx),
            None => Err(BlockError::ViewsNotAvailableOnLogicalBlocks),
        }
    }

    /// Get a read-only view of this block's storage
    fn block_view(&self) -> BlockResult<view::BlockView<'_, S>> {
        match self.is_local() {
            Some(views) => views.local_block_view(),
            None => Err(BlockError::ViewsNotAvailableOnLogicalBlocks),
        }
    }

    /// Get a mutable view of this block's storage
    fn block_view_mut(&mut self) -> BlockResult<view::BlockViewMut<'_, S>> {
        match self.is_local_mut() {
            Some(views) => views.local_block_view_mut(),
            None => Err(BlockError::ViewsNotAvailableOnLogicalBlocks),
        }
    }
}

pub trait BlockDataViews<S: Storage> {
    /// Get a read-only view of this block's storage for a layer
    fn local_layer_view(
        &self,
        layer_idx: usize,
        outer_idx: usize,
    ) -> BlockResult<view::LayerView<'_, S>>;

    /// Get a read-only view of this block's fragment storage for a layer
    fn local_layer_view_fragment(
        &self,
        fragment_idx: usize,
        layer_idx: usize,
        outer_idx: usize,
    ) -> BlockResult<view::LayerView<'_, S>>;

    /// Get a mutable view of this block's storage for a layer
    fn local_layer_view_mut(
        &mut self,
        layer_idx: usize,
        outer_idx: usize,
    ) -> BlockResult<view::LayerViewMut<'_, S>>;

    /// Get a mutable view of this block's fragment storage for a layer
    fn local_layer_view_fragment_mut(
        &mut self,
        fragment_idx: usize,
        layer_idx: usize,
        outer_idx: usize,
    ) -> BlockResult<view::LayerViewMut<'_, S>>;

    /// Get a read-only view of this block's storage
    fn local_block_view(&self) -> BlockResult<view::BlockView<'_, S>>;

    /// Get a mutable view of this block's storage
    fn local_block_view_mut(&mut self) -> BlockResult<view::BlockViewMut<'_, S>>;
}

pub trait BlockDataProvider: StorageTypeProvider {
    type Locality: LocalityProvider;

    fn block_data(&self) -> &impl BlockDataExt<Self::StorageType>;
}

pub trait BlockDataProviderMut: BlockDataProvider {
    type Locality: LocalityProvider;

    fn block_data_mut(&mut self) -> &mut impl BlockDataExt<Self::StorageType>;
}
