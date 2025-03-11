// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! While KV blocks can be formed from any memory allocation, we highly encourage you to
//! use large slabs of pinned or device memory.
//!
//! The primary reason for this to efficiently map to the RDMA transport layers which perform
//! better and have less overheads when using fewer large regions of registered memory vs many
//! smaller regions.
//!
//! To this end, we encourage the developer if using the BYO-memory option to allocate either
//! a single large tensor or a set of tensors, one per layer, to effectively map to the NIXL
//! dataplane.

use cudarc::driver::{CudaContext, CudaStream};
use derive_builder::Builder;
use dynemo_runtime::{error, raise, ErrorContext, Result};
use std::sync::Arc;
use validator::{Validate, ValidationError};

use super::storage::{DType, OwnedStorage, Storage, StorageType, TensorView};
extern "C" {
    fn copy_blocks_3d(
        src_data: *const std::ffi::c_void,
        dst_data: *mut std::ffi::c_void,
        h_src_block_ids: *const std::os::raw::c_int,
        h_dst_block_ids: *const std::os::raw::c_int,
        num_block_pairs: std::os::raw::c_int,
        prefix_dim: std::os::raw::c_int,
        src_blocks: std::os::raw::c_int,
        dst_blocks: std::os::raw::c_int,
        suffix_dim: std::os::raw::c_int,
        elem_size: std::os::raw::c_int,
    ) -> std::os::raw::c_int;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvLayout {
    /// Tensor is laid out as [kv, block, head, head_dim]
    KvFirst,

    /// Tensor is laid out as [block, kv, head, head_dim]
    BlockFirst,
}

#[derive(Debug, Clone, Builder)]
pub struct KvModelDetails {
    /// The number of layers in the model
    number_of_layers: usize,

    /// The number of heads in the tensor
    number_of_heads: usize,

    /// The size of each head in the tensor
    head_size: usize,

    /// Data type of the tensor
    dtype: DType,
}

impl KvModelDetails {
    pub fn number_of_elements_per_token_per_layer(&self) -> usize {
        2 * self.number_of_heads * self.head_size
    }

    pub fn bytes_per_token_per_layer(&self) -> usize {
        self.number_of_elements_per_token_per_layer() * self.dtype.size_in_bytes()
    }

    // pub fn number_of_elements_per_token(&self) -> usize {
    //     self.number_of_elements_per_token_per_layer() * self.number_of_layers
    // }

    // pub fn bytes_per_token(&self) -> usize {
    //     self.number_of_elements_per_token() * self.dtype.size_in_bytes()
    // }
}

#[derive(Debug, Clone, Builder, Validate)]
#[validate(schema(function = "validate_block_details", skip_on_field_errors = true))]
pub struct KvBlockDetails {
    /// The layout of the tensor
    layout: KvLayout,

    /// The size of each block in the tensor
    block_size: usize,

    /// The rank of the current process in the tensor parallel group
    #[builder(default = "0")]
    tp_ranks: usize,

    /// The size of the tensor parallel group
    #[builder(default = "1")]
    tp_size: usize,

    /// The details of the model
    model_details: KvModelDetails,
}

impl KvBlockDetails {
    pub fn bytes_per_token_block_per_layer(&self) -> usize {
        (self.model_details.bytes_per_token_per_layer() * self.block_size) / self.tp_size
    }
}

fn validate_block_details(block_details: &KvBlockDetails) -> Result<(), ValidationError> {
    // tp size must evenly divide the number of heads
    if block_details.model_details.number_of_heads % block_details.tp_size != 0 {
        return Err(ValidationError::new("tp_size must evenly divide num_heads"));
    }

    Ok(())
}

#[derive(Debug)]
pub struct KvBlockStorage {
    /// The details of the model
    block_details: KvBlockDetails,

    /// The type of storage
    storage_type: StorageType,

    /// Optional cuda device
    cuda_device: Option<Arc<CudaContext>>,

    /// Number of blocks
    number_of_blocks: usize,

    /// Layers
    layers: Vec<KvLayer>,
}

impl KvBlockStorage {
    pub fn new(layers: Vec<KvLayer>) -> Result<Self> {
        if layers.is_empty() {
            raise!("Layers must not be empty");
        }

        // validate all layers have the same type
        let storage_type = layers[0].storage.storage_type();

        let cuda_device = match storage_type {
            StorageType::Device(device) => Some(CudaContext::new(device)?),
            _ => None,
        };

        for layer in &layers {
            if layer.storage.storage_type() != storage_type {
                raise!("All layers must have the same storage type");
            }
        }

        // validate all layers have the same number of blocks
        let number_of_blocks = layers[0].number_of_blocks;
        for layer in &layers {
            if layer.number_of_blocks != number_of_blocks {
                raise!("All layers must have the same number of blocks");
            }
        }

        // extract the details from the first layer, construct ModelDetails, BlockDetails
        let model_details = KvModelDetailsBuilder::default()
            .number_of_layers(layers.len())
            .number_of_heads(layers[0].number_of_heads)
            .head_size(layers[0].head_size)
            .dtype(layers[0].dtype)
            .build()?;

        let block_details = KvBlockDetailsBuilder::default()
            .layout(layers[0].layout.clone())
            .block_size(layers[0].block_size)
            .model_details(model_details.clone())
            .tp_size(layers[0].tp_size)
            .tp_ranks(layers[0].tp_rank)
            .build()?;

        block_details.validate()?;

        let bytes_per_token_block = block_details.bytes_per_token_block_per_layer();

        let storage_type = layers[0].storage.storage_type();

        // validate all layers have enough capacity to store hold the block data
        for layer in &layers {
            if layer.storage.storage_size() < bytes_per_token_block {
                raise!("All layers must have enough capacity to store hold the block data");
            }
        }

        Ok(Self {
            block_details,
            storage_type,
            number_of_blocks,
            cuda_device,
            layers,
        })
    }

    pub fn allocate(
        number_of_blocks: usize,
        block_details: KvBlockDetails,
        storage_type: StorageType,
    ) -> Result<Self> {
        block_details.validate()?;

        // determine the number of blocks
        let bytes = block_details.bytes_per_token_block_per_layer() * number_of_blocks;

        let mut layers = Vec::new();

        // for each layer, create a device storage object, then for a kv layer
        for layer in 0..block_details.model_details.number_of_layers {
            let storage = OwnedStorage::create(bytes, storage_type).with_context(|| {
                error!("Failed to allocate memory for KV BlockStorage for layer {layer}")
            })?;
            let layer = KvLayer::from_storage(&block_details, number_of_blocks, storage);
            layers.push(layer);
        }

        Self::new(layers).context("Validating KvBlockStorage")
    }

    /// Get an immutable reference to a layer
    pub fn layer(&self, layer: usize) -> Result<&KvLayer> {
        if layer >= self.layers.len() {
            raise!(
                "Layer index {} out of bounds (max {})",
                layer,
                self.layers.len() - 1
            );
        }
        Ok(&self.layers[layer])
    }

    /// Get a mutable reference to a layer
    pub fn layer_mut(&mut self, layer: usize) -> Result<&mut KvLayer> {
        if layer >= self.layers.len() {
            raise!(
                "Layer index {} out of bounds (max {})",
                layer,
                self.layers.len() - 1
            );
        }
        Ok(&mut self.layers[layer])
    }
}

#[derive(Debug, Builder)]
pub struct KvLayer {
    /// The layout of the tensor
    layout: KvLayout,

    /// The storage of the tensor
    storage: OwnedStorage,

    /// The number of blocks in the tensor
    number_of_blocks: usize,

    /// The size of each block in the tensor
    block_size: usize,

    /// The number of heads in the tensor
    number_of_heads: usize,

    /// The size of each head in the tensor
    head_size: usize,

    /// DataType
    dtype: DType,

    /// The tensor parallel size (default is 1)
    #[builder(default = 1)]
    tp_size: usize,

    /// The tensor parallel rank (default is 0)
    #[builder(default = 0)]
    tp_rank: usize,
}

impl Storage for KvLayer {
    fn storage_type(&self) -> StorageType {
        self.storage.storage_type()
    }

    fn get_pointer(&self) -> u64 {
        self.storage.get_pointer()
    }

    fn storage_size(&self) -> usize {
        self.storage.storage_size()
    }
}

impl KvLayer {
    fn from_storage(
        block_details: &KvBlockDetails,
        number_of_blocks: usize,
        storage: OwnedStorage,
    ) -> Self {
        Self {
            storage,
            number_of_blocks,
            layout: block_details.layout.clone(),
            block_size: block_details.block_size,
            number_of_heads: block_details.model_details.number_of_heads,
            head_size: block_details.model_details.head_size,
            dtype: block_details.model_details.dtype,
            tp_size: block_details.tp_size,
            tp_rank: block_details.tp_ranks,
        }
    }
    pub fn view(&self) -> Result<TensorView<'_, Self, 5>> {
        // Calculate dimensions based on layout
        let dims = match self.layout {
            KvLayout::KvFirst => [
                2, // K and V as first dimension
                self.number_of_blocks,
                self.block_size,
                self.number_of_heads / self.tp_size,
                self.head_size,
            ],
            KvLayout::BlockFirst => [
                self.number_of_blocks,
                2, // K and V as second dimension
                self.block_size,
                self.number_of_heads / self.tp_size,
                self.head_size,
            ],
        };

        // Verify dimensions make sense
        if self.number_of_heads % self.tp_size != 0 {
            raise!(
                "Number of heads ({}) is not divisible by tp_size ({})",
                self.number_of_heads,
                self.tp_size
            );
        }

        // Log dimensions for debugging
        tracing::debug!(
            "Creating TensorView with dims: {:?}, dtype: {:?}, size: {}",
            dims,
            self.dtype,
            self.dtype.size_in_bytes()
        );
        // Create and return the view
        let view = TensorView::new(self, dims, self.dtype.size_in_bytes())
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        Ok(view)
    }

    pub fn copy_blocks_to(
        &self,
        src_block_ids: &[usize],
        dst: &mut KvLayer,
        dst_block_ids: &[usize],
        stream: &CudaStream,
    ) -> Result<()> {
        if src_block_ids.len() != dst_block_ids.len() {
            raise!("src_block_ids and dst_block_ids must have the same length");
        }

        if self.layout != dst.layout {
            raise!("src and dst must have the same layout");
        }

        match (self.storage.storage_type(), dst.storage.storage_type()) {
            (StorageType::Pinned, StorageType::Pinned) => {
                raise!("Pinned to Pinned copy not implemented");
            }
            (StorageType::Pinned, StorageType::Device(_)) => {}
            (StorageType::Device(_), StorageType::Pinned) => {}
            (StorageType::Device(_), StorageType::Device(_)) => {
                raise!("Device to Device copy not implemented");
            }
            (StorageType::System, _) => {
                raise!("System to Device copy not implemented");
            }
            (_, StorageType::System) => {
                raise!("Device to System copy not implemented");
            }
        };

        let h_src_block_ids = src_block_ids
            .iter()
            .map(|id| *id as i32)
            .collect::<Vec<_>>();

        let h_dst_block_ids = dst_block_ids
            .iter()
            .map(|id| *id as i32)
            .collect::<Vec<_>>();

        let num_block_pairs = src_block_ids.len() as i32;

        let prefix_dim = match self.layout {
            KvLayout::KvFirst => 2,
            KvLayout::BlockFirst => 1,
        };

        let suffix_dim = self.head_size * (self.number_of_heads / self.tp_size) * self.block_size;
        let suffix_dim = match self.layout {
            KvLayout::KvFirst => suffix_dim,
            KvLayout::BlockFirst => 2 * suffix_dim,
        };

        let elem_size = self.dtype.size_in_bytes();

        let src_blocks = self.number_of_blocks as i32;
        let dst_blocks = dst.number_of_blocks as i32;

        unsafe {
            let rc = copy_blocks_3d(
                self.storage.get_pointer() as *const std::ffi::c_void,
                dst.storage.get_pointer() as *mut std::ffi::c_void,
                h_src_block_ids.as_ptr() as *const std::os::raw::c_int,
                h_dst_block_ids.as_ptr() as *const std::os::raw::c_int,
                num_block_pairs,
                prefix_dim as i32,
                src_blocks,
                dst_blocks,
                suffix_dim as i32,
                elem_size as i32,
            );

            if rc != 0 {
                raise!("Failed to copy blocks");
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[test]
    fn test_kv_block_storage_kv_first() -> Result<()> {
        let model_details = KvModelDetailsBuilder::default()
            .number_of_layers(2)
            .number_of_heads(4)
            .head_size(8)
            .dtype(DType::F32)
            .build()?;

        let block_details = KvBlockDetailsBuilder::default()
            .layout(KvLayout::KvFirst)
            .block_size(8)
            .tp_size(1)
            .tp_ranks(0)
            .model_details(model_details)
            .build()?;

        // Create the storage blocks
        let mut h_blocks =
            KvBlockStorage::allocate(32, block_details.clone(), StorageType::Pinned)?;

        let mut d_blocks =
            KvBlockStorage::allocate(32, block_details.clone(), StorageType::Device(0))?;

        println!("Allocated pinned and device blocks");
        println!("Letting layer 0 on host to be 1s");

        // Use separate scopes to manage borrows
        {
            // Get a mutable reference to a layer
            let layer = h_blocks.layer_mut(0)?;

            // Create a mutable view and work with it
            let mut view = layer.view()?;

            // Get shape information before creating the ndarray view
            let shape = *view.shape();
            println!("TensorView shape: {:?}", shape);

            // Create and use the mutable ndarray view in its own scope
            {
                let mut nd_view = view.as_ndarray_view_mut::<f32>()?;
                let ones = ndarray::Array::from_shape_fn(nd_view.dim(), |_| 1.0);
                nd_view.assign(&ones);

                // Verify some values while we have the view
                assert_eq!(nd_view[[0, 0, 0, 0, 0]], 1.0);
                assert_eq!(nd_view[[1, 0, 0, 0, 0]], 1.0);
            }
            // nd_view is dropped here, releasing the mutable borrow
        }

        // Copy data to device
        let context = CudaContext::new(0)?;
        let stream = context.new_stream()?;

        println!("Copying data to device");

        {
            let h_view = h_blocks.layer(0)?.view().unwrap();
            let mut d_view = d_blocks.layer_mut(0)?.view().unwrap();
            h_view.copy_to(&mut d_view, &stream).unwrap();
            stream.synchronize().unwrap();
        }

        println!("Setting all values on host back to 0");

        // Set all values on host back to 0
        {
            let mut h_layer = h_blocks.layer_mut(0)?.view()?;
            let mut nd_view = h_layer.as_ndarray_view_mut::<f32>()?;
            let zeros = ndarray::Array::from_shape_fn(nd_view.dim(), |_| 0.0);
            nd_view.assign(&zeros);

            assert_eq!(nd_view[[0, 0, 0, 0, 0]], 0.0);
            assert_eq!(nd_view[[1, 0, 0, 0, 0]], 0.0);
        }

        println!("Copying data back to host");

        // Copy data back to host
        {
            let d_view = d_blocks.layer(0)?.view()?;
            let mut h_view = h_blocks.layer_mut(0)?.view()?;
            d_view.copy_to(&mut h_view, &stream)?;
            stream.synchronize()?;
        }

        println!("Verifying host data is 1");

        // Verify the host data is not back to 1
        {
            let h_layer = h_blocks.layer(0)?.view()?;
            let nd_view = h_layer.as_ndarray_view::<f32>()?;
            assert_eq!(nd_view[[0, 0, 0, 0, 0]], 1.0);
            assert_eq!(nd_view[[1, 0, 0, 0, 0]], 1.0);
        }

        Ok(())
    }

    #[rstest]
    #[test]
    fn test_kv_block_storage_kv_first_direct() -> Result<()> {
        let model_details = KvModelDetailsBuilder::default()
            .number_of_layers(2)
            .number_of_heads(4)
            .head_size(8)
            .dtype(DType::F32)
            .build()?;

        let block_details = KvBlockDetailsBuilder::default()
            .layout(KvLayout::KvFirst)
            .block_size(8)
            .tp_size(1)
            .tp_ranks(0)
            .model_details(model_details)
            .build()?;

        // Create the storage blocks
        let mut h_blocks =
            KvBlockStorage::allocate(32, block_details.clone(), StorageType::Pinned)?;

        let mut d_blocks =
            KvBlockStorage::allocate(32, block_details.clone(), StorageType::Device(0))?;

        println!("Allocated pinned and device blocks");
        println!("Letting layer 0 on host to be 1s");

        // Use separate scopes to manage borrows
        {
            // Get a mutable reference to a layer
            let layer = h_blocks.layer_mut(0)?;

            // Create a mutable view and work with it
            let mut view = layer.view()?;

            // Get shape information before creating the ndarray view
            let shape = *view.shape();
            println!("TensorView shape: {:?}", shape);

            // Create and use the mutable ndarray view in its own scope
            {
                let mut nd_view = view.as_ndarray_view_mut::<f32>()?;
                let ones = ndarray::Array::from_shape_fn(nd_view.dim(), |_| 1.0);
                nd_view.assign(&ones);

                // Verify some values while we have the view
                assert_eq!(nd_view[[0, 0, 0, 0, 0]], 1.0);
                assert_eq!(nd_view[[1, 0, 0, 0, 0]], 1.0);
            }
            // nd_view is dropped here, releasing the mutable borrow
        }

        // Copy data to device
        let context = CudaContext::new(0)?;
        let stream = context.new_stream()?;

        println!("Copying data to device");

        {
            let blocks = (0..32).collect::<Vec<_>>();

            let h_layer = h_blocks.layer(0).unwrap();
            let mut d_layer = d_blocks.layer_mut(0).unwrap();
            h_layer
                .copy_blocks_to(&blocks, &mut d_layer, &blocks, &stream)
                .unwrap();
            stream.synchronize().unwrap();
        }

        println!("Setting all values on host back to 0");

        // Set all values on host back to 0
        {
            let mut h_layer = h_blocks.layer_mut(0)?.view()?;
            let mut nd_view = h_layer.as_ndarray_view_mut::<f32>()?;
            let zeros = ndarray::Array::from_shape_fn(nd_view.dim(), |_| 0.0);
            nd_view.assign(&zeros);

            assert_eq!(nd_view[[0, 0, 0, 0, 0]], 0.0);
            assert_eq!(nd_view[[1, 0, 0, 0, 0]], 0.0);
        }

        println!("Copying data back to host");

        // Copy data back to host
        {
            let blocks = (0..32).collect::<Vec<_>>();

            let mut h_layer = h_blocks.layer_mut(0).unwrap();
            let d_layer = d_blocks.layer(0).unwrap();
            d_layer
                .copy_blocks_to(&blocks, &mut h_layer, &blocks, &stream)
                .unwrap();
            stream.synchronize().unwrap();
        }

        println!("Verifying host data is 1");

        // Verify the host data is not back to 1
        {
            let h_layer = h_blocks.layer(0)?.view()?;
            let nd_view = h_layer.as_ndarray_view::<f32>()?;
            assert_eq!(nd_view[[0, 0, 0, 0, 0]], 1.0);
            assert_eq!(nd_view[[1, 0, 0, 0, 0]], 1.0);
        }

        Ok(())
    }

    #[rstest]
    #[test]
    fn test_kv_block_storage_kv_first_direct_with_block_vals() -> Result<()> {
        let number_of_blocks = 8;

        let model_details = KvModelDetailsBuilder::default()
            .number_of_layers(2)
            .number_of_heads(2)
            .head_size(2)
            .dtype(DType::F32)
            .build()?;

        let block_details = KvBlockDetailsBuilder::default()
            .layout(KvLayout::KvFirst)
            .block_size(4)
            .tp_size(1)
            .tp_ranks(0)
            .model_details(model_details)
            .build()?;

        // Create the storage blocks
        let mut h_blocks =
            KvBlockStorage::allocate(number_of_blocks, block_details.clone(), StorageType::Pinned)?;

        let mut d_blocks = KvBlockStorage::allocate(
            number_of_blocks,
            block_details.clone(),
            StorageType::Device(0),
        )?;

        println!("Allocated pinned and device blocks");
        println!("Letting layer 0 on host to be 1s");

        let layout = h_blocks.layer(0).unwrap().layout.clone();
        let shape = h_blocks.layer(0).unwrap().view().unwrap().shape().clone();

        println!("shape: {:?}", shape);

        // Use separate scopes to manage borrows
        {
            // Get a mutable reference to a layer
            let layer = h_blocks.layer_mut(0)?;

            // Create a mutable view and work with it
            let mut view = layer.view()?;

            // Create and use the mutable ndarray view in its own scope
            {
                let mut nd_view = view.as_ndarray_view_mut::<f32>()?;

                // iter over nd_view and set the values equal the the block index
                // all kv and v of block 42 have values 42
                match layout {
                    KvLayout::KvFirst => {
                        for kv_idx in 0..shape[0] {
                            for block_idx in 0..shape[1] {
                                for bs_idx in 0..shape[2] {
                                    for head_idx in 0..shape[3] {
                                        for head_dim_idx in 0..shape[4] {
                                            nd_view[[
                                                kv_idx,
                                                block_idx,
                                                bs_idx,
                                                head_idx,
                                                head_dim_idx,
                                            ]] = block_idx as f32;
                                        }
                                    }
                                }
                            }
                        }

                        assert_eq!(nd_view[[0, 0, 0, 0, 0]], 0.0);
                        assert_eq!(nd_view[[1, 0, 0, 0, 0]], 0.0);
                        assert_eq!(nd_view[[1, 0, 1, 1, 1]], 0.0);
                        assert_eq!(nd_view[[0, 2, 0, 0, 0]], 2.0);
                        assert_eq!(nd_view[[1, 2, 0, 0, 0]], 2.0);
                        assert_eq!(nd_view[[1, 2, 1, 1, 1]], 2.0);
                    }
                    KvLayout::BlockFirst => {
                        for block_idx in 0..shape[0] {
                            for kv_idx in 0..shape[1] {
                                for bs_idx in 0..shape[2] {
                                    for head_idx in 0..shape[3] {
                                        for head_dim_idx in 0..shape[4] {
                                            nd_view[[block_idx, kv_idx, head_idx, head_dim_idx]] =
                                                block_idx as f32;
                                        }
                                    }
                                }
                            }
                        }

                        assert_eq!(nd_view[[0, 0, 0, 0, 0]], 0.0);
                        assert_eq!(nd_view[[0, 1, 1, 1, 1]], 0.0);

                        assert_eq!(nd_view[[1, 0, 0, 0, 0]], 1.0);
                        assert_eq!(nd_view[[1, 1, 1, 1, 1]], 1.0);
                    }
                }
            }
            // nd_view is dropped here, releasing the mutable borrow
        }

        // Copy data to device
        let context = CudaContext::new(0)?;
        let stream = context.new_stream()?;

        println!("Copying data to device");

        {
            let blocks = (0..number_of_blocks).collect::<Vec<_>>();

            let h_layer = h_blocks.layer(0).unwrap();
            let mut d_layer = d_blocks.layer_mut(0).unwrap();
            h_layer
                .copy_blocks_to(&blocks, &mut d_layer, &blocks, &stream)
                .unwrap();
            stream.synchronize().unwrap();
        }

        println!("Setting all values on host back to 0");

        // Set all values on host back to 0
        {
            let mut h_layer = h_blocks.layer_mut(0)?.view()?;
            let mut nd_view = h_layer.as_ndarray_view_mut::<f32>()?;
            let zeros = ndarray::Array::from_shape_fn(nd_view.dim(), |_| 0.0);
            nd_view.assign(&zeros);

            assert_eq!(nd_view[[0, 0, 0, 0, 0]], 0.0);
            assert_eq!(nd_view[[1, 0, 0, 0, 0]], 0.0);
            assert_eq!(nd_view[[0, 1, 1, 1, 1]], 0.0);
            assert_eq!(nd_view[[1, 1, 1, 1, 1]], 0.0);
        }

        println!("Copying data back to host");

        let src_blocks = &[1, 2, 2, 3, 5];
        let dst_blocks = &[0, 3, 2, 1, 4];

        // Copy data back to host
        {
            let mut h_layer = h_blocks.layer_mut(0).unwrap();
            let d_layer = d_blocks.layer(0).unwrap();
            d_layer
                .copy_blocks_to(src_blocks, &mut h_layer, dst_blocks, &stream)
                .unwrap();
            stream.synchronize().unwrap();
        }

        println!("Verifying host data is 1");

        // Verify the host data is not back to 1
        {
            let h_layer = h_blocks.layer(0)?.view()?;
            let nd_view = h_layer.as_ndarray_view::<f32>()?;

            println!("nd_view: {:?}", nd_view);

            // validate

            for i in 0..src_blocks.len() {
                println!(
                    "Validating src block {} -> dst block {}",
                    src_blocks[i], dst_blocks[i]
                );
                let expected_value = src_blocks[i] as f32;
                match layout {
                    KvLayout::KvFirst => {
                        assert_eq!(nd_view[[0, dst_blocks[i], 0, 0, 0]], expected_value);
                        assert_eq!(nd_view[[1, dst_blocks[i], 0, 0, 0]], expected_value);
                        assert_eq!(nd_view[[0, dst_blocks[i], 1, 1, 1]], expected_value);
                        assert_eq!(nd_view[[1, dst_blocks[i], 1, 1, 1]], expected_value);
                    }
                    KvLayout::BlockFirst => {
                        assert_eq!(nd_view[[dst_blocks[i], 0, 0, 0, 0]], expected_value);
                        assert_eq!(nd_view[[dst_blocks[i], 0, 1, 1, 1]], expected_value);
                        assert_eq!(nd_view[[dst_blocks[i], 1, 0, 0, 0]], expected_value);
                        assert_eq!(nd_view[[dst_blocks[i], 1, 1, 1, 1]], expected_value);
                    }
                }
            }
        }

        Ok(())
    }
}
