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
use dynemo_runtime::{error, raise, utils::pool::Returnable, ErrorContext, Result};
use std::{
    ptr::NonNull,
    sync::{Arc, Mutex},
};
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

    fn create_copy_stream(
        stream: *mut *mut std::ffi::c_void,
        num_layers: std::os::raw::c_int,
        num_blocks: std::os::raw::c_int,
    ) -> std::os::raw::c_int;

    fn copy_stream_prepare_block_ids(
        cs: *mut std::ffi::c_void,
        src_block_ids: *const std::os::raw::c_int,
        dst_block_ids: *const std::os::raw::c_int,
        num_block_pairs: std::os::raw::c_int,
    ) -> std::os::raw::c_int;

    fn copy_stream_launch(
        cs: *mut std::ffi::c_void,
        src_data: *const std::ffi::c_void,
        dst_data: *mut std::ffi::c_void,
        prefix_dim: std::os::raw::c_int,
        suffix_dim: std::os::raw::c_int,
        elem_size: std::os::raw::c_int,
        src_block_dim: std::os::raw::c_int,
        dst_block_dim: std::os::raw::c_int,
    ) -> std::os::raw::c_int;

    fn copy_stream_sync(cs: *mut std::ffi::c_void) -> std::os::raw::c_int;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvLayout {
    /// Tensor is laid out as [kv, block, head, head_dim]
    KvFirst,

    /// Tensor is laid out as [block, kv, head, head_dim]
    BlockFirst,
}

#[derive(Debug, Clone, Builder, PartialEq, Eq)]
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
    tp_rank: usize,

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

    pub fn is_compatible(&self, other: &KvBlockDetails) -> bool {
        self.layout == other.layout
            && self.block_size == other.block_size
            && self.tp_size == other.tp_size
            && self.model_details == other.model_details
    }
}

fn validate_block_details(block_details: &KvBlockDetails) -> Result<(), ValidationError> {
    // tp size must evenly divide the number of heads
    if block_details.model_details.number_of_heads % block_details.tp_size != 0 {
        return Err(ValidationError::new("tp_size must evenly divide num_heads"));
    }

    if block_details.tp_rank >= block_details.tp_size {
        return Err(ValidationError::new("tp_rank must be less than tp_size"));
    }

    if block_details.tp_size > block_details.model_details.number_of_heads {
        return Err(ValidationError::new("tp_size must be less than num_heads"));
    }

    Ok(())
}

#[derive(Debug, Builder, Validate)]
#[validate(schema(function = "validate_kv_layer", skip_on_field_errors = true))]
pub struct KvLayer {
    /// The layout of the tensor
    layout: KvLayout,

    /// The storage of the tensor
    storage: OwnedStorage,

    /// The number of blocks in the tensor
    #[validate(range(min = 1))]
    number_of_blocks: usize,

    /// The size of each block in the tensor
    #[validate(range(min = 1))]
    block_size: usize,

    /// The number of heads in the tensor of the canonical model
    /// The actual number for this layer is this number divided by tp_size
    #[validate(range(min = 1))]
    number_of_heads: usize,

    /// The size of each head in the tensor
    #[validate(range(min = 1))]
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

fn validate_kv_layer(layer: &KvLayer) -> Result<(), ValidationError> {
    if layer.number_of_heads % layer.tp_size != 0 {
        return Err(ValidationError::new(
            "number_of_heads must be divisible by tp_size",
        ));
    }

    if layer.tp_rank >= layer.tp_size {
        return Err(ValidationError::new("tp_rank must be less than tp_size"));
    }

    if layer.tp_size > layer.number_of_heads {
        return Err(ValidationError::new(
            "tp_size must be less than number_of_heads",
        ));
    }

    let dims = layer.layer_shape();
    let elements = dims.iter().product::<usize>();
    let bytes = elements * layer.dtype.size_in_bytes();

    if layer.storage.storage_size() < bytes {
        return Err(ValidationError::new(
            "storage must be at least as large as the layer",
        ));
    }

    Ok(())
}

impl KvLayer {}

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
    ) -> Result<Self> {
        let layer = Self {
            storage,
            number_of_blocks,
            layout: block_details.layout.clone(),
            block_size: block_details.block_size,
            number_of_heads: block_details.model_details.number_of_heads,
            head_size: block_details.model_details.head_size,
            dtype: block_details.model_details.dtype,
            tp_size: block_details.tp_size,
            tp_rank: block_details.tp_rank,
        };

        layer.validate()?;

        Ok(layer)
    }

    /// Get the shape of the layer
    pub fn layer_shape(&self) -> [usize; 5] {
        match self.layout {
            KvLayout::KvFirst => [
                2, // K and V as first dimension
                self.number_of_blocks,
                self.block_size,
                self.number_of_heads / self.tp_size,
                self.head_size,
            ],
            KvLayout::BlockFirst => [
                self.number_of_blocks,
                2,
                self.block_size,
                self.number_of_heads / self.tp_size,
                self.head_size,
            ],
        }
    }

    /// Get a view of the layer
    pub fn view(&self) -> Result<TensorView<'_, Self, 5>> {
        // Calculate dimensions based on layout
        let dims = self.layer_shape();

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

    /// Perform a copy of blocks from one layer to another
    /// This launch a cuda kernel to perform the copy
    pub fn copy_blocks_to(
        &self,
        src_block_ids: &[usize],
        dst: &mut KvLayer,
        dst_block_ids: &[usize],
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

/// This object holds a set of layers that are used to store the KV cache
impl KvBlockStorage {
    /// Create a new KvBlockStorage object
    /// This allows you to bring in a set of layers that are already allocated
    pub fn from_layers(layers: Vec<KvLayer>) -> Result<Self> {
        if layers.is_empty() {
            raise!("Layers must not be empty");
        }

        // validate all layers have the same type
        let storage_type = layers[0].storage.storage_type();

        let cuda_device = match &storage_type {
            StorageType::Device(device) => Some(device.clone()),
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
            .tp_rank(layers[0].tp_rank)
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

    /// Given a number of blocks and the block details, allocate the storage for the layers
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
            let storage = OwnedStorage::create(bytes, storage_type.clone()).with_context(|| {
                error!("Failed to allocate memory for KV BlockStorage for layer {layer}")
            })?;
            let layer = KvLayer::from_storage(&block_details, number_of_blocks, storage)?;
            layers.push(layer);
        }

        Self::from_layers(layers).context("Validating KvBlockStorage")
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

    // pub fn suffix_dim(&self) -> usize {
    //     let value = match self.block_details.layout {
    //         KvLayout::KvFirst => self.block_details.model_details.number_of_heads * self.block_details.block_size,

    //         // s![block_id, 0..block_size, 0..
    //         KvLayout::BlockFirst => 2 *
    //     };
    // }
}

/// This struct holds the details of the layers to be copied
/// We should not have to recompute this for each copy stream, simply once for each
/// block set and each direction.
///
/// If we have two block sets -- one host, one device, the we need two copies of
/// this object, one H2D copies and another for D2H copies.
///
/// If we direct address into flash storage, then we will need another pair for H2F
/// and F2H copies.
///
/// Note: When we register two block sets, we need to validate that the suffix dimensions
/// is equivalent in both sets.
///
/// We may in the future need to add src_suffix_dim, dst_suffix_dim, src_suffix_stride and
/// dst_suffix_stride to the details to support non-unit strides. Today, the copy kernel
/// does not support that.
#[derive(Debug, Clone, Builder, Default, Validate)]
#[validate(schema(
    function = "validate_copy_stream_layer_details",
    skip_on_field_errors = true
))]
pub struct CopyStreamLayerDetails {
    /// The source layer pointer
    src_layer_ptrs: Vec<u64>,

    /// The destination layer pointer
    dst_layer_ptrs: Vec<u64>,

    /// The non-contiguous dimension above the block_dimension
    #[validate(range(min = 1))]
    prefix_dim: i32,

    /// The size of the source blocks dimension in the layer shape
    #[validate(range(min = 1))]
    src_block_dim: i32,

    /// The size of the destination blocks dimension in the layer shape
    #[validate(range(min = 1))]
    dst_block_dim: i32,

    /// The contiguous dimension below the block_dimension
    #[validate(range(min = 1))]
    suffix_dim: i32,

    /// The element size in bytes
    #[validate(range(min = 1, max = 8))]
    elem_size: i32,
}

impl CopyStreamLayerDetails {
    pub fn new(src: &KvBlockDetails, dst: &KvBlockDetails) -> Result<Self> {
        if !src.is_compatible(dst) {
            return Err(error!("src and dst must have compatible block details"));
        }

        let suffix_dim = match &src.layout {
            KvLayout::KvFirst => src.model_details.number_of_heads * src.block_size,
            KvLayout::BlockFirst => 2 * src.model_details.number_of_heads * src.block_size,
        };

        let elem_size = src.model_details.dtype.size_in_bytes();

        unimplemented!()
    }
}

fn validate_copy_stream_layer_details(
    layer_details: &CopyStreamLayerDetails,
) -> Result<(), ValidationError> {
    if layer_details.src_layer_ptrs.is_empty() {
        return Err(ValidationError::new("src_layer_ptrs must not be empty"));
    }

    if layer_details.dst_layer_ptrs.is_empty() {
        return Err(ValidationError::new("dst_layer_ptrs must not be empty"));
    }

    if layer_details.src_layer_ptrs.len() != layer_details.dst_layer_ptrs.len() {
        return Err(ValidationError::new(
            "src_layer_ptrs and dst_layer_ptrs must have the same length",
        ));
    }
    Ok(())
}

#[derive(Debug)]
pub struct CopyStreamContext {
    /// Pointer to the C++ copy stream object
    c_handle: NonNull<std::ffi::c_void>,

    /// Maximum number of layers used to initialize the C++ object
    max_num_layers: usize,

    /// Maximum number of blocks used to initialize the C++ object
    max_num_blocks: usize,

    /// Whether the layers have been staged
    staged_layers: bool,

    /// Whether the block ids have been staged
    staged_block_ids: bool,

    /// Doorbells for each layer
    layer_doorbells: Vec<bool>,

    // block ids
    src_block_ids: Vec<i32>,
    dst_block_ids: Vec<i32>,

    // layer details
    layer_details: Arc<CopyStreamLayerDetails>,
}

impl CopyStreamContext {
    pub fn new(max_num_layers: usize, max_num_blocks: usize) -> Result<Self> {
        let mut c_handle = std::ptr::null_mut();
        let rc = unsafe {
            create_copy_stream(&mut c_handle, max_num_layers as i32, max_num_blocks as i32)
        };
        if rc != 0 {
            return Err(error!("Failed to create copy stream"));
        }

        let layer_doorbells = vec![false; max_num_layers];

        Ok(Self {
            c_handle: NonNull::new(c_handle).ok_or(error!("Failed to create copy stream"))?,
            max_num_layers,
            max_num_blocks,
            staged_layers: false,
            staged_block_ids: false,
            layer_doorbells,

            // block ids
            src_block_ids: Vec::new(),
            dst_block_ids: Vec::new(),

            // layer details
            layer_details: Arc::new(CopyStreamLayerDetails::default()),
        })
    }
}

unsafe impl Send for CopyStream {}
unsafe impl Sync for CopyStream {}

/// This object holds a stateful copy stream for the copy_blocks_3d kernel
/// Each instance will hold:
/// - device memory for the block ids
/// - a cuda stream
/// - a cuda event
pub struct CopyStream {
    state: Arc<Mutex<CopyStreamContext>>,
}

impl CopyStream {
    pub fn new(num_layers: usize, num_blocks: usize) -> Result<Self> {
        let state = CopyStreamContext::new(num_layers, num_blocks)?;
        Ok(Self {
            state: Arc::new(Mutex::new(state)),
        })
    }

    /// Prepare the layer pointers for the copy kernel
    /// See [CopyStreamLayerDetails] for more details
    /// - src_layer_ptrs: the source layer pointers
    /// - dst_layer_ptrs: the destination layer pointers
    /// - prefix_dim: the non-contiguous dimension above the block_dimension
    /// - src_blocks_dim: the size of the source blocks dimension in the layer shape
    /// - dst_blocks_dim: the size of the destination blocks dimension in the layer shape
    /// - suffix_dim: the contiguous dimension below the block_dimension
    /// - elem_size: the element size in bytes
    pub fn prepare_layer(&mut self, details: Arc<CopyStreamLayerDetails>) -> Result<()> {
        let mut state = self.state.lock().unwrap();

        let layer_count = details.src_layer_ptrs.len();

        if state.max_num_layers < layer_count {
            return Err(error!(
                "Number of layers {} exceeds max number of layers {}",
                layer_count, state.max_num_layers
            ));
        }

        if state.staged_layers {
            return Err(error!("Layers already loaded"));
        }

        state.staged_layers = true;
        state.layer_details = details;

        assert!(state.layer_doorbells.len() >= layer_count);
        state
            .layer_doorbells
            .iter_mut()
            .for_each(|doorbell| *doorbell = false);

        Ok(())
    }

    /// Prepare the block ids for the copy kernel
    /// See [CopyStreamLayerDetails] for more details
    /// - src_block_ids: the source block ids
    /// - dst_block_ids: the destination block ids
    pub fn prepare_block_ids(&self, src_block_ids: &[i32], dst_block_ids: &[i32]) -> Result<()> {
        if src_block_ids.len() != dst_block_ids.len() {
            return Err(error!(
                "src_block_ids and dst_block_ids must have the same length"
            ));
        }

        // we could disable the unique block id test in production as it adds some overhead
        #[cfg(debug_assertions)]
        {
            // validate that the dst block ids are unique
            let dst_block_ids_set: std::collections::HashSet<_> = dst_block_ids.iter().collect();
            if dst_block_ids_set.len() != dst_block_ids.len() {
                return Err(error!("dst_block_ids must be unique"));
            }

            // validate that the src block ids are unique
            let src_block_ids_set: std::collections::HashSet<_> = src_block_ids.iter().collect();
            if src_block_ids_set.len() != src_block_ids.len() {
                return Err(error!("src_block_ids must be unique"));
            }
        }

        let mut state = self.state.lock().unwrap();

        if state.max_num_blocks < src_block_ids.len() {
            return Err(error!(
                "Number of blocks {} exceeds max number of blocks {}",
                src_block_ids.len(),
                state.max_num_blocks
            ));
        }

        if !state.staged_layers {
            return Err(error!("Layers must be loaded before preparing block ids"));
        }

        if state.staged_block_ids {
            return Err(error!("Block ids already loaded"));
        }

        // we need to copy the block ids to the state so we don't have to block on the async xfer
        // of the lists from host to device
        state.src_block_ids = src_block_ids.to_vec();
        state.dst_block_ids = dst_block_ids.to_vec();

        // transfer the block ids to the device
        // this can be safely done without blocking as the copy stream state is
        let rc = unsafe {
            copy_stream_prepare_block_ids(
                state.c_handle.as_ptr() as *mut std::ffi::c_void,
                state.src_block_ids.as_ptr() as *const std::os::raw::c_int,
                state.dst_block_ids.as_ptr() as *const std::os::raw::c_int,
                state.src_block_ids.len() as i32,
            )
        };

        if rc != 0 {
            return Err(error!("Failed to prepare block ids"));
        }

        state.staged_block_ids = true;

        Ok(())
    }

    pub fn trigger_layer(&self, layer: usize) -> Result<()> {
        let mut state = self.state.lock().unwrap();

        if layer >= state.layer_details.src_layer_ptrs.len() {
            return Err(error!(
                "layer index {} out of bounds (max {})",
                layer,
                state.layer_details.src_layer_ptrs.len() - 1
            ));
        }

        if state.layer_doorbells[layer] {
            tracing::debug!("Layer {} already triggered; this is a no-op", layer);
            return Ok(());
        }

        let cs = state.c_handle.as_ptr() as *mut std::ffi::c_void;
        let src_data = state.layer_details.src_layer_ptrs[layer] as *const std::ffi::c_void;
        let dst_data = state.layer_details.dst_layer_ptrs[layer] as *mut std::ffi::c_void;

        let rc = unsafe {
            copy_stream_launch(
                cs,
                src_data,
                dst_data,
                state.layer_details.prefix_dim,
                state.layer_details.suffix_dim,
                state.layer_details.elem_size,
                state.layer_details.src_block_dim,
                state.layer_details.dst_block_dim,
            )
        };

        if rc != 0 {
            return Err(error!("Failed to execute layer {} copy", layer));
        }

        state.layer_doorbells[layer] = true;

        Ok(())
    }
}

impl Returnable for CopyStream {
    fn on_return(&mut self) -> () {
        let mut state = self.state.lock().unwrap();

        // reset the staged flags
        state.staged_block_ids = false;
        state.staged_layers = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[test]
    fn test_kv_block_storage_kv_first() -> Result<()> {
        let device = CudaContext::new(0)?;

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
            .tp_rank(0)
            .model_details(model_details)
            .build()?;

        // Create the storage blocks
        let mut h_blocks =
            KvBlockStorage::allocate(32, block_details.clone(), StorageType::Pinned)?;

        let mut d_blocks = KvBlockStorage::allocate(
            32,
            block_details.clone(),
            StorageType::Device(device.clone()),
        )?;

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
        let stream = device.new_stream()?;

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
        let device = CudaContext::new(0)?;

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
            .tp_rank(0)
            .model_details(model_details)
            .build()?;

        // Create the storage blocks
        let mut h_blocks =
            KvBlockStorage::allocate(32, block_details.clone(), StorageType::Pinned)?;

        let mut d_blocks = KvBlockStorage::allocate(
            32,
            block_details.clone(),
            StorageType::Device(device.clone()),
        )?;

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

        println!("Copying data to device");

        {
            let blocks = (0..32).collect::<Vec<_>>();

            let h_layer = h_blocks.layer(0).unwrap();
            let mut d_layer = d_blocks.layer_mut(0).unwrap();
            h_layer
                .copy_blocks_to(&blocks, &mut d_layer, &blocks)
                .unwrap();
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
                .copy_blocks_to(&blocks, &mut h_layer, &blocks)
                .unwrap();
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
    #[case(KvLayout::KvFirst)]
    #[case(KvLayout::BlockFirst)]
    #[test]
    fn test_kv_block_storage_layouts(#[case] layout: KvLayout) -> Result<()> {
        let device = CudaContext::new(0)?;

        let layout_name = match layout {
            KvLayout::KvFirst => "KvFirst",
            KvLayout::BlockFirst => "BlockFirst",
        };
        println!("Testing layout: {}", layout_name);

        let number_of_blocks = 8;

        let model_details = KvModelDetailsBuilder::default()
            .number_of_layers(2)
            .number_of_heads(2)
            .head_size(2)
            .dtype(DType::F32)
            .build()?;

        let block_details = KvBlockDetailsBuilder::default()
            .layout(layout)
            .block_size(4)
            .tp_size(1)
            .tp_rank(0)
            .model_details(model_details)
            .build()?;

        // Create the storage blocks
        let mut h_blocks =
            KvBlockStorage::allocate(number_of_blocks, block_details.clone(), StorageType::Pinned)?;

        let mut d_blocks = KvBlockStorage::allocate(
            number_of_blocks,
            block_details.clone(),
            StorageType::Device(device.clone()),
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
                                            nd_view[[
                                                block_idx,
                                                kv_idx,
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
                .copy_blocks_to(&blocks, &mut d_layer, &blocks)
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
                .copy_blocks_to(src_blocks, &mut h_layer, dst_blocks)
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

    #[rstest]
    #[case(KvLayout::KvFirst)]
    #[case(KvLayout::BlockFirst)]
    #[test]
    fn test_kv_block_copy_stream(#[case] layout: KvLayout) -> Result<()> {
        let layout_name = match layout {
            KvLayout::KvFirst => "KvFirst",
            KvLayout::BlockFirst => "BlockFirst",
        };
        println!("Testing layout: {}", layout_name);

        let device = CudaContext::new(0)?;

        let number_of_cpu_blocks = 128;
        let number_of_gpu_blocks = 64;

        let model_details = KvModelDetailsBuilder::default()
            .number_of_layers(2)
            .number_of_heads(2)
            .head_size(2)
            .dtype(DType::F32)
            .build()?;

        let block_details = KvBlockDetailsBuilder::default()
            .layout(layout)
            .block_size(4)
            .tp_size(1)
            .tp_rank(0)
            .model_details(model_details)
            .build()?;

        // Create the storage blocks
        let mut h_blocks = KvBlockStorage::allocate(
            number_of_cpu_blocks,
            block_details.clone(),
            StorageType::Pinned,
        )?;

        let mut d_blocks = KvBlockStorage::allocate(
            number_of_gpu_blocks,
            block_details.clone(),
            StorageType::Device(device),
        )?;

        println!("Allocated pinned and device blocks");
        println!("Letting layer 0 on host to be 1s");

        let layout = h_blocks.layer(0).unwrap().layout.clone();
        let shape = h_blocks.layer(0).unwrap().view().unwrap().shape().clone();

        println!("shape: {:?}", shape);

        Ok(())
    }
}
