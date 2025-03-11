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

use derive_builder::Builder;
use dynemo_runtime::{error, raise, ErrorContext, Result};
use std::sync::Arc;
use validator::{Validate, ValidationError};

use super::storage::{DType, OwnedStorage, Storage, StorageType, TensorView};

#[derive(Debug, Clone)]
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
            let storage = OwnedStorage::create(bytes, storage_type.clone()).with_context(|| {
                error!("Failed to allocate memory for KV BlockStorage for layer {layer}")
            })?;
            let layer = KvLayer::from_storage(&block_details, number_of_blocks, storage);
            layers.push(layer);
        }

        Ok(Self::new(layers).context("Validating KvBlockStorage")?)
    }

    pub fn layer(&self, layer: usize) -> Result<&KvLayer> {
        let layer = self
            .layers
            .get(layer)
            .ok_or(error!("Layer {layer} not found"))?;

        Ok(layer)
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
    pub fn view(&self) -> Result<TensorView<'_, Self, 4>> {
        // Check if storage type is compatible with creating a view
        match self.storage.storage_type() {
            StorageType::Device(_) => {
                raise!("Cannot view device storage as a tensor. Use copy_to_host first.");
            }
            StorageType::Pinned | StorageType::System => {}
        }

        // Calculate dimensions based on layout
        let dims = match self.layout {
            KvLayout::KvFirst => [
                2, // K and V as first dimension
                self.number_of_blocks,
                self.number_of_heads / self.tp_size,
                self.head_size,
            ],
            KvLayout::BlockFirst => [
                self.number_of_blocks,
                2, // K and V as second dimension
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

        let device_blocks =
            KvBlockStorage::allocate(16, block_details.clone(), StorageType::Device(0))?;
        let pinned_blocks =
            KvBlockStorage::allocate(32, block_details.clone(), StorageType::Pinned)?;

        let view = pinned_blocks.layer(0)?.view()?;
        let ndarray_view = view.as_ndarray_view::<f32>()?;

        println!("pinned_blocks_view: {:?}", ndarray_view.shape());

        // // set the view values to 1.0
        // pinned_blocks_view.assign(&ndarray::Array::from_shape_fn((16, 2, 4, 8), |_| 1.0));

        Ok(())
    }
}
