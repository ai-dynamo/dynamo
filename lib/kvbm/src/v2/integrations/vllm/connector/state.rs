// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! State machine for managing worker lifecycle in the scheduler connector.
//!
//! This module provides a type-safe state machine for managing the progression
//! of a scheduler worker from initialization through registration to synchronization
//! and active operation.

use std::sync::Arc;

use anyhow::{Result, bail};
use dynamo_nova::Nova;
use tracing::debug;

use dynamo_memory::{
    Buffer, TensorDescriptor,
    nixl::{NixlAgent, NixlRegisterExt},
};

use crate::{
    physical::layout::InnerShape,
    v2::{
        integrations::{AttentionConfig, vllm::KvbmVllmConfig},
        physical::layout::{
            BlockDimension, LayerSeparateLayout, Layout, LayoutConfig, validate_tensor_shapes,
            validate_tensor_strides,
        },
    },
};

/// Configuration for scheduler worker initialization.
#[derive(Clone)]
pub struct SchedulerWorkerConfig {
    vllm_config: KvbmVllmConfig,
}

impl std::fmt::Debug for SchedulerWorkerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SchedulerWorkerConfig")
            .field("vllm_config", &self.vllm_config)
            .finish()
    }
}

impl SchedulerWorkerConfig {
    // /// Create a new configuration from KvbmVllmConfig
    // pub fn new(vllm_config: KvbmVllmConfig) -> Self {
    //     Self { vllm_config }
    // }

    /// Get page size (block size) from vLLM config
    pub fn page_size(&self) -> usize {
        self.vllm_config.attention.block_size()
    }

    pub fn dtype_width_bytes(&self) -> usize {
        2
    }
}

/// Main state enum representing the scheduler worker lifecycle.
#[derive(Clone)]
pub struct SchedulerWorker {
    config: KvbmVllmConfig,
    nixl: NixlAgent,
    nova: Arc<Nova>,
}

impl SchedulerWorker {
    /// Register KV cache tensors and transition to AfterRegistered state.
    ///
    /// This function validates tensors, infers layout parameters, creates the appropriate
    /// layout (FullyContiguous or LayerSeparate).
    ///
    /// # Arguments
    /// * `tensors` - Vector of torch tensors (one per layer)
    /// * `num_device_blocks` - Number of device blocks to allocate
    ///
    pub fn register_kv(
        self,
        tensors: Vec<Arc<dyn TensorDescriptor>>,
        num_device_blocks: usize,
    ) -> Result<()> {
        // Validate tensors
        if tensors.is_empty() {
            bail!("Cannot register empty tensor list");
        }

        // Validate shapes are consistent across all tensors
        let shape = validate_tensor_shapes(&tensors)?;

        // Validate strides and get tensor format (NHD/HND)
        let tensor_format = validate_tensor_strides(&tensors)?;

        // Determine layout dimensions from shape
        // Shape is expected to be [dim0, dim1, page_tokens, hidden_dim] or similar
        // We need to figure out which dimension contains the blocks
        let (block_dim, outer_dim) = if shape[0] >= num_device_blocks {
            // First dimension contains blocks [num_blocks, outer_dim, ...]
            debug!(
                "Block dimension: BlockIsFirstDim (shape[{}]={} >= num_blocks={})",
                0, shape[0], num_device_blocks
            );
            (BlockDimension::BlockIsFirstDim, shape[1])
        } else if shape[1] >= num_device_blocks {
            // Second dimension contains blocks [outer_dim, num_blocks, ...]
            debug!(
                "Block dimension: BlockIsSecondDim (shape[{}]={} >= num_blocks={})",
                1, shape[1], num_device_blocks
            );
            (BlockDimension::BlockIsSecondDim, shape[0])
        } else {
            bail!(
                "Unexpected tensor shape {:?}: num_device_blocks {} not found in first two dimensions",
                shape,
                num_device_blocks
            );
        };

        // Calculate inner dimension from remaining shape
        // The remaining dimensions should give us page_size * inner_dim
        let page_size = self.config.attention.block_size(); // from vllm_config.attention.block_size()
        let inner_dim_product: usize = shape[2..].iter().product();

        // Validate that page_size divides evenly into the product
        if !inner_dim_product.is_multiple_of(page_size) {
            bail!(
                "Page size {} doesn't divide evenly into inner dimensions (product: {})",
                page_size,
                inner_dim_product
            );
        }

        let inner_dim = inner_dim_product / page_size;

        // // 5. Use AttentionConfig to infer inner shape (with smart dimension checking)
        // let inner_shape = config.vllm_config.attention.infer_inner_shape(&shape[2..]);

        // debug!(
        //     "InnerShape determination: page_size={}, inner_dim={}, tensor_format={:?} => shape={:?}",
        //     page_size, inner_dim, tensor_format, inner_shape
        // );

        // 6. Build LayoutConfig using the builder pattern with InnerShape
        let layout_config = LayoutConfig::builder()
            .num_blocks(num_device_blocks)
            .num_layers(tensors.len())
            .outer_dim(outer_dim)
            .page_size(page_size)
            .inner_dim(inner_dim)
            .dtype_width_bytes(self.config.attention.cache_dtype_bytes())
            .alignment(1) // Default alignment, could be made configurable
            .inner_shape(InnerShape::Unknown)
            .build()?;

        // 6. Create memory wrappers for each tensor
        let memory: Vec<Buffer> = tensors
            .into_iter()
            .map(|tensor| {
                let registered = tensor
                    .register(&self.nixl, None)
                    .map_err(|_| anyhow::anyhow!("NIXL registration failed"))?;
                Ok(Buffer::new(registered))
            })
            .collect::<Result<Vec<_>>>()?;

        // 7. Create LayerSeparateLayout with the configuration and memory
        let num_layers = layout_config.num_layers;
        let final_inner_shape = layout_config.inner_shape;

        let layout = Arc::new(LayerSeparateLayout::new(layout_config, memory, block_dim)?)
            as Arc<dyn Layout>;

        // todo: implement the following
        // create a transfer context specific to the gpu device id
        // create a transfer manager from the context
        // create a direct worker from the transfer manager
        // register the layout with the direct worker as the G1 layout

        // TODO: Store layout and return it
        let _layout = layout;
        Ok(())
    }

    // TODO: Implement layout() method after state machine is complete
    // /// Get the layout (if after registration).
    // pub fn layout(&self) -> Option<Arc<dyn Layout>> { ... }

    // Future methods for active message integration:
    // pub fn sync_with_leader(self, leader_info: LeaderInfo) -> Result<Self>
    // pub fn activate(self) -> Result<Self>
    // pub fn handle_message(self, message: ActiveMessage) -> Result<Self>
}

// TODO: Re-enable tests when SchedulerWorkerStateMachine is implemented
// #[cfg(test)]
// mod tests { ... }
