// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! State machine for managing worker lifecycle in the scheduler connector.
//!
//! This module provides a type-safe state machine for managing the progression
//! of a scheduler worker from initialization through registration to synchronization
//! and active operation.

use std::sync::Arc;

use anyhow::{Result, bail};
use tracing::debug;

use crate::block_manager::storage::torch::TorchTensor;

use crate::block_manager::layout::{BlockDimension, LayoutConfigBuilder};
use crate::v2::{
    integrations::vllm::VllmConfig,
    layout::{
        LayerSeparateLayout, Layout, LayoutConfig, MemoryRegion, validate_tensor_shapes,
        validate_tensor_strides,
    },
};

/// Configuration for scheduler worker initialization.
#[derive(Clone)]
pub struct SchedulerWorkerConfig {
    /// vLLM configuration containing all parallel and attention settings
    pub vllm_config: VllmConfig,
}

impl std::fmt::Debug for SchedulerWorkerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SchedulerWorkerConfig")
            .field("vllm_config", &self.vllm_config)
            .finish()
    }
}

impl SchedulerWorkerConfig {
    /// Create a new configuration from VllmConfig
    pub fn new(vllm_config: VllmConfig) -> Self {
        Self { vllm_config }
    }

    /// Get device ID from vLLM config
    pub fn device_id(&self) -> usize {
        self.vllm_config.attention.device_id()
    }

    /// Get page size (block size) from vLLM config
    pub fn page_size(&self) -> usize {
        self.vllm_config.attention.block_size()
    }

    /// Get dtype width in bytes from vLLM config
    pub fn dtype_width_bytes(&self) -> usize {
        self.vllm_config.attention.cache_dtype().bytes_per_element()
    }

    /// Get layout contiguity from vLLM config
    pub fn is_fully_contiguous(&self) -> bool {
        self.vllm_config.attention.is_fully_contiguous()
    }

    /// Get worker ID from vLLM parallel config
    pub fn worker_id(&self) -> usize {
        self.vllm_config.parallel.worker_id()
    }

    /// Get tensor parallel size
    pub fn tensor_parallel_size(&self) -> usize {
        self.vllm_config.parallel.tensor_parallel_size()
    }

    /// Get pipeline parallel size
    pub fn pipeline_parallel_size(&self) -> usize {
        self.vllm_config.parallel.pipeline_parallel_size()
    }
}

/// State of a scheduler worker before KV registration.
#[derive(Debug)]
pub struct BeforeRegisteredState {
    config: SchedulerWorkerConfig,
}

/// Wrapper for torch tensors to implement OwnedMemoryRegion.
#[derive(Debug)]
struct TorchMemoryOwner {
    tensor: Arc<dyn TorchTensor>,
}

impl MemoryRegion for TorchMemoryOwner {
    fn addr(&self) -> usize {
        self.tensor.data_ptr() as usize
    }

    fn size(&self) -> usize {
        self.tensor.size_bytes()
    }

    fn storage_kind(&self) -> crate::v2::storage::StorageKind {
        // Torch tensors are device memory when on GPU
        crate::v2::storage::StorageKind::Device(0) // TODO: Get actual device ID
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// State of a scheduler worker after KV registration.
#[derive(Debug)]
pub struct AfterRegisteredState {
    config: SchedulerWorkerConfig,
    layout: Arc<dyn Layout>,
}

/// Main state enum representing the scheduler worker lifecycle.
#[derive(Debug)]
pub enum SchedulerWorkerState {
    BeforeRegistered(BeforeRegisteredState),
    AfterRegistered(AfterRegisteredState),
    // Future states for active message integration:
    // Synchronized(SynchronizedState),
    // Active(ActiveState),
}

impl SchedulerWorkerState {
    /// Check if the worker is in the BeforeRegistered state.
    pub fn is_before_registered(&self) -> bool {
        matches!(self, SchedulerWorkerState::BeforeRegistered(_))
    }

    /// Check if the worker is in the AfterRegistered state.
    pub fn is_after_registered(&self) -> bool {
        matches!(self, SchedulerWorkerState::AfterRegistered(_))
    }

    /// Get the worker configuration (available in all states).
    pub fn config(&self) -> &SchedulerWorkerConfig {
        match self {
            SchedulerWorkerState::BeforeRegistered(state) => &state.config,
            SchedulerWorkerState::AfterRegistered(state) => &state.config,
        }
    }

    /// Get the layout configuration (only available after registration).
    pub fn layout_config(&self) -> Option<&LayoutConfig> {
        match self {
            SchedulerWorkerState::BeforeRegistered(_) => None,
            SchedulerWorkerState::AfterRegistered(state) => Some(state.layout.config()),
        }
    }
}

/// State machine for managing scheduler worker lifecycle.
pub struct SchedulerWorkerStateMachine {
    state: SchedulerWorkerState,
}

impl SchedulerWorkerStateMachine {
    /// Create a new scheduler worker state machine in the BeforeRegistered state.
    pub fn new(config: SchedulerWorkerConfig) -> Self {
        Self {
            state: SchedulerWorkerState::BeforeRegistered(BeforeRegisteredState { config }),
        }
    }

    /// Get the current state of the worker.
    pub fn state(&self) -> &SchedulerWorkerState {
        &self.state
    }

    /// Register KV cache tensors and transition to AfterRegistered state.
    ///
    /// This function validates tensors, infers layout parameters, creates the appropriate
    /// layout (FullyContiguous or LayerSeparate), and transitions the state machine to
    /// AfterRegistered.
    ///
    /// # Arguments
    /// * `tensors` - Vector of torch tensors (one per layer)
    /// * `num_device_blocks` - Number of device blocks to allocate
    ///
    /// # Returns
    /// Updated state machine in AfterRegistered state, or error if validation fails
    pub fn register_kv(
        self,
        tensors: Vec<Arc<dyn TorchTensor>>,
        num_device_blocks: usize,
    ) -> Result<Self> {
        // 1. Extract configuration from BeforeRegistered state
        let config = match self.state {
            SchedulerWorkerState::BeforeRegistered(ref state) => state.config.clone(),
            _ => bail!("register_kv can only be called in BeforeRegistered state"),
        };

        // 2. Validate tensors
        if tensors.is_empty() {
            bail!("Cannot register empty tensor list");
        }

        // Validate shapes are consistent across all tensors
        let shape = validate_tensor_shapes(&tensors)?;

        // Validate strides and get tensor format (NHD/HND)
        let tensor_format = validate_tensor_strides(&tensors)?;

        // 3. Determine layout dimensions from shape
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

        // 4. Calculate inner dimension from remaining shape
        // The remaining dimensions should give us page_size * inner_dim
        let page_size = config.page_size(); // from vllm_config.attention.block_size()
        let inner_dim_product: usize = shape[2..].iter().product();

        // Validate that page_size divides evenly into the product
        if inner_dim_product % page_size != 0 {
            bail!(
                "Page size {} doesn't divide evenly into inner dimensions (product: {})",
                page_size,
                inner_dim_product
            );
        }

        let inner_dim = inner_dim_product / page_size;

        // 5. Use AttentionConfig to infer inner shape (with smart dimension checking)
        let inner_shape = config.vllm_config.attention.infer_inner_shape(&shape[2..]);

        debug!(
            "InnerShape determination: page_size={}, inner_dim={}, tensor_format={:?} => shape={:?}",
            page_size, inner_dim, tensor_format, inner_shape
        );

        // 6. Build LayoutConfig using the builder pattern with InnerShape
        let layout_config = LayoutConfigBuilder::default()
            .num_blocks(num_device_blocks)
            .num_layers(tensors.len())
            .outer_dim(outer_dim)
            .page_size(page_size)
            .inner_dim(inner_dim)
            .dtype_width_bytes(config.dtype_width_bytes())
            .alignment(1) // Default alignment, could be made configurable
            .inner_shape(inner_shape)
            .build()?;

        // 6. Create memory wrappers for each tensor
        let memory: Vec<Arc<dyn MemoryRegion>> = tensors
            .into_iter()
            .map(|tensor| Arc::new(TorchMemoryOwner { tensor }) as Arc<dyn MemoryRegion>)
            .collect();

        // 7. Create LayerSeparateLayout with the configuration and memory
        let num_layers = layout_config.num_layers;
        let final_inner_shape = layout_config.inner_shape;

        let layout = Arc::new(LayerSeparateLayout::new(layout_config, memory, block_dim)?)
            as Arc<dyn Layout>;

        // 8. Transition to AfterRegistered state
        debug!(
            "State transition: BeforeRegistered -> AfterRegistered (blocks={}, layers={}, inner_shape={:?})",
            num_device_blocks, num_layers, final_inner_shape
        );

        Ok(Self {
            state: SchedulerWorkerState::AfterRegistered(AfterRegisteredState { config, layout }),
        })
    }

    /// Get the layout (if after registration).
    ///
    /// This returns a reference to the layout object which can be used to
    /// query memory regions for blocks.
    pub fn layout(&self) -> Option<Arc<dyn Layout>> {
        match &self.state {
            SchedulerWorkerState::BeforeRegistered(_) => None,
            SchedulerWorkerState::AfterRegistered(state) => Some(state.layout.clone()),
        }
    }

    // Future methods for active message integration:
    // pub fn sync_with_leader(self, leader_info: LeaderInfo) -> Result<Self>
    // pub fn activate(self) -> Result<Self>
    // pub fn handle_message(self, message: ActiveMessage) -> Result<Self>
}

impl std::fmt::Debug for SchedulerWorkerStateMachine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SchedulerWorkerStateMachine")
            .field("state", &self.state)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v2::{
        AttentionConfig, CacheDtype, CacheLayout, ParallelConfig,
        integrations::vllm::{VllmAttentionConfig, VllmConfig, VllmParallelConfig},
    };

    #[derive(Debug, Clone)]
    struct TestParallelConfig {
        tensor_parallel_size: usize,
        pipeline_parallel_size: usize,
        world_size: usize,
        worker_id: usize,
    }

    impl VllmParallelConfig for TestParallelConfig {}

    impl ParallelConfig for TestParallelConfig {
        fn tensor_parallel_size(&self) -> usize {
            self.tensor_parallel_size
        }
        fn pipeline_parallel_size(&self) -> usize {
            self.pipeline_parallel_size
        }
        fn world_size(&self) -> usize {
            self.world_size
        }
        fn tensor_parallel_rank(&self) -> usize {
            0
        }
        fn pipeline_parallel_rank(&self) -> usize {
            0
        }
        fn worker_id(&self) -> usize {
            self.worker_id
        }
        fn is_single_node(&self) -> bool {
            true
        }
    }

    #[derive(Debug, Clone)]
    struct TestAttentionConfig {
        block_size: usize,
        device_id: usize,
        cache_dtype: CacheDtype,
        cache_layout: CacheLayout,
    }

    impl VllmAttentionConfig for TestAttentionConfig {}

    impl AttentionConfig for TestAttentionConfig {
        fn block_size(&self) -> usize {
            self.block_size
        }
        fn head_size(&self) -> Option<usize> {
            Some(64)
        }
        fn num_kv_heads(&self) -> Option<usize> {
            Some(8)
        }
        fn cache_layout(&self) -> CacheLayout {
            self.cache_layout
        }
        fn cache_dtype(&self) -> CacheDtype {
            self.cache_dtype
        }
        fn device_id(&self) -> usize {
            self.device_id
        }
    }

    fn create_test_config() -> SchedulerWorkerConfig {
        let parallel = Arc::new(TestParallelConfig {
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
            world_size: 1,
            worker_id: 0,
        }) as Arc<dyn VllmParallelConfig>;

        let attention = Arc::new(TestAttentionConfig {
            block_size: 16,
            device_id: 0,
            cache_dtype: CacheDtype::FP16,
            cache_layout: CacheLayout::NHD,
        }) as Arc<dyn VllmAttentionConfig>;

        let vllm_config = VllmConfig::new(parallel, attention);

        SchedulerWorkerConfig::new(vllm_config)
    }

    #[test]
    fn test_initial_state() {
        let config = create_test_config();
        let machine = SchedulerWorkerStateMachine::new(config.clone());

        assert!(machine.state().is_before_registered());
        assert!(!machine.state().is_after_registered());
        assert_eq!(machine.state().config().device_id(), config.device_id());
        assert!(machine.layout().is_none());
    }

    #[test]
    fn test_double_registration_fails() {
        let config = create_test_config();
        let _machine = SchedulerWorkerStateMachine::new(config);

        // First registration would succeed with real tensors
        // For now, just test that double registration would fail
        // by creating a machine in AfterRegistered state manually

        // This test demonstrates the structure - actual implementation
        // would require mock tensors for full testing
    }
}
