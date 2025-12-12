// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! PendingWorkerState - Cached tensor info for deferred NIXL registration.
//!
//! During distributed initialization, workers call `register_kv_caches` before
//! the leader has finished coordination. This module provides state caching
//! so that NIXL registration can be deferred until the leader triggers it
//! via the `configure_layouts` handler.
//!
//! # Initialization Flow
//!
//! 1. Worker calls `register_kv_caches` → tensors cached in `PendingWorkerState`
//! 2. Worker exports Nova peer address as handshake metadata (no NIXL yet)
//! 3. Leader collects handshake metadata and coordinates layout creation
//! 4. Leader calls `configure_layouts` RPC on each worker
//! 5. Worker completes NIXL registration and creates DirectWorker
//! 6. Worker creates G2/G3 layouts based on leader config
//! 7. Worker returns updated metadata with all layouts

use std::{path::PathBuf, sync::Arc};

use anyhow::Result;
use dynamo_memory::TensorDescriptor;

use crate::{
    KvbmRuntime,
    logical::LogicalLayoutHandle,
    physical::transfer::context::TokioRuntime,
    v2::{
        distributed::worker::{DirectWorker, LeaderLayoutConfig, WorkerLayoutResponse},
        physical::{
            TransferManager,
            layout::{BlockDimension, LayoutConfig, PhysicalLayoutBuilder},
        },
    },
};

use super::GpuInfo;

/// Cached state from `register_kv_caches` for deferred initialization.
///
/// This struct holds all the information needed to complete NIXL registration
/// once the leader triggers initialization via the `configure_layouts` handler.
#[derive(Debug)]
pub struct PendingWorkerState {
    /// CUDA device ID where tensors are allocated.
    pub cuda_device_id: usize,

    /// KV cache tensors, one per layer, all on the same CUDA device.
    pub tensors: Vec<Arc<dyn TensorDescriptor>>,

    /// Number of device blocks from vLLM's cache config.
    pub num_device_blocks: usize,

    /// Block/page size for the KV cache.
    pub page_size: usize,

    /// Data type width in bytes (e.g., 2 for fp16).
    pub dtype_width_bytes: usize,

    /// GPU info for logging.
    pub gpu_info: GpuInfo,

    /// Layout configuration determined from tensor shapes.
    pub layout_config: LayoutConfig,

    /// Block dimension (first or second dimension).
    pub block_dim: BlockDimension,
}

impl PendingWorkerState {
    /// Create a new PendingWorkerState from register_kv_caches arguments.
    ///
    /// Validates that all tensors are on the same CUDA device.
    ///
    /// # Arguments
    /// * `tensors` - KV cache tensors, one per layer
    /// * `num_device_blocks` - Number of device blocks from vLLM's cache config
    /// * `page_size` - Block/page size for the KV cache
    /// * `dtype_width_bytes` - Data type width in bytes
    /// * `layout_config` - Layout configuration determined from tensor shapes
    /// * `block_dim` - Block dimension (first or second dimension)
    ///
    /// # Errors
    /// - If tensors is empty
    /// - If tensors are on different CUDA devices
    /// - If first tensor is not on a CUDA device
    pub fn new(
        tensors: Vec<Arc<dyn TensorDescriptor>>,
        num_device_blocks: usize,
        page_size: usize,
        dtype_width_bytes: usize,
        layout_config: LayoutConfig,
        block_dim: BlockDimension,
    ) -> Result<Self> {
        use anyhow::{bail, ensure};
        use dynamo_memory::TensorDescriptorExt;

        if tensors.is_empty() {
            bail!("no tensors to register");
        }

        // Validate and extract CUDA device ID
        let cuda_device_id = tensors[0]
            .cuda_device_id()
            .ok_or_else(|| anyhow::anyhow!("first tensor not on CUDA device"))?;

        for (i, tensor) in tensors[1..].iter().enumerate() {
            ensure!(
                tensor.cuda_device_id() == Some(cuda_device_id),
                "tensor {} on different CUDA device than tensor 0",
                i + 1
            );
        }

        let gpu_info = GpuInfo::from_device_index(cuda_device_id);

        tracing::debug!(
            cuda_device = cuda_device_id,
            gpu_uuid = ?gpu_info.uuid,
            num_tensors = tensors.len(),
            num_device_blocks,
            page_size,
            dtype_width_bytes,
            ?layout_config,
            ?block_dim,
            "Created PendingWorkerState - NIXL registration deferred"
        );

        Ok(Self {
            cuda_device_id,
            tensors,
            num_device_blocks,
            page_size,
            dtype_width_bytes,
            gpu_info,
            layout_config,
            block_dim,
        })
    }

    /// Complete NIXL registration and create DirectWorker.
    ///
    /// This method is called when the leader triggers initialization via
    /// the `configure_layouts` handler. It:
    /// 1. Builds the TransferManager
    /// 2. Determines layout from tensor shapes
    /// 3. Builds PhysicalLayout with NIXL registration
    /// 4. Creates DirectWorker with G1 handle
    /// 5. Creates G2/G3 layouts based on leader config
    ///
    /// # Arguments
    /// * `nova` - Nova instance for event system and runtime
    /// * `config` - Leader-provided layout configuration
    ///
    /// # Returns
    /// Tuple of (DirectWorker, WorkerLayoutResponse)
    #[tracing::instrument(level = "debug", skip_all, fields(instance_id = ?runtime.nova.instance_id()))]
    pub fn complete_initialization(
        self,
        runtime: &KvbmRuntime,
        config: LeaderLayoutConfig,
    ) -> Result<(Arc<DirectWorker>, WorkerLayoutResponse)> {
        tracing::info!("Starting complete_initialization");

        let mut created_layouts = vec![];

        let nixl_agent = runtime
            .nixl_agent
            .clone()
            .ok_or_else(|| anyhow::anyhow!("NIXL agent not found"))?;

        // 1. Build TransferManager and NixlAgent
        tracing::info!("Building TransferManager with NIXL backend");
        let transfer_manager = TransferManager::builder()
            .event_system(runtime.nova.events().local().clone())
            .cuda_device_id(self.cuda_device_id)
            .tokio_runtime(TokioRuntime::Handle(runtime.nova.runtime().clone()))
            .nixl_agent(nixl_agent.clone())
            .build()?;

        // 2. Use pre-computed layout configuration
        tracing::debug!(
            ?self.layout_config,
            ?self.block_dim,
            "Using pre-computed KV layout configuration"
        );

        // 3. Build PhysicalLayout with NIXL registration
        let physical_layout = PhysicalLayoutBuilder::new(nixl_agent.clone())
            .with_config(self.layout_config.clone())
            .layer_separate(self.block_dim)
            .with_external_device_regions(self.tensors)?
            .build()?;

        tracing::debug!("Built physical layout with NIXL-registered memory");
        created_layouts.push(LogicalLayoutHandle::G1);

        // 4. Register layout with TransferManager → get G1 handle
        tracing::info!(
            num_blocks = self.num_device_blocks,
            "Registering G1 (device) layout - external tensors from vLLM"
        );
        let g1_handle = transfer_manager.register_layout(physical_layout)?;
        tracing::info!(?g1_handle, "G1 (device) layout registered successfully");

        // 5. Build DirectWorker and set G1 handle
        let direct_worker = Arc::new(DirectWorker::new(transfer_manager.clone()));
        direct_worker.set_g1_handle(g1_handle)?;

        // 6. Create G2/G3 layouts based on leader config (G2 is REQUIRED)
        tracing::info!(
            host_block_count = config.host_block_count,
            disk_block_count = ?config.disk_block_count,
            "Creating G2/G3 layouts via configure_additional_layouts()"
        );

        let mut host_layout = self.layout_config.clone();
        host_layout.num_blocks = config.host_block_count;
        let host_layout = PhysicalLayoutBuilder::new(nixl_agent.clone())
            .with_config(host_layout)
            .fully_contiguous()
            .allocate_pinned(Some(self.cuda_device_id as u32))
            .build()?;

        let g2_handle = transfer_manager.register_layout(host_layout)?;
        direct_worker.set_g2_handle(g2_handle)?;
        created_layouts.push(LogicalLayoutHandle::G2);

        // todo: we need to get a path from the the config and create a unique file based on the nova instance_id
        if let Some(disk_blocks) = config.disk_block_count {
            let mut disk_layout = self.layout_config.clone();
            disk_layout.num_blocks = disk_blocks;
            let disk_layout = PhysicalLayoutBuilder::new(nixl_agent.clone())
                .with_config(disk_layout)
                .fully_contiguous()
                .allocate_disk(Some(PathBuf::from(format!(
                    "/tmp/kvbm_g3_{}.bin",
                    runtime.nova.instance_id()
                ))))
                .build()?;

            let g3_handle = transfer_manager.register_layout(disk_layout)?;
            direct_worker.set_g3_handle(g3_handle)?;
            created_layouts.push(LogicalLayoutHandle::G3);
        }

        let response = WorkerLayoutResponse {
            metadata: direct_worker.export_metadata()?,
            created_layouts,
        };

        tracing::debug!("complete_initialization finished");

        Ok((direct_worker, response))
    }
}

#[cfg(test)]
mod tests {
    // Tests would require mock tensor setup - defer to integration tests
}
