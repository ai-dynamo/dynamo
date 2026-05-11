// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
//! 2. Worker exports Velo peer address as handshake metadata (no NIXL yet)
//! 3. Leader collects handshake metadata and coordinates layout creation
//! 4. Leader calls `configure_layouts` RPC on each worker
//! 5. Worker completes NIXL registration and creates DirectWorker
//! 6. Worker creates G2/G3 layouts based on leader config
//! 7. Worker returns updated metadata with all layouts

use std::{path::PathBuf, sync::Arc};

use anyhow::Result;
use derive_builder::Builder;
use dynamo_memory::TensorDescriptor;

use kvbm_physical::transfer::context::TokioRuntime;

use kvbm_engine::object::create_object_client;
use kvbm_engine::worker::{DirectWorker, LeaderLayoutConfig, WorkerLayoutResponse};

use crate::KvbmRuntime;
use kvbm_common::LogicalLayoutHandle;
use kvbm_physical::TransferManager;
use kvbm_physical::layout::{BlockDimension, LayoutConfig, PhysicalLayoutBuilder};
use kvbm_physical::transfer::TransferCapabilities;

/// Cached state from `register_kv_caches` for deferred initialization.
///
/// This struct holds all the information needed to complete NIXL registration
/// once the leader triggers initialization via the `configure_layouts` handler.
#[derive(Debug, Builder)]
#[builder(pattern = "owned", build_fn(private, name = "build_internal"))]
pub struct PendingWorkerState {
    /// CUDA device ID where tensors are allocated.
    pub cuda_device_id: usize,

    /// KV cache tensors, one per layer, all on the same CUDA device.
    pub tensors: Vec<Arc<dyn TensorDescriptor>>,

    /// Number of device blocks from vLLM's cache config.
    pub num_device_blocks: usize,

    /// Block/page size for the KV cache.
    #[expect(dead_code)]
    pub page_size: usize,

    /// Data type width in bytes (e.g., 2 for fp16).
    #[expect(dead_code)]
    pub dtype_width_bytes: usize,

    /// Layout configuration determined from tensor shapes.
    pub layout_config: LayoutConfig,

    /// Block dimension (first or second dimension).
    pub block_dim: BlockDimension,
}

impl PendingWorkerStateBuilder {
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
    pub fn build(mut self) -> Result<PendingWorkerState> {
        use anyhow::{bail, ensure};
        use dynamo_memory::TensorDescriptorExt;

        // Validate tensors first (before build_internal which requires cuda_device_id)
        let tensors = self
            .tensors
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("`tensors` must be initialized"))?;

        if tensors.is_empty() {
            bail!("no tensors to register");
        }

        // Extract and validate CUDA device ID from tensors
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

        // Set cuda_device_id on builder before calling build_internal
        self.cuda_device_id = Some(cuda_device_id);

        self.build_internal()
            .map_err(|e| anyhow::anyhow!("failed to build PendingWorkerState: {}", e))
    }
}

impl PendingWorkerState {
    /// Create a new PendingWorkerState builder.
    pub fn builder() -> PendingWorkerStateBuilder {
        PendingWorkerStateBuilder::default()
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
    /// * `runtime` - KvbmRuntime providing event system and tokio handle
    /// * `config` - Leader-provided layout configuration
    ///
    /// # Returns
    /// Tuple of (DirectWorker, WorkerLayoutResponse)
    #[tracing::instrument(level = "debug", skip_all, fields(instance_id = ?runtime.messenger().instance_id()))]
    pub fn complete_initialization(
        self,
        runtime: &KvbmRuntime,
        config: LeaderLayoutConfig,
    ) -> Result<(Arc<DirectWorker>, WorkerLayoutResponse)> {
        tracing::info!("Starting complete_initialization");

        let mut created_layouts = vec![];

        let nixl_agent = runtime
            .nixl_agent()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("NIXL agent not found"))?;

        // 1. Build TransferManager and NixlAgent.
        //
        // When the cache config requests host bypass (DYN_KVBM_DISK_CACHE_GB
        // set, DYN_KVBM_CPU_CACHE_GB unset), enable GDS so the strategy layer
        // selects direct G1↔G3 transfers instead of trying to stage through a
        // G2 tier that doesn't exist. `with_gds_if_supported()` probes the
        // host once and falls back to allow_gds=false if the probe fails — in
        // that case the first G1↔G3 transfer will error loudly, which is the
        // correct signal that bypass mode isn't viable on this host.
        let bypass_host = runtime.config().cache.bypass_host_cache();
        let capabilities = if bypass_host {
            TransferCapabilities::default().with_gds_if_supported()
        } else {
            TransferCapabilities::default()
        };
        tracing::info!(
            bypass_host,
            allow_gds = capabilities.allow_gds,
            "Building TransferManager with NIXL backend"
        );
        let transfer_manager = TransferManager::builder()
            .event_system(runtime.event_system())
            .cuda_device_id(self.cuda_device_id)
            .tokio_runtime(TokioRuntime::Handle(runtime.tokio()))
            .observability(runtime.observability().clone())
            .capabilities(capabilities)
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

        // 5. Build optional object client with rank-based key formatting
        // Uses object config from leader to ensure all workers have consistent settings
        // Note: We use block_in_place because this sync function may be called from
        // within a tokio async context (e.g., RPC handler)
        let object_client = if let Some(object_config) = &config.object {
            let client = tokio::task::block_in_place(|| {
                runtime
                    .handle()
                    .block_on(create_object_client(object_config, Some(config.rank)))
            })?;
            tracing::info!(
                rank = config.rank,
                "Object storage client configured from leader config"
            );
            Some(client)
        } else {
            None
        };

        // 6. Create G2/G3 layouts based on leader config and parallelism mode
        //
        // For ReplicatedData mode: only rank 0 gets G2/G3 layouts
        // For TensorParallel mode: all workers get G2/G3 layouts
        // For host-bypass mode (DYN_KVBM_DISK_CACHE_GB set, DYN_KVBM_CPU_CACHE_GB
        // unset): G2 is skipped on every rank — transfers go G1↔G3 directly via
        // GDS. G3 still gets allocated normally.
        let skip_g2_g3 =
            config.parallelism == kvbm_config::ParallelismMode::ReplicatedData && config.rank > 0;
        let bypass_host = runtime.config().cache.bypass_host_cache();

        let (g2_handle, g3_handle) = if skip_g2_g3 {
            tracing::info!(
                rank = config.rank,
                parallelism = ?config.parallelism,
                "Skipping G2/G3 layout creation (ReplicatedData mode, rank > 0)"
            );
            (None, None)
        } else {
            tracing::info!(
                host_block_count = config.host_block_count,
                disk_block_count = ?config.disk_block_count,
                parallelism = ?config.parallelism,
                bypass_host,
                "Creating G2/G3 layouts via configure_additional_layouts()"
            );

            let g2_handle = if bypass_host {
                tracing::info!(
                    "Skipping G2 layout allocation (host-bypass mode: G1↔G3 direct via GDS)"
                );
                None
            } else {
                let mut host_layout = self.layout_config.clone();
                host_layout.num_blocks = config.host_block_count;

                let total_bytes = host_layout.required_bytes() as u64;
                tracing::info!(
                    host_block_count = config.host_block_count,
                    bytes_per_block = host_layout.bytes_per_block(),
                    total_gb = total_bytes / (1024 * 1024 * 1024),
                    "Allocating pinned host memory for G2 layout"
                );

                let host_layout = PhysicalLayoutBuilder::new(nixl_agent.clone())
                    .with_config(host_layout)
                    .fully_contiguous()
                    .allocate_pinned(Some(self.cuda_device_id as u32))
                    .build()
                    .map_err(|e| {
                        tracing::error!(
                            host_block_count = config.host_block_count,
                            total_gb = total_bytes / (1024 * 1024 * 1024),
                            error = %e,
                            "Failed to allocate pinned host memory for G2 layout"
                        );
                        e
                    })?;

                let handle = transfer_manager.register_layout(host_layout)?;
                created_layouts.push(LogicalLayoutHandle::G2);
                Some(handle)
            };

            // todo: we need to get a path from the the config and create a unique file based on the velo instance_id
            let g3_handle = if let Some(disk_blocks) = config.disk_block_count {
                let mut disk_layout = self.layout_config.clone();
                disk_layout.num_blocks = disk_blocks;

                let disk_total_bytes = disk_layout.required_bytes() as u64;
                tracing::info!(
                    disk_block_count = disk_blocks,
                    bytes_per_block = disk_layout.bytes_per_block(),
                    total_gb = disk_total_bytes / (1024 * 1024 * 1024),
                    "Allocating disk-backed memory for G3 layout"
                );

                let g3_path = PathBuf::from(format!(
                    "/tmp/kvbm_g3_{}.bin",
                    runtime.messenger().instance_id()
                ));

                // Register the path for unlink-on-signal before allocation, so that
                // if `fallocate` is interrupted by SIGINT/SIGTERM after `open(O_CREAT)`
                // has already created the file, the cleanup task still removes it.
                // Clean shutdowns continue to be handled by `DiskStorage`'s Drop impl.
                crate::connector::disk_cleanup::register(g3_path.clone());

                let disk_layout = PhysicalLayoutBuilder::new(nixl_agent.clone())
                    .with_config(disk_layout)
                    .fully_contiguous()
                    .allocate_disk(Some(g3_path.clone()))
                    .build()
                    .map_err(|e| {
                        tracing::error!(
                            disk_block_count = disk_blocks,
                            total_gb = disk_total_bytes / (1024 * 1024 * 1024),
                            error = %e,
                            "Failed to allocate disk-backed memory for G3 layout"
                        );
                        e
                    })?;

                let handle = transfer_manager.register_layout(disk_layout)?;
                created_layouts.push(LogicalLayoutHandle::G3);

                // Proactive unlink: remove the directory entry now that NIXL has
                // registered the file. The `DiskStorage` fd inside the registered
                // layout keeps the inode alive — POSIX/UCX continue using the fd —
                // but the kernel reclaims the disk space on *any* process exit
                // (Ctrl+C → vLLM IPC shutdown, SIGKILL, panic-abort, segfault).
                // This is the primary cleanup path; `Drop` and the signal task
                // are belt-and-suspenders for environments where this race
                // (pre-registration crash) leaves a partial file behind.
                match std::fs::remove_file(&g3_path) {
                    Ok(()) => {
                        crate::connector::disk_cleanup::deregister(&g3_path);
                        tracing::info!(
                            path = %g3_path.display(),
                            "G3 cache file unlinked from filesystem (held by fd until process exit)"
                        );
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => tracing::warn!(
                        path = %g3_path.display(),
                        error = %e,
                        "failed to unlink G3 cache file after NIXL registration"
                    ),
                }

                Some(handle)
            } else {
                None
            };

            (g2_handle, g3_handle)
        };

        // 7. Build DirectWorker with all handles via builder pattern
        let mut builder = DirectWorker::builder()
            .manager(transfer_manager.clone())
            .g1_handle(g1_handle)
            .rank(config.rank);

        // Optional G2 handle (not present for ReplicatedData rank > 0)
        if let Some(g2) = g2_handle {
            builder = builder.g2_handle(g2);
        }

        // Optional G3 handle
        if let Some(g3) = g3_handle {
            builder = builder.g3_handle(g3);
        }

        // optional g4/object client
        if let Some(client) = object_client {
            builder = builder.object_client(client);
        }

        let direct_worker = Arc::new(builder.build()?);

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
