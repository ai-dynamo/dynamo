// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

use crate::physical::{
    layout::{LayoutConfig, PhysicalLayout},
    manager::{SerializedLayout, TransferManager},
    transfer::{BounceBuffer, TransferOptions, context::TransferCompleteNotification},
};

use super::*;

/// DirectWorker executes transfer operations using a local TransferManager.
///
/// # Execution State vs Coordination State
///
/// DirectWorker maintains **execution state** - the handles and manager needed to
/// actually perform RDMA/local transfers. This is distinct from **coordination state**
/// which the leader tracks in [`CoordinatedWorker`].
///
/// When a leader wraps a DirectWorker in a CoordinatedWorker:
/// - DirectWorker: owns handles for TransferManager execution
/// - CoordinatedWorker: tracks the same handles for leader coordination
///
/// This duplication is intentional - DirectWorker needs handles to execute,
/// and CoordinatedWorker provides a uniform API regardless of whether the
/// inner worker is local (DirectWorker) or remote (NovaWorkerClient).
///
/// # Usage
///
/// DirectWorker is typically:
/// 1. Created during deferred initialization (see [`PendingWorkerState`])
/// 2. Wrapped by [`NovaWorkerService`] to expose RPC handlers
/// 3. Wrapped by [`CoordinatedWorker`] for leader coordination
///
/// [`CoordinatedWorker`]: super::CoordinatedWorker
/// [`PendingWorkerState`]: crate::v2::integrations::connector::worker::PendingWorkerState
/// [`NovaWorkerService`]: super::NovaWorkerService
pub struct DirectWorker {
    // =========================================================================
    // Execution State - needed by TransferManager to perform operations
    // =========================================================================
    /// G1 (GPU KV cache) layout handle - set during Phase 2 registration.
    /// Required for GPU-to-GPU and GPU-to-Host transfers.
    g1_handle: OnceLock<LayoutHandle>,

    /// G2 (Host/pinned cache) layout handle - set during Phase 3 coordination.
    /// Required for Host-to-GPU and Host-to-Disk transfers.
    g2_handle: OnceLock<LayoutHandle>,

    /// G3 (Disk cache) layout handle - set during Phase 3 coordination if disk tier enabled.
    /// Required for Disk-to-Host transfers.
    g3_handle: OnceLock<LayoutHandle>,

    /// The transfer manager that executes actual data movement.
    manager: TransferManager,

    /// Remote handle mappings for peer-to-peer transfers.
    /// Key: (InstanceId, LogicalLayoutHandle) → remote LayoutHandle
    ///
    /// Populated by `connect_remote` when this worker imports metadata from
    /// a peer instance. Used by `execute_remote_onboard_for_instance` to
    /// resolve logical handles to physical handles for RDMA transfers.
    ///
    /// Note: This is per-instance mapping (no rank), suitable for single-worker
    /// scenarios. For multi-worker asymmetric TP, use CoordinatedWorker's
    /// rank-aware remote_handles instead.
    remote_handles: RwLock<HashMap<(InstanceId, LogicalLayoutHandle), LayoutHandle>>,
}

impl DirectWorker {
    pub fn new(manager: TransferManager) -> Self {
        Self {
            g1_handle: OnceLock::new(),
            g2_handle: OnceLock::new(),
            g3_handle: OnceLock::new(),
            manager,
            remote_handles: RwLock::new(HashMap::new()),
        }
    }

    /// Set the G1 layout handle (once only).
    ///
    /// This is called during Phase 2 when GPU KV cache tensors are registered with NIXL.
    pub fn set_g1_handle(&self, handle: LayoutHandle) -> Result<()> {
        self.g1_handle
            .set(handle)
            .map_err(|_| anyhow::anyhow!("G1 handle already set"))
    }

    /// Set the G2 layout handle (once only).
    ///
    /// This is called during Phase 3 when host/pinned cache is created.
    pub fn set_g2_handle(&self, handle: LayoutHandle) -> Result<()> {
        self.g2_handle
            .set(handle)
            .map_err(|_| anyhow::anyhow!("G2 handle already set"))
    }

    /// Set the G3 layout handle (once only).
    ///
    /// This is called during Phase 3 when disk cache is created (if enabled).
    pub fn set_g3_handle(&self, handle: LayoutHandle) -> Result<()> {
        self.g3_handle
            .set(handle)
            .map_err(|_| anyhow::anyhow!("G3 handle already set"))
    }

    /// Get the G1 layout handle (if set).
    pub fn g1_handle(&self) -> Option<LayoutHandle> {
        self.g1_handle.get().copied()
    }

    /// Get the G2 layout handle (if set).
    pub fn g2_handle(&self) -> Option<LayoutHandle> {
        self.g2_handle.get().copied()
    }

    /// Get the G3 layout handle (if set).
    pub fn g3_handle(&self) -> Option<LayoutHandle> {
        self.g3_handle.get().copied()
    }

    /// Get a reference to the TransferManager.
    pub fn transfer_manager(&self) -> &TransferManager {
        &self.manager
    }

    /// Create a bounce buffer specification from a layout handle and block IDs.
    pub fn create_bounce_buffer(
        &self,
        handle: LayoutHandle,
        block_ids: Vec<BlockId>,
    ) -> Result<BounceBuffer> {
        Ok(BounceBuffer::from_handle(handle, block_ids))
    }

    /// Export serialized layout metadata with proper logical type mappings.
    ///
    /// This exports layouts with their logical types (G1, G2, G3) so that
    /// remote instances can correctly identify which handle corresponds to
    /// which tier during RDMA transfers.
    pub fn export_metadata(&self) -> Result<SerializedLayout> {
        self.export_metadata_with_logical_types()
    }

    /// Export metadata with logical type annotations for each registered handle.
    fn export_metadata_with_logical_types(&self) -> Result<SerializedLayout> {
        let mut descriptors = Vec::new();

        // Build descriptors for each registered logical handle
        if let Some(handle) = self.g1_handle() {
            descriptors.push(
                self.manager
                    .build_logical_descriptor(handle, LogicalLayoutHandle::G1)?,
            );
        }
        if let Some(handle) = self.g2_handle() {
            descriptors.push(
                self.manager
                    .build_logical_descriptor(handle, LogicalLayoutHandle::G2)?,
            );
        }
        if let Some(handle) = self.g3_handle() {
            descriptors.push(
                self.manager
                    .build_logical_descriptor(handle, LogicalLayoutHandle::G3)?,
            );
        }

        // Pack with worker address and NIXL metadata
        let worker_address = self.manager.worker_address();
        let nixl_metadata = self.manager.get_nixl_metadata()?;

        SerializedLayout::pack(worker_address, nixl_metadata, descriptors)
    }

    /// Import serialized layout metadata into the transfer manager.
    pub fn import_metadata(&self, metadata: SerializedLayout) -> Result<Vec<LayoutHandle>> {
        self.manager.import_metadata(metadata)
    }

    // /// Export the layout configuration from the G1 handle for leader validation.
    // ///
    // /// The leader gathers this from all workers and validates they match before
    // /// proceeding with G2/G3 layout creation.
    // pub fn export_layout_config(&self) -> Result<LayoutConfig> {
    //     let g1_handle = self
    //         .g1_handle()
    //         .ok_or_else(|| anyhow::anyhow!("G1 handle not set - cannot export layout config"))?;

    //     self.manager.get_layout_config(g1_handle)
    // }

    // /// Configure additional layouts (G2, G3) based on leader-provided configuration.
    // ///
    // /// This is called via the Nova handler during Phase 3 coordination.
    // /// The method:
    // /// 1. Enables required NIXL backends (POSIX for host, GDS for disk)
    // /// 2. Creates G2 (host/pinned) layout if host_block_count > 0
    // /// 3. Creates G3 (disk) layout if disk_block_count is set
    // /// 4. Exports updated metadata including all layouts
    // ///
    // /// # Arguments
    // /// * `config` - Leader-provided configuration specifying block counts and backends
    // ///
    // /// # Returns
    // /// Response containing updated metadata and list of created layouts
    // pub fn configure_additional_layouts(
    //     &self,
    //     config: LeaderLayoutConfig,
    // ) -> Result<WorkerLayoutResponse> {
    //     // Get G1 config as template for G2/G3
    //     let g1_config = self.export_layout_config()?;
    //     let mut created_layouts = vec![LogicalLayoutHandle::G1];

    //     // Get NixlAgent (shared across all layouts)
    //     let nixl_agent = self.manager.nixl_agent().clone();

    //     // ========== CREATE G2 (PINNED HOST) ==========
    //     if config.host_block_count > 0 {
    //         // 1. Build G2 config (same dims as G1, different block count)
    //         let g2_config = LayoutConfig::builder()
    //             .num_blocks(config.host_block_count)
    //             .num_layers(g1_config.num_layers)
    //             .outer_dim(g1_config.outer_dim)
    //             .page_size(g1_config.page_size)
    //             .inner_dim(g1_config.inner_dim)
    //             .dtype_width_bytes(g1_config.dtype_width_bytes)
    //             .alignment(g1_config.alignment)
    //             .inner_shape(g1_config.inner_shape)
    //             .build()
    //             .map_err(|e| anyhow::anyhow!("Failed to build G2 config: {:?}", e))?;

    //         // 2. Create PhysicalLayout with pinned storage (builder handles CUDA)
    //         let g2_layout = PhysicalLayout::builder(nixl_agent.clone())
    //             .with_config(g2_config)
    //             .fully_contiguous()
    //             .allocate_pinned(false) // numa_aware=false
    //             .build()?;

    //         // 3. Register with TransferManager
    //         let g2_handle = self.manager.register_layout(g2_layout)?;

    //         // 4. Store handle for future transfers
    //         self.set_g2_handle(g2_handle)?;
    //         created_layouts.push(LogicalLayoutHandle::G2);

    //         tracing::info!(
    //             ?g2_handle,
    //             block_count = config.host_block_count,
    //             "Created G2 (pinned host) layout"
    //         );
    //     }

    //     // ========== CREATE G3 (DISK STORAGE) ==========
    //     if let Some(disk_blocks) = config.disk_block_count {
    //         // Note: POSIX backend must be configured when NixlAgent is created.
    //         // The builder handles disk storage allocation via mmap independently.

    //         // 1. Build G3 config
    //         let g3_config = LayoutConfig::builder()
    //             .num_blocks(disk_blocks)
    //             .num_layers(g1_config.num_layers)
    //             .outer_dim(g1_config.outer_dim)
    //             .page_size(g1_config.page_size)
    //             .inner_dim(g1_config.inner_dim)
    //             .dtype_width_bytes(g1_config.dtype_width_bytes)
    //             .alignment(g1_config.alignment)
    //             .inner_shape(g1_config.inner_shape)
    //             .build()
    //             .map_err(|e| anyhow::anyhow!("Failed to build G3 config: {:?}", e))?;

    //         // 3. Create PhysicalLayout with disk storage
    //         let g3_layout = PhysicalLayout::builder(nixl_agent.clone())
    //             .with_config(g3_config)
    //             .fully_contiguous()
    //             .allocate_disk(None) // None = temp file (auto-cleanup)
    //             .build()?;

    //         // 4. Register with TransferManager
    //         let g3_handle = self.manager.register_layout(g3_layout)?;
    //         self.set_g3_handle(g3_handle)?;
    //         created_layouts.push(LogicalLayoutHandle::G3);

    //         tracing::info!(
    //             ?g3_handle,
    //             block_count = disk_blocks,
    //             use_gds = config.backend_config.enable_gds,
    //             "Created G3 (disk) layout"
    //         );
    //     }

    //     // Export updated metadata (now includes G2/G3 layouts)
    //     let metadata = self.export_metadata()?;

    //     Ok(WorkerLayoutResponse {
    //         metadata,
    //         created_layouts,
    //     })
    // }
}

impl WorkerTransfers for DirectWorker {
    fn execute_local_transfer(
        &self,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_block_ids: Arc<[BlockId]>,
        dst_block_ids: Arc<[BlockId]>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        use LogicalLayoutHandle::*;

        let src_layout = match &src {
            G1 => self.g1_handle(),
            G2 => self.g2_handle(),
            G3 => self.g3_handle(),
            G4 => return Err(anyhow::anyhow!("G4 is not supported for local transfers")),
        }
        .ok_or_else(|| anyhow::anyhow!("Source layout not registered: {:?}", src))?;

        let dst_layout = match &dst {
            G1 => self.g1_handle(),
            G2 => self.g2_handle(),
            G3 => self.g3_handle(),
            G4 => return Err(anyhow::anyhow!("G4 is not supported for local transfers")),
        }
        .ok_or_else(|| anyhow::anyhow!("Destination layout not registered: {:?}", dst))?;

        self.manager.execute_transfer(
            src_layout,
            &src_block_ids,
            dst_layout,
            &dst_block_ids,
            options,
        )
    }

    fn execute_remote_onboard(
        &self,
        src: RemoteDescriptor,
        dst: LogicalLayoutHandle,
        dst_block_ids: Arc<[BlockId]>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        use LogicalLayoutHandle::*;

        let dst_layout = match &dst {
            G1 => self.g1_handle(),
            G2 => self.g2_handle(),
            G3 => self.g3_handle(),
            G4 => return Err(anyhow::anyhow!("G4 is not supported for remote transfers")),
        }
        .ok_or_else(|| anyhow::anyhow!("Destination layout not registered: {:?}", dst))?;

        match src {
            RemoteDescriptor::Layout { handle, block_ids } => {
                let block_ids_arc: Arc<[BlockId]> = block_ids.into();
                self.manager.execute_transfer(
                    handle,
                    &block_ids_arc,
                    dst_layout,
                    &dst_block_ids,
                    options,
                )
            }
            RemoteDescriptor::Object { keys: _ } => {
                todo!("implement remote object transfer")
            }
        }
    }

    fn execute_remote_offload(
        &self,
        src: LogicalLayoutHandle,
        dst: RemoteDescriptor,
        src_block_ids: Arc<[BlockId]>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        todo!("implement remote offload")
    }

    fn connect_remote(
        &self,
        instance_id: InstanceId,
        metadata: Vec<SerializedLayout>,
    ) -> Result<ConnectRemoteResponse> {
        // DirectWorker expects exactly 1 metadata item
        if metadata.len() != 1 {
            anyhow::bail!(
                "DirectWorker expects exactly 1 metadata item, got {}",
                metadata.len()
            );
        }
        let meta = metadata.into_iter().next().unwrap();

        // Unpack to extract logical type info
        let unpacked = meta.unpack()?;

        // Store mappings
        {
            let mut handles = self.remote_handles.write().unwrap();
            for descriptor in &unpacked.layouts {
                handles.insert((instance_id, descriptor.logical_type), descriptor.handle);
            }
        }

        // Import so NIXL knows about the remote (repack to pass ownership)
        let repacked = SerializedLayout::pack(
            unpacked.worker_address,
            unpacked.nixl_metadata,
            unpacked.layouts,
        )?;
        self.manager.import_metadata(repacked)?;

        Ok(ConnectRemoteResponse::ready())
    }

    fn has_remote_metadata(&self, instance_id: InstanceId) -> bool {
        let handles = self.remote_handles.read().unwrap();
        handles.keys().any(|(id, _)| *id == instance_id)
    }

    fn execute_remote_onboard_for_instance(
        &self,
        instance_id: InstanceId,
        remote_logical_type: LogicalLayoutHandle,
        src_block_ids: Vec<BlockId>,
        dst: LogicalLayoutHandle,
        dst_block_ids: Arc<[BlockId]>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        let handles = self.remote_handles.read().unwrap();
        let remote_handle = handles
            .get(&(instance_id, remote_logical_type))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "No remote {:?} handle for instance {}",
                    remote_logical_type,
                    instance_id
                )
            })?;

        let descriptor = RemoteDescriptor::Layout {
            handle: *remote_handle,
            block_ids: src_block_ids,
        };

        self.execute_remote_onboard(descriptor, dst, dst_block_ids, options)
    }
}

impl Worker for DirectWorker {
    fn g2_handle(&self) -> Option<LayoutHandle> {
        self.g2_handle.get().copied()
    }

    fn export_metadata(&self) -> Result<SerializedLayoutResponse> {
        // Use the logical-type-aware export
        self.export_metadata_with_logical_types()
            .map(SerializedLayoutResponse::ready)
    }

    fn import_metadata(&self, metadata: SerializedLayout) -> Result<ImportMetadataResponse> {
        self.manager
            .import_metadata(metadata)
            .map(ImportMetadataResponse::ready)
    }
}
