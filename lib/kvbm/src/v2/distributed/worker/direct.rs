// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use derive_builder::Builder;
use futures::future::BoxFuture;

use crate::physical::layout::PhysicalLayout;
use crate::physical::{
    manager::{SerializedLayout, TransferManager},
    transfer::{BounceBuffer, TransferOptions, context::TransferCompleteNotification},
};
use crate::v2::distributed::object::ObjectBlockOps;

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
/// 1. Created via `DirectWorker::builder()` during deferred initialization
/// 2. Wrapped by [`NovaWorkerService`] to expose RPC handlers
/// 3. Wrapped by [`CoordinatedWorker`] for leader coordination
///
/// [`CoordinatedWorker`]: super::CoordinatedWorker
/// [`NovaWorkerService`]: super::NovaWorkerService
#[derive(Builder)]
#[builder(pattern = "owned")]
pub struct DirectWorker {
    // =========================================================================
    // Execution State - needed by TransferManager to perform operations
    // =========================================================================
    /// The transfer manager that executes actual data movement.
    manager: TransferManager,

    /// G1 (GPU KV cache) layout handle - set during initialization.
    /// Required for GPU-to-GPU and GPU-to-Host transfers.
    #[builder(default, setter(strip_option))]
    g1_handle: Option<LayoutHandle>,

    /// G2 (Host/pinned cache) layout handle - set during initialization.
    /// Required for Host-to-GPU and Host-to-Disk transfers.
    #[builder(default, setter(strip_option))]
    g2_handle: Option<LayoutHandle>,

    /// G3 (Disk cache) layout handle - set during initialization if disk tier enabled.
    /// Required for Disk-to-Host transfers.
    #[builder(default, setter(strip_option))]
    g3_handle: Option<LayoutHandle>,

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
    #[builder(default = "RwLock::new(HashMap::new())")]
    remote_handles: RwLock<HashMap<(InstanceId, LogicalLayoutHandle), LayoutHandle>>,

    // =========================================================================
    // Object Storage State
    // =========================================================================
    /// Worker rank (set during initialization from LeaderLayoutConfig).
    /// Used to augment object keys for unique storage across SPMD workers.
    #[builder(default, setter(strip_option))]
    rank: Option<usize>,

    /// Optional object storage client for G4 tier operations.
    /// Set during initialization if object storage is enabled.
    #[builder(default, setter(strip_option))]
    object_client: Option<Arc<dyn ObjectBlockOps>>,
}

impl DirectWorker {
    /// Create a new builder for DirectWorker.
    ///
    /// # Example
    /// ```rust,ignore
    /// let worker = DirectWorker::builder()
    ///     .manager(manager)
    ///     .g1_handle(g1_handle)
    ///     .g2_handle(g2_handle)
    ///     .g3_handle(g3_handle)
    ///     .build();
    /// ```
    pub fn builder() -> DirectWorkerBuilder {
        DirectWorkerBuilder::default()
    }

    /// Get the worker rank (if set).
    pub fn rank(&self) -> Option<usize> {
        self.rank
    }

    /// Get the object storage client (if set).
    pub fn object_client(&self) -> Option<&Arc<dyn ObjectBlockOps>> {
        self.object_client.as_ref()
    }

    /// Get the G1 layout handle (if set).
    pub fn g1_handle(&self) -> Option<LayoutHandle> {
        self.g1_handle
    }

    /// Get the G2 layout handle (if set).
    pub fn g2_handle(&self) -> Option<LayoutHandle> {
        self.g2_handle
    }

    /// Get the G3 layout handle (if set).
    pub fn g3_handle(&self) -> Option<LayoutHandle> {
        self.g3_handle
    }

    /// Get a reference to the TransferManager.
    pub fn transfer_manager(&self) -> &TransferManager {
        &self.manager
    }

    /// Resolve a logical layout handle to a physical layout.
    ///
    /// # Arguments
    /// * `logical` - The logical layout handle (G1, G2, G3)
    ///
    /// # Returns
    /// The physical layout for the given logical handle, or an error if not found.
    pub fn resolve_layout(&self, logical: LogicalLayoutHandle) -> Result<PhysicalLayout> {
        use LogicalLayoutHandle::*;

        let physical_handle = match logical {
            G1 => self.g1_handle(),
            G2 => self.g2_handle(),
            G3 => self.g3_handle(),
            _ => None,
        }
        .ok_or_else(|| anyhow::anyhow!("No layout registered for {:?}", logical))?;

        self.manager
            .get_physical_layout(physical_handle)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Layout handle {:?} not found in TransferManager",
                    physical_handle
                )
            })
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
                // RDMA onboard from remote layout
                let block_ids_arc: Arc<[BlockId]> = block_ids.into();
                self.manager.execute_transfer(
                    handle,
                    &block_ids_arc,
                    dst_layout,
                    &dst_block_ids,
                    options,
                )
            }
            RemoteDescriptor::Object { keys } => {
                // Object storage onboard (e.g., S3 → G2)
                let object_client = self
                    .object_client
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Object client not configured"))?
                    .clone();

                // Resolve destination physical layout
                let dst_physical = self.resolve_layout(dst)?;
                let block_ids_vec: Vec<BlockId> = dst_block_ids.to_vec();

                // Create event for completion notification
                let ctx = self.manager.context();
                let event = ctx.event_system().new_event()?;
                let handle = event.handle();
                let awaiter = ctx.event_system().awaiter(handle)?;

                // Spawn task to execute object storage read
                ctx.tokio().spawn(async move {
                    let results = object_client
                        .get_blocks_with_layout(keys.clone(), dst_physical, block_ids_vec)
                        .await;

                    // Check if any failed
                    let failed: Vec<_> = results.iter().filter(|r| r.is_err()).collect();
                    if failed.is_empty() {
                        let _ = event.trigger();
                    } else {
                        let error_msg = format!(
                            "{} of {} blocks failed to download",
                            failed.len(),
                            results.len()
                        );
                        let _ = event.poison(error_msg);
                    }
                });

                Ok(TransferCompleteNotification::from_awaiter(awaiter))
            }
        }
    }

    fn execute_remote_offload(
        &self,
        src: LogicalLayoutHandle,
        src_block_ids: Arc<[BlockId]>,
        dst: RemoteDescriptor,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        match dst {
            RemoteDescriptor::Layout { handle, block_ids } => {
                // RDMA offload to remote layout
                let src_layout = match &src {
                    LogicalLayoutHandle::G1 => self.g1_handle(),
                    LogicalLayoutHandle::G2 => self.g2_handle(),
                    LogicalLayoutHandle::G3 => self.g3_handle(),
                    LogicalLayoutHandle::G4 => {
                        return Err(anyhow::anyhow!("G4 cannot be used as source for offload"));
                    }
                }
                .ok_or_else(|| anyhow::anyhow!("Source layout not registered: {:?}", src))?;

                let block_ids_arc: Arc<[BlockId]> = block_ids.into();
                self.manager.execute_transfer(
                    src_layout,
                    &src_block_ids,
                    handle,
                    &block_ids_arc,
                    _options,
                )
            }
            RemoteDescriptor::Object { keys } => {
                // Object storage offload (e.g., G2 → S3)
                let object_client = self
                    .object_client
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Object client not configured"))?
                    .clone();

                // Resolve source physical layout
                let src_physical = self.resolve_layout(src)?;
                let block_ids_vec: Vec<BlockId> = src_block_ids.to_vec();

                // Create event for completion notification
                let ctx = self.manager.context();
                let event = ctx.event_system().new_event()?;
                let handle = event.handle();
                let awaiter = ctx.event_system().awaiter(handle)?;

                // Spawn task to execute object storage write
                ctx.tokio().spawn(async move {
                    let results = object_client
                        .put_blocks_with_layout(keys.clone(), src_physical, block_ids_vec)
                        .await;

                    // Check if any failed
                    let failed: Vec<_> = results.iter().filter(|r| r.is_err()).collect();
                    if failed.is_empty() {
                        let _ = event.trigger();
                    } else {
                        let error_msg = format!(
                            "{} of {} blocks failed to upload",
                            failed.len(),
                            results.len()
                        );
                        let _ = event.poison(error_msg);
                    }
                });

                Ok(TransferCompleteNotification::from_awaiter(awaiter))
            }
        }
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
    fn g1_handle(&self) -> Option<LayoutHandle> {
        self.g1_handle
    }

    fn g2_handle(&self) -> Option<LayoutHandle> {
        self.g2_handle
    }

    fn g3_handle(&self) -> Option<LayoutHandle> {
        self.g3_handle
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

impl ObjectBlockOps for DirectWorker {
    fn has_blocks(
        &self,
        keys: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Vec<(SequenceHash, Option<usize>)>> {
        // Object client handles rank-based key prefixing internally
        if let Some(client) = self.object_client.as_ref() {
            client.has_blocks(keys)
        } else {
            // No object client configured - return all keys as not found
            Box::pin(async move { keys.into_iter().map(|k| (k, None)).collect() })
        }
    }

    fn put_blocks(
        &self,
        keys: Vec<SequenceHash>,
        src_layout: LogicalLayoutHandle,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        // Resolve logical handle to physical layout
        let physical_layout = match self.resolve_layout(src_layout) {
            Ok(layout) => layout,
            Err(e) => {
                tracing::error!(?src_layout, error = %e, "Failed to resolve layout for put_blocks");
                return Box::pin(async move { keys.into_iter().map(Err).collect() });
            }
        };

        // Object client handles rank-based key prefixing internally
        if let Some(client) = self.object_client.as_ref() {
            client.put_blocks_with_layout(keys, physical_layout, block_ids)
        } else {
            // No object client configured - return all keys as failed
            tracing::warn!("put_blocks called but no object client configured");
            Box::pin(async move { keys.into_iter().map(Err).collect() })
        }
    }

    fn get_blocks(
        &self,
        keys: Vec<SequenceHash>,
        dst_layout: LogicalLayoutHandle,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        // Resolve logical handle to physical layout
        let physical_layout = match self.resolve_layout(dst_layout) {
            Ok(layout) => layout,
            Err(e) => {
                tracing::error!(?dst_layout, error = %e, "Failed to resolve layout for get_blocks");
                return Box::pin(async move { keys.into_iter().map(Err).collect() });
            }
        };

        // Object client handles rank-based key prefixing internally
        if let Some(client) = self.object_client.as_ref() {
            client.get_blocks_with_layout(keys, physical_layout, block_ids)
        } else {
            // No object client configured - return all keys as failed
            tracing::warn!("get_blocks called but no object client configured");
            Box::pin(async move { keys.into_iter().map(Err).collect() })
        }
    }
}
