// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::RwLock;

use crate::physical::{
    manager::{SerializedLayout, TransferManager},
    transfer::{BounceBuffer, TransferOptions, context::TransferCompleteNotification},
};

use super::*;

pub struct DirectWorker {
    g1_handle: Option<LayoutHandle>,
    g2_handle: Option<LayoutHandle>,
    g3_handle: Option<LayoutHandle>,
    manager: TransferManager,

    /// Remote handle mappings: (InstanceId, LogicalLayoutHandle) -> remote LayoutHandle.
    /// Populated by `connect_remote` for later use by `execute_remote_onboard_for_instance`.
    remote_handles: RwLock<HashMap<(InstanceId, LogicalLayoutHandle), LayoutHandle>>,
}

impl DirectWorker {
    pub fn new(manager: TransferManager) -> Self {
        Self {
            g1_handle: None,
            g2_handle: None,
            g3_handle: None,
            manager,
            remote_handles: RwLock::new(HashMap::new()),
        }
    }

    /// Set the G1 layout handle.
    pub fn with_g1_handle(mut self, handle: LayoutHandle) -> Self {
        self.g1_handle = Some(handle);
        self
    }

    /// Set the G2 layout handle.
    pub fn with_g2_handle(mut self, handle: LayoutHandle) -> Self {
        self.g2_handle = Some(handle);
        self
    }

    /// Set the G3 layout handle.
    pub fn with_g3_handle(mut self, handle: LayoutHandle) -> Self {
        self.g3_handle = Some(handle);
        self
    }

    /// Get the G2 layout handle (if set).
    pub fn g2_handle(&self) -> Option<LayoutHandle> {
        self.g2_handle
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
        if let Some(handle) = self.g1_handle {
            descriptors.push(
                self.manager
                    .build_logical_descriptor(handle, LogicalLayoutHandle::G1)?,
            );
        }
        if let Some(handle) = self.g2_handle {
            descriptors.push(
                self.manager
                    .build_logical_descriptor(handle, LogicalLayoutHandle::G2)?,
            );
        }
        if let Some(handle) = self.g3_handle {
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
            G1 => self.g1_handle,
            G2 => self.g2_handle,
            G3 => self.g3_handle,
            G4 => return Err(anyhow::anyhow!("G4 is not supported for local transfers")),
        }
        .ok_or_else(|| anyhow::anyhow!("Source layout not registered: {:?}", src))?;

        let dst_layout = match &dst {
            G1 => self.g1_handle,
            G2 => self.g2_handle,
            G3 => self.g3_handle,
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
            G1 => self.g1_handle,
            G2 => self.g2_handle,
            G3 => self.g3_handle,
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
        self.g2_handle
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
