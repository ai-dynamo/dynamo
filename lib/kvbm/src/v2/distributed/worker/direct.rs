// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
}

impl DirectWorker {
    pub fn new(manager: TransferManager) -> Self {
        Self {
            g1_handle: None,
            g2_handle: None,
            g3_handle: None,
            manager,
        }
    }

    /// Create a bounce buffer specification from a layout handle and block IDs.
    pub fn create_bounce_buffer(
        &self,
        handle: LayoutHandle,
        block_ids: Vec<BlockId>,
    ) -> Result<BounceBuffer> {
        Ok(BounceBuffer::from_handle(handle, block_ids))
    }

    /// Export serialized layout metadata from the transfer manager.
    pub fn export_metadata(&self) -> Result<SerializedLayout> {
        self.manager.export_metadata()
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
}

impl Worker for DirectWorker {
    fn export_metadata(&self) -> Result<SerializedLayoutResponse> {
        self.manager
            .export_metadata()
            .map(SerializedLayoutResponse::ready)
    }

    fn import_metadata(&self, metadata: SerializedLayout) -> Result<ImportMetadataResponse> {
        self.manager
            .import_metadata(metadata)
            .map(ImportMetadataResponse::ready)
    }
}
