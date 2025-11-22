// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::physical::{
    manager::TransportManager,
    transfer::{TransferOptions, context::TransferCompleteNotification},
};

use super::*;

pub struct DirectWorker {
    g1_handle: Option<LayoutHandle>,
    g2_handle: LayoutHandle,
    g3_handle: Option<LayoutHandle>,
    manager: TransportManager,
}

impl DirectWorker {
    /// Create a bounce buffer specification from a layout handle and block IDs.
    pub fn create_bounce_buffer(
        &self,
        handle: LayoutHandle,
        block_ids: Vec<BlockId>,
    ) -> Result<std::sync::Arc<dyn crate::physical::transfer::BounceBufferSpec>> {
        self.manager.create_bounce_buffer(handle, block_ids)
    }
}

impl Worker for DirectWorker {
    fn execute_local_transfer(
        &self,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_block_ids: Vec<BlockId>,
        dst_block_ids: Vec<BlockId>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        use LogicalLayoutHandle::*;

        let src_layout = match &src {
            G1 => self.g1_handle,
            G2 => Some(self.g2_handle),
            G3 => self.g3_handle,
            G4 => return Err(anyhow::anyhow!("G4 is not supported for local transfers")),
        }
        .ok_or_else(|| anyhow::anyhow!("Invalid source layout"))?;

        let dst_layout = match &dst {
            G1 => self.g1_handle,
            G2 => Some(self.g2_handle),
            G3 => self.g3_handle,
            G4 => return Err(anyhow::anyhow!("G4 is not supported for local transfers")),
        }
        .ok_or_else(|| anyhow::anyhow!("Invalid destination layout"))?;

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
        dst_block_ids: Vec<BlockId>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        use LogicalLayoutHandle::*;

        let dst_layout = match &dst {
            G1 => self.g1_handle,
            G2 => Some(self.g2_handle),
            G3 => self.g3_handle,
            G4 => return Err(anyhow::anyhow!("G4 is not supported for remote transfers")),
        }
        .ok_or_else(|| anyhow::anyhow!("Invalid destination layout"))?;

        match src {
            RemoteDescriptor::Layout { handle, block_ids } => self.manager.execute_transfer(
                handle,
                &block_ids,
                dst_layout,
                &dst_block_ids,
                options,
            ),
            RemoteDescriptor::Object { keys } => {
                todo!("implement remote object transfer")
            }
        }
    }

    fn execute_remote_offload(
        &self,
        src: LogicalLayoutHandle,
        dst: RemoteDescriptor,
        src_block_ids: Vec<BlockId>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        todo!("implement remote offload")
    }
}
