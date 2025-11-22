// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod direct;
mod nova;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::{
    physical::{manager::LayoutHandle, transfer::TransferCompleteNotification},
    v2::{BlockId, InstanceId, SequenceHash, logical::LogicalLayoutHandle},
};

pub use nova::{NovaWorkerClient, NovaWorkerService};

#[derive(Serialize, Deserialize)]
pub enum RemoteDescriptor {
    Layout {
        handle: LayoutHandle,
        block_ids: Vec<BlockId>,
    },
    Object {
        keys: Vec<SequenceHash>,
    },
}

pub trait Worker: Send + Sync {
    /// Execute a local transfer between two logical layouts.
    ///
    /// # Arguments
    /// * `src` - The source layout handle
    /// * `dst` - The destination layout handle
    /// * `src_block_ids` - The source block IDs
    /// * `dst_block_ids` - The destination block IDs
    /// * `options` - Transfer options (layer range, bounce buffers, etc.)
    ///
    /// # Returns
    /// A future that completes when the transfer is complete
    fn execute_local_transfer(
        &self,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_block_ids: Vec<BlockId>,
        dst_block_ids: Vec<BlockId>,
        options: crate::physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification>;

    /// Execute a remote transfer from a remote layout to a local logical layout.
    ///
    /// This represents a NIXL transfer.
    ///
    /// # Arguments
    /// * `src` - Remote sources can take several forms, see [`RemoteDescriptor`]
    /// * `dst` - The destination layout handle
    /// * `dst_block_ids` - The destination block IDs
    /// * `options` - Transfer options (layer range, bounce buffers, etc.)
    ///
    /// # Returns
    /// A future that completes when the transfer is complete
    fn execute_remote_onboard(
        &self,
        src: RemoteDescriptor,
        dst: LogicalLayoutHandle,
        dst_block_ids: Vec<BlockId>,
        options: crate::physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification>;

    /// Execute a remote offload from a local logical layout to a remote descriptor.
    ///
    /// This represents a NIXL offload.
    ///
    /// # Arguments
    /// * `src` - The source layout handle
    /// * `dst` - The destination remote descriptor
    /// * `src_block_ids` - The source block IDs
    /// * `options` - Transfer options (layer range, bounce buffers, etc.)
    ///
    /// # Returns
    /// A future that completes when the offload is complete
    fn execute_remote_offload(
        &self,
        src: LogicalLayoutHandle,
        dst: RemoteDescriptor,
        src_block_ids: Vec<BlockId>,
        options: crate::physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification>;
}
