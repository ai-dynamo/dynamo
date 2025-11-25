// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod direct;
mod nova;

use anyhow::Result;
use futures::future::{Either, Ready, ready};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::{
    physical::{
        manager::{LayoutHandle, SerializedLayout},
        transfer::TransferCompleteNotification,
    },
    v2::{BlockId, InstanceId, SequenceHash, logical::LogicalLayoutHandle},
};

pub use nova::{NovaWorkerClient, NovaWorkerService};

pub type SerializedResponseAwaiter = dynamo_nova::am::TypedUnaryResult<SerializedLayout>;
pub type ImportMetadataResponseAwaiter = dynamo_nova::am::TypedUnaryResult<Vec<LayoutHandle>>;

pub struct SerializedLayoutResponse {
    awaiter: Either<Ready<Result<SerializedLayout>>, SerializedResponseAwaiter>,
}

impl SerializedLayoutResponse {
    pub fn ready(layout: SerializedLayout) -> Self {
        Self {
            awaiter: Either::Left(ready(Ok(layout))),
        }
    }

    pub fn from_awaiter(awaiter: SerializedResponseAwaiter) -> Self {
        Self {
            awaiter: Either::Right(awaiter),
        }
    }

    pub fn could_yield(&self) -> bool {
        matches!(self.awaiter, Either::Right(_))
    }
}

impl std::future::IntoFuture for SerializedLayoutResponse {
    type Output = Result<SerializedLayout>;
    type IntoFuture = Either<Ready<Result<SerializedLayout>>, SerializedResponseAwaiter>;

    fn into_future(self) -> Self::IntoFuture {
        self.awaiter
    }
}

pub struct ImportMetadataResponse {
    awaiter: Either<Ready<Result<Vec<LayoutHandle>>>, ImportMetadataResponseAwaiter>,
}

impl ImportMetadataResponse {
    pub fn ready(handles: Vec<LayoutHandle>) -> Self {
        Self {
            awaiter: Either::Left(ready(Ok(handles))),
        }
    }

    pub fn from_awaiter(awaiter: ImportMetadataResponseAwaiter) -> Self {
        Self {
            awaiter: Either::Right(awaiter),
        }
    }

    pub fn could_yield(&self) -> bool {
        matches!(self.awaiter, Either::Right(_))
    }
}

impl std::future::IntoFuture for ImportMetadataResponse {
    type Output = Result<Vec<LayoutHandle>>;
    type IntoFuture = Either<Ready<Result<Vec<LayoutHandle>>>, ImportMetadataResponseAwaiter>;

    fn into_future(self) -> Self::IntoFuture {
        self.awaiter
    }
}

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

    /// Export the local metadata for this worker.
    ///
    /// # Returns
    /// A [`crate::physical::manager::SerializedLayout`] containing the local metadata
    fn export_metadata(&self) -> Result<SerializedLayoutResponse>;

    /// Import the remote metadata for this worker.
    ///
    /// # Arguments
    /// * `metadata` - A [`crate::physical::manager::SerializedLayout`] containing the remote metadata
    ///
    /// # Returns
    /// A vector of [`crate::physical::manager::LayoutHandle`] for the imported remote layouts
    fn import_metadata(&self, metadata: SerializedLayout) -> Result<ImportMetadataResponse>;
}
