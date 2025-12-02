// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod coordinated;
mod direct;
mod nova;

pub use coordinated::CoordinatedWorker;
pub use direct::DirectWorker;

use anyhow::Result;
use dynamo_nova::events::LocalEventWaiter;
use futures::future::{Either, Ready, ready};
use serde::{Deserialize, Serialize};
use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

pub use crate::{
    physical::{
        manager::{LayoutHandle, SerializedLayout},
        transfer::TransferCompleteNotification,
    },
    v2::{BlockId, InstanceId, SequenceHash, logical::LogicalLayoutHandle},
};

pub use nova::{NovaWorkerClient, NovaWorkerService, NovaWorkerServiceBuilder};

/// Boxed future for serialized layout responses - allows both typed_unary and raw unary results
pub type SerializedResponseAwaiter = Pin<Box<dyn Future<Output = Result<SerializedLayout>> + Send>>;
/// Boxed future for import metadata responses
pub type ImportMetadataResponseAwaiter =
    Pin<Box<dyn Future<Output = Result<Vec<LayoutHandle>>> + Send>>;

pub struct SerializedLayoutResponse {
    awaiter: Either<Ready<Result<SerializedLayout>>, SerializedResponseAwaiter>,
}

impl SerializedLayoutResponse {
    pub fn ready(layout: SerializedLayout) -> Self {
        Self {
            awaiter: Either::Left(ready(Ok(layout))),
        }
    }

    pub fn from_boxed(awaiter: SerializedResponseAwaiter) -> Self {
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

    pub fn from_boxed(awaiter: ImportMetadataResponseAwaiter) -> Self {
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

/// Response type for `connect_remote` operations.
///
/// This type represents the completion state of a remote metadata import
/// with handle mapping storage. Like other response types, it can be awaited.
///
/// For direct workers, this is typically ready immediately.
/// For replicated workers, this aggregates multiple underlying imports.
pub struct ConnectRemoteResponse {
    awaiter: ConnectRemoteAwaiter,
}

pub enum ConnectRemoteAwaiter {
    Ready(Ready<Result<()>>),
    Event(LocalEventWaiter),
}

impl std::future::Future for ConnectRemoteAwaiter {
    type Output = Result<()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.get_mut() {
            Self::Ready(ready) => Pin::new(ready).poll(cx),
            Self::Event(waiter) => Pin::new(waiter).poll(cx),
        }
    }
}

impl ConnectRemoteResponse {
    /// Create a response that is already completed.
    ///
    /// This is used when the connect operation completes synchronously,
    /// such as for DirectWorker with local metadata import.
    pub fn ready() -> Self {
        Self {
            awaiter: ConnectRemoteAwaiter::Ready(ready(Ok(()))),
        }
    }

    /// Create a response from an event waiter.
    ///
    /// This is used when the connect operation requires waiting for
    /// multiple underlying operations to complete (e.g., ReplicatedWorker).
    pub fn from_awaiter(awaiter: LocalEventWaiter) -> Self {
        Self {
            awaiter: ConnectRemoteAwaiter::Event(awaiter),
        }
    }

    /// Check if the response can yield the current task.
    pub fn could_yield(&self) -> bool {
        matches!(self.awaiter, ConnectRemoteAwaiter::Event(_))
    }
}

impl std::future::IntoFuture for ConnectRemoteResponse {
    type Output = Result<()>;
    type IntoFuture = ConnectRemoteAwaiter;

    fn into_future(self) -> Self::IntoFuture {
        self.awaiter
    }
}

/// Remote descriptor for transfer operations.
#[derive(Serialize, Deserialize, Clone)]
pub enum RemoteDescriptor {
    Layout {
        handle: LayoutHandle,
        block_ids: Vec<BlockId>,
    },
    Object {
        keys: Vec<SequenceHash>,
    },
}

/// Configuration sent from leader to workers for G2/G3 layout creation.
///
/// This message is sent via Nova RPC during Phase 3 coordination.
/// Workers use this to create additional cache tiers beyond G1 (GPU KV).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderLayoutConfig {
    /// Number of host/pinned blocks for G2 tier.
    pub host_block_count: usize,

    /// Number of disk blocks for G3 tier (None = no disk tier).
    pub disk_block_count: Option<usize>,
}

/// Worker's response after configuring additional layouts (G2, G3).
///
/// Returned in response to a `LeaderLayoutConfig` request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerLayoutResponse {
    /// Full exported metadata including all registered layouts (G1, G2, G3).
    pub metadata: SerializedLayout,

    /// Which logical layouts were successfully created in this operation.
    pub created_layouts: Vec<LogicalLayoutHandle>,
}

pub trait WorkerTransfers: Send + Sync {
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
        src_block_ids: Arc<[BlockId]>,
        dst_block_ids: Arc<[BlockId]>,
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
        dst_block_ids: Arc<[BlockId]>,
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
        src_block_ids: Arc<[BlockId]>,
        options: crate::physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification>;

    /// Connect to a remote instance by importing its metadata and storing handle mappings.
    ///
    /// This method stores the handle mappings internally for later use by
    /// `execute_remote_onboard_for_instance`. The metadata is also imported into
    /// the underlying transfer manager so NIXL knows about the remote.
    ///
    /// # Arguments
    /// * `instance_id` - The unique identifier of the remote instance
    /// * `metadata` - Serialized layout metadata from the remote instance.
    ///   For DirectWorker, expects exactly 1 element.
    ///   For ReplicatedWorker, expects one element per worker (in rank order).
    ///
    /// # Returns
    /// A response that completes when the metadata has been imported and mappings stored.
    fn connect_remote(
        &self,
        instance_id: InstanceId,
        metadata: Vec<SerializedLayout>,
    ) -> Result<ConnectRemoteResponse>;

    /// Check if remote metadata has been imported for an instance.
    ///
    /// Returns true if `connect_remote` has been successfully called for this instance.
    fn has_remote_metadata(&self, instance_id: InstanceId) -> bool;

    /// Execute a remote onboard transfer using stored handle mapping.
    ///
    /// This method looks up the remote handle from the stored mapping
    /// (established via `connect_remote`) and executes the transfer.
    ///
    /// # Arguments
    /// * `instance_id` - The remote instance to pull from
    /// * `remote_logical_type` - The logical layout type on the remote (e.g., G2)
    /// * `src_block_ids` - Block IDs on the remote to pull
    /// * `dst` - Local destination logical layout
    /// * `dst_block_ids` - Local destination block IDs
    /// * `options` - Transfer options
    ///
    /// # Errors
    /// Returns error if remote metadata hasn't been imported for this instance.
    fn execute_remote_onboard_for_instance(
        &self,
        instance_id: InstanceId,
        remote_logical_type: LogicalLayoutHandle,
        src_block_ids: Vec<BlockId>,
        dst: LogicalLayoutHandle,
        dst_block_ids: Arc<[BlockId]>,
        options: crate::physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification>;
}

pub trait Worker: WorkerTransfers + Send + Sync {
    /// Get the G1 layout handle for this worker (if configured).
    ///
    /// Returns None if no G1 layout has been registered with this worker.
    fn g1_handle(&self) -> Option<LayoutHandle>;

    /// Get the G2 layout handle for this worker (if configured).
    ///
    /// Returns None if no G2 layout has been registered with this worker.
    fn g2_handle(&self) -> Option<LayoutHandle>;

    /// Get the G3 layout handle for this worker (if configured).
    ///
    /// Returns None if no G3 layout has been registered with this worker.
    fn g3_handle(&self) -> Option<LayoutHandle>;

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
