// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Peer discovery for the Dynamo Active Message system.

use anyhow::Result;
use futures::future::BoxFuture;
use std::fmt;
use std::sync::Arc;

mod address;
mod identity;
mod manager;

pub use address::{PeerInfo, WorkerAddress};
pub use identity::{InstanceId, WorkerId};
pub use manager::PeerDiscoveryManager;

/// Error type for discovery operations.
#[derive(Debug, thiserror::Error)]
pub enum DiscoveryError {
    /// Worker ID collision detected - same worker_id registered to different instance
    #[error(
        "Worker ID collision: worker_id {0} already registered to instance {1}, attempted to register to {2}"
    )]
    WorkerIdCollision(WorkerId, InstanceId, InstanceId),

    /// Address checksum mismatch during re-registration
    #[error("Address checksum mismatch for instance {0}: existing=0x{1:016x}, new=0x{2:016x}")]
    ChecksumMismatch(InstanceId, u64, u64),

    /// Instance already registered - duplicate registration detected
    #[error("Instance {0} is already registered")]
    InstanceAlreadyRegistered(InstanceId),

    /// Backend-specific error
    #[error("Backend error: {0}")]
    Backend(#[from] anyhow::Error),
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum DiscoveryQueryError {
    #[error("Not found")]
    NotFound,

    #[error("Backend error: {0}")]
    Backend(Arc<anyhow::Error>),
}

pub type AwaitableQueryResult = BoxFuture<'static, Result<PeerInfo, DiscoveryQueryError>>;
pub type AwaitableRegisterResult = BoxFuture<'static, Result<(), DiscoveryError>>;

/// Trait for discovering [`PeerInfo`] by [`WorkerId`] or [`InstanceId`].
pub trait PeerDiscovery: Send + Sync + fmt::Debug {
    /// Lookup peer by worker_id.
    fn discover_by_worker_id(&self, worker_id: WorkerId) -> AwaitableQueryResult;

    /// Lookup peer by instance_id.
    fn discover_by_instance_id(&self, instance_id: InstanceId) -> AwaitableQueryResult;

    /// Register this peer in the discovery system.
    fn register_instance(
        &self,
        instance_id: InstanceId,
        worker_address: WorkerAddress,
    ) -> AwaitableRegisterResult;

    /// Unregister this peer from the discovery system.
    fn unregister_instance(&self, instance_id: InstanceId) -> AwaitableRegisterResult;
}
