// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The state connections a Relay opens to each router replica. Not part of
//! discovery; shown so the flow is complete. One implementation serves every
//! mode.
//!
//! Subject to change: the transport (a push RPC from the Relay, or a reverse
//! tunnel) is not decided yet.

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use super::PoolDiscoveryError;
use super::record::PoolRecord;
use super::replicas::ReplicaEndpoint;
use crate::global_view::PoolKey;

/// Router side. Checks each caller with `verify`, then `admit`.
#[async_trait]
pub trait RelayListener: Send + Sync {
    async fn accept(
        &self,
        cancel_token: CancellationToken,
    ) -> Result<(PoolRecord, Box<dyn StateConnection>), PoolDiscoveryError>;
}

/// Relay side. Dials every replica in its `ReplicaDirectory`.
#[async_trait]
pub trait RelayDialer: Send + Sync {
    async fn dial(
        &self,
        replica: &ReplicaEndpoint,
        pool: &PoolKey,
        cancel_token: CancellationToken,
    ) -> Result<Box<dyn StateConnection>, PoolDiscoveryError>;
}

/// One open connection, on either side.
#[async_trait]
pub trait StateConnection: Send + Sync {
    /// Resolves when the other side disconnects.
    async fn closed(&self);
}
