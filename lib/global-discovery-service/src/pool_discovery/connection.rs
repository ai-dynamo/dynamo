// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The state connections a Relay opens to each router replica. Not part of
//! discovery; shown so the flow is complete. One implementation serves every
//! mode.
//!
//! Subject to change: the transport (a push RPC from the Relay, or a reverse
//! tunnel) is not decided yet.

use std::sync::Arc;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use super::PoolDiscoveryError;
use super::record::PoolRecord;
use super::replicas::{ReplicaDirectory, ReplicaEndpoint};
use dynamo_global_router_shared::PoolKey;

/// Router side. Checks each caller with `verify`, then `admit`.
#[async_trait]
pub trait StateReceiver: Send + Sync {
    async fn accept(
        &self,
        cancel_token: CancellationToken,
    ) -> Result<(PoolRecord, Box<dyn StateConnection>), PoolDiscoveryError>;
}

/// Relay side, one replica. The only part that depends on the transport.
#[async_trait]
pub trait StateDialer: Send + Sync {
    async fn dial(
        &self,
        replica: &ReplicaEndpoint,
        pool: &PoolKey,
        cancel_token: CancellationToken,
    ) -> Result<Box<dyn StateConnection>, PoolDiscoveryError>;
}

/// Relay side, every replica. Keeps one connection to each replica in the
/// `ReplicaDirectory`: dials on `Added`, closes on `Removed`, and dials
/// again when a connection closes. The same in every mode, because the
/// directory already hides the mode.
// Signature only; the body comes with the transport.
#[allow(dead_code)]
pub struct StateBroadcaster {
    replicas: Arc<ReplicaDirectory>,
    dialer: Arc<dyn StateDialer>,
    pool: PoolKey,
}

impl StateBroadcaster {
    pub fn new(
        replicas: Arc<ReplicaDirectory>,
        dialer: Arc<dyn StateDialer>,
        pool: PoolKey,
    ) -> Self {
        Self {
            replicas,
            dialer,
            pool,
        }
    }

    /// Runs until `cancel_token` fires.
    #[allow(unused_variables)]
    pub async fn run(self, cancel_token: CancellationToken) -> Result<(), PoolDiscoveryError> {
        todo!()
    }
}

/// One open connection, on either side.
#[async_trait]
pub trait StateConnection: Send + Sync {
    /// Resolves when the other side disconnects.
    async fn closed(&self);
}
