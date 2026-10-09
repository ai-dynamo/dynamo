// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pool side, running next to the Relay in a workload cluster.

use async_trait::async_trait;
use tokio::sync::watch;

use super::PoolDiscoveryError;
use super::record::PoolRecord;
use super::replicas::ReplicaEndpoint;

/// gRPC and Hybrid modes call `RegisterPool` on the regional router address.
/// Kubernetes mode writes the record to the hub and renews its Lease.
#[async_trait]
pub trait PoolAnnouncer: Send + Sync {
    async fn announce(
        &self,
        record: PoolRecord,
    ) -> Result<Box<dyn Announcement>, PoolDiscoveryError>;
}

/// RAII guard. Keeps the pool registered while alive. Call `withdraw` on a
/// clean shutdown; dropping only stops heartbeats, and the lease expires.
#[async_trait]
pub trait Announcement: Send + Sync {
    /// Router replicas this pool must open state connections to.
    fn replicas(&self) -> watch::Receiver<Vec<ReplicaEndpoint>>;

    /// Unregisters the pool and waits for the backend to confirm.
    async fn withdraw(self: Box<Self>) -> Result<(), PoolDiscoveryError>;
}
