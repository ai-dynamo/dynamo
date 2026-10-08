// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Relay side, in a workload cluster. Uses a hub credential that RBAC limits
//! to the pool's own namespace.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use dynamo_kv_router::pool_discovery::{
    Announcement, PoolAnnouncer, PoolDiscoveryError, PoolRecord, ReplicaDirectory, ReplicaEndpoint,
};
use kube::Client;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

pub struct KubernetesPoolAnnouncer {
    hub: Client,
    /// Router replicas as listed in the hub.
    replicas: Arc<ReplicaDirectory>,
    lease_duration: Duration,
}

impl KubernetesPoolAnnouncer {
    pub fn new(hub: Client, replicas: Arc<ReplicaDirectory>, lease_duration: Duration) -> Self {
        todo!()
    }
}

#[async_trait]
impl PoolAnnouncer for KubernetesPoolAnnouncer {
    /// Server-side applies the `DynamoPoolExport`, creates the `Lease`, and
    /// starts renewing it.
    async fn announce(
        &self,
        record: PoolRecord,
    ) -> Result<Box<dyn Announcement>, PoolDiscoveryError> {
        todo!()
    }
}

/// Drop stops the renewals and deletes the record and the Lease.
pub struct KubernetesAnnouncement {
    renewals: CancellationToken,
    replicas: watch::Receiver<Vec<ReplicaEndpoint>>,
}

impl Announcement for KubernetesAnnouncement {
    fn replicas(&self) -> watch::Receiver<Vec<ReplicaEndpoint>> {
        todo!()
    }
}

impl Drop for KubernetesAnnouncement {
    fn drop(&mut self) {
        todo!()
    }
}
