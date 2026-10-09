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
use kube::{Api, Client};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use super::crd::DynamoPoolExport;

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

pub struct KubernetesAnnouncement {
    exports: Api<DynamoPoolExport>,
    name: String,
    renewals: CancellationToken,
    replicas: watch::Receiver<Vec<ReplicaEndpoint>>,
}

#[async_trait]
impl Announcement for KubernetesAnnouncement {
    fn replicas(&self) -> watch::Receiver<Vec<ReplicaEndpoint>> {
        todo!()
    }

    /// Stops the renewals and deletes the record. The Lease's
    /// `ownerReference` deletes the Lease with it.
    async fn withdraw(self: Box<Self>) -> Result<(), PoolDiscoveryError> {
        todo!()
    }
}

/// Stops the renewals only. Without `withdraw`, the Lease expires.
impl Drop for KubernetesAnnouncement {
    fn drop(&mut self) {
        todo!()
    }
}
