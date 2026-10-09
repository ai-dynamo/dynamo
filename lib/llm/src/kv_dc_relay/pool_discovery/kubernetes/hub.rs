// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Router replica side.

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use dynamo_kv_router::global_view::PoolKey;
use dynamo_kv_router::pool_discovery::watch::{Directory, Snapshot, WatchStream};
use dynamo_kv_router::pool_discovery::{
    BuildPoolDiscovery, PeerCredentials, PoolAdmission, PoolDirectory, PoolDiscovery,
    PoolDiscoveryError, PoolRecord, RelayAuthenticator, RelayIdentity, ReplicaEndpoint, ReplicaId,
    VerifiedRelayIdentity,
};
use dynamo_runtime::discovery::Discovery;
use k8s_openapi::api::coordination::v1::Lease;
use kube::{Api, Client};
use tokio_util::sync::CancellationToken;

use super::crd::DynamoPoolExport;

/// Startup for Kubernetes mode. `registrar` is `None`.
pub struct KubernetesPoolDiscovery {
    pub hub: Client,
    pub lease_duration: Duration,
    /// Label selector for the pool namespaces.
    pub pool_selector: String,
    /// Audience the Relay tokens must carry.
    pub audience: String,
}

#[async_trait]
impl BuildPoolDiscovery for KubernetesPoolDiscovery {
    async fn build(self) -> Result<PoolDiscovery, PoolDiscoveryError> {
        todo!()
    }
}

/// Pools from `DynamoPoolExport` + `Lease` watches. Lists are `Exact`.
pub struct KubernetesPoolDirectory {
    exports: Api<DynamoPoolExport>,
    leases: Api<Lease>,
    lease_duration: Duration,
}

impl KubernetesPoolDirectory {
    pub fn new(hub: Client, pool_selector: &str, lease_duration: Duration) -> Self {
        todo!()
    }

    /// Keys whose Lease renewal this replica has not seen for
    /// `lease_duration`, on this replica's clock. Emitted as `Removed`.
    fn expired(&self, now: Instant) -> Vec<PoolKey> {
        todo!()
    }
}

#[async_trait]
impl Directory<PoolRecord, PoolKey> for KubernetesPoolDirectory {
    async fn check_connection(&self) -> Result<(), PoolDiscoveryError> {
        todo!()
    }

    async fn list(&self) -> Result<Snapshot<PoolRecord>, PoolDiscoveryError> {
        todo!()
    }

    /// From the reflector store.
    async fn get(&self, key: &PoolKey) -> Result<Option<PoolRecord>, PoolDiscoveryError> {
        todo!()
    }

    async fn list_and_watch(
        &self,
        cancel_token: Option<CancellationToken>,
    ) -> Result<WatchStream<PoolRecord, PoolKey>, PoolDiscoveryError> {
        todo!()
    }
}

/// Router replicas, from the records they already write through runtime
/// `Discovery`. Not Kubernetes-specific.
pub struct RuntimeReplicaDirectory {
    discovery: Arc<dyn Discovery>,
}

impl RuntimeReplicaDirectory {
    pub fn new(discovery: Arc<dyn Discovery>) -> Self {
        todo!()
    }
}

#[async_trait]
impl Directory<ReplicaEndpoint, ReplicaId> for RuntimeReplicaDirectory {
    async fn check_connection(&self) -> Result<(), PoolDiscoveryError> {
        todo!()
    }

    async fn list(&self) -> Result<Snapshot<ReplicaEndpoint>, PoolDiscoveryError> {
        todo!()
    }

    async fn get(&self, key: &ReplicaId) -> Result<Option<ReplicaEndpoint>, PoolDiscoveryError> {
        todo!()
    }

    async fn list_and_watch(
        &self,
        cancel_token: Option<CancellationToken>,
    ) -> Result<WatchStream<ReplicaEndpoint, ReplicaId>, PoolDiscoveryError> {
        todo!()
    }
}

/// Checks a Relay's hub ServiceAccount token with the hub's `TokenReview`
/// API. The identity is `system:serviceaccount:<namespace>:<name>`.
pub struct TokenReviewAuthenticator {
    hub: Client,
    audience: String,
}

#[async_trait]
impl RelayAuthenticator for TokenReviewAuthenticator {
    async fn authenticate(
        &self,
        credentials: &PeerCredentials,
    ) -> Result<RelayIdentity, PoolDiscoveryError> {
        todo!()
    }
}

/// `PoolDirectory::get`, then the caller must match the record's
/// `relay_identity`. Not Kubernetes-specific.
pub struct ExportAdmission {
    pools: Arc<PoolDirectory>,
}

#[async_trait]
impl PoolAdmission for ExportAdmission {
    async fn admit(
        &self,
        key: &PoolKey,
        caller: &VerifiedRelayIdentity,
    ) -> Result<PoolRecord, PoolDiscoveryError> {
        todo!()
    }
}
