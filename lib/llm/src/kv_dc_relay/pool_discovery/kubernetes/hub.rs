// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Router replica side.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use dynamo_kv_router::global_view::PoolKey;
use dynamo_kv_router::global_view::state::PoolLocation;
use dynamo_kv_router::pool_discovery::watch::{Directory, Snapshot, WatchStream};
use dynamo_kv_router::pool_discovery::{
    BuildPoolDiscovery, PeerCredentials, PoolAdmission, PoolDirectory, PoolDiscovery,
    PoolDiscoveryError, PoolRecord, RelayAuthenticator, RelayIdentity, VerifiedRelayIdentity,
};
use k8s_openapi::api::discovery::v1::EndpointSlice;
use kube::{Api, Client};
use tokio_util::sync::CancellationToken;

use super::endpoints::RouterService;
use super::mcs::ServiceImport;

/// Startup for Kubernetes mode. `registrar` is `None`.
pub struct McsPoolDiscovery {
    pub hub: Client,
    /// Label selector on `ServiceImport`s. The operator sets the label through
    /// the `ServiceExport`'s `exportedLabels`.
    pub pool_selector: String,
    pub router: RouterService,
    /// Keyed by About API cluster ID. ClusterProfile properties can fill it.
    pub locations: HashMap<String, PoolLocation>,
}

#[async_trait]
impl BuildPoolDiscovery for McsPoolDiscovery {
    async fn build(self) -> Result<PoolDiscovery, PoolDiscoveryError> {
        todo!()
    }
}

/// One pool per (source cluster, namespace, name), from imported
/// EndpointSlices. Lists are `Exact`. Each slice event updates one pool.
pub struct McsPoolDirectory {
    imports: Api<ServiceImport>,
    slices: Api<EndpointSlice>,
    locations: HashMap<String, PoolLocation>,
}

impl McsPoolDirectory {
    pub fn new(hub: Client, pool_selector: &str, locations: HashMap<String, PoolLocation>) -> Self {
        todo!()
    }

    /// From the slice's source-cluster and service-name labels and its
    /// namespace. One `ServiceImport` can hold several clusters' pools, so
    /// pools are split by slice, not by import.
    fn pool_key(slice: &EndpointSlice) -> Option<PoolKey> {
        todo!()
    }

    /// A pool is live while its PoolRelay endpoint is Ready.
    fn is_live(slice: &EndpointSlice) -> bool {
        todo!()
    }
}

#[async_trait]
impl Directory<PoolRecord, PoolKey> for McsPoolDirectory {
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

/// MCS does not authenticate; the network is trusted. Names the caller from
/// the SPIFFE ID that mesh mTLS proved, and rejects other credentials.
pub struct MeshAuthenticator;

#[async_trait]
impl RelayAuthenticator for MeshAuthenticator {
    async fn authenticate(
        &self,
        credentials: &PeerCredentials,
    ) -> Result<RelayIdentity, PoolDiscoveryError> {
        todo!()
    }
}

/// `PoolDirectory::get`. If the record has a `relay_identity`, the caller
/// must match it. Not Kubernetes-specific.
pub struct DirectoryAdmission {
    pools: Arc<PoolDirectory>,
}

#[async_trait]
impl PoolAdmission for DirectoryAdmission {
    async fn admit(
        &self,
        key: &PoolKey,
        caller: &VerifiedRelayIdentity,
    ) -> Result<PoolRecord, PoolDiscoveryError> {
        todo!()
    }
}
