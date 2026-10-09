// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Router replica side.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;

use async_trait::async_trait;
use dynamo_kv_router::global_view::PoolKey;
use dynamo_kv_router::global_view::state::PoolLocation;
use dynamo_kv_router::pool_discovery::watch::{Directory, Snapshot, WatchStream};
use dynamo_kv_router::pool_discovery::{
    BuildPoolDiscovery, McsImplementation, PeerCredentials, PoolAdmission, PoolDirectory,
    PoolDiscovery, PoolDiscoveryError, PoolRecord, RelayAuthenticator, RelayIdentity,
    VerifiedRelayIdentity,
};
use k8s_openapi::api::discovery::v1::EndpointSlice;
use kube::{Api, Client};
use tokio_util::sync::CancellationToken;

use super::endpoints::RouterService;

/// Startup for Kubernetes mode. `registrar` is `None`.
pub struct McsPoolDiscovery {
    /// The router's own cluster, for its replica slices.
    pub hub: Client,
    /// Where the imported pool slices are. The hub for standard MCS. For
    /// Karmada, the Karmada control plane: Karmada collects every exported
    /// Service's slices there, so no per-pool `ServiceImport` is needed.
    pub imports: Client,
    pub mcs: McsImplementation,
    /// See `PoolDiscoveryMode::Kubernetes`.
    pub pool_service_suffix: String,
    pub router: RouterService,
    pub relay_auth: RelayAuth,
    /// Keyed by About API cluster ID. ClusterProfile properties can fill it.
    pub locations: HashMap<String, PoolLocation>,
}

/// How the router names a Relay that dials in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelayAuth {
    /// V1: a trusted network with no mesh. See
    /// [`TrustedNetworkAuthenticator`].
    TrustedNetwork,
    /// Mesh mTLS. Pool records carry no expected identity, because MCS does
    /// not export the Relay's SPIFFE ID.
    Mesh,
}

#[async_trait]
impl BuildPoolDiscovery for McsPoolDiscovery {
    async fn build(self) -> Result<PoolDiscovery, PoolDiscoveryError> {
        todo!()
    }
}

/// One pool per (source cluster, namespace, DGD name), from imported
/// EndpointSlices. MCS merges same-named exports from several clusters into
/// one import, so pools are split by source cluster. Lists are `Exact`. Each
/// slice event updates one pool.
pub struct McsPoolDirectory {
    slices: Api<EndpointSlice>,
    mcs: McsImplementation,
    pool_service_suffix: String,
    /// Sets `relay_identity` to the PoolRelay endpoint's address, for
    /// [`TrustedNetworkAuthenticator`].
    bind_relay_address: bool,
    locations: HashMap<String, PoolLocation>,
}

impl McsPoolDirectory {
    pub fn new(
        imports: Client,
        mcs: McsImplementation,
        pool_service_suffix: String,
        bind_relay_address: bool,
        locations: HashMap<String, PoolLocation>,
    ) -> Self {
        todo!()
    }

    /// Source cluster from [`super::mcs::source_cluster`]; DGD name from the
    /// Service name without `pool_service_suffix`. `None` for imports that
    /// are not pools.
    fn pool_key(&self, slice: &EndpointSlice) -> Option<PoolKey> {
        todo!()
    }

    /// A pool is live while its PoolRelay endpoint is Ready. Ready can be
    /// stale when a cluster loses contact with the MCS implementation, so the
    /// runtime also drops a pool whose Relay connection is gone.
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

/// The identity both sides use for an address: `ip:<address>`.
pub fn address_identity(address: IpAddr) -> RelayIdentity {
    todo!()
}

/// For a trusted network with no mesh. Accepts only
/// `PeerCredentials::PeerAddress` and names the caller with
/// [`address_identity`], which [`DirectoryAdmission`] matches against the
/// pool's Ready PoolRelay endpoint. Traffic between clusters must not be
/// source-NATed: exclude the peer ranges in the AKS ip-masq-agent or the AWS
/// VPC CNI, for example.
pub struct TrustedNetworkAuthenticator;

#[async_trait]
impl RelayAuthenticator for TrustedNetworkAuthenticator {
    async fn authenticate(
        &self,
        credentials: &PeerCredentials,
    ) -> Result<RelayIdentity, PoolDiscoveryError> {
        todo!()
    }
}

/// Names the caller from the SPIFFE ID that mesh mTLS proved, and rejects
/// other credentials. Used with [`RelayAuth::Mesh`].
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
