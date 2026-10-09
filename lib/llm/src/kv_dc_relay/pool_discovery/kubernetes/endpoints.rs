// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Router replicas from EndpointSlices. One type serves both sides: the
//! router reads its own Service's slices in the hub, and the Relay reads the
//! slices MCS imports into its cluster.

use async_trait::async_trait;
use dynamo_kv_router::pool_discovery::watch::{Directory, Snapshot, WatchStream};
use dynamo_kv_router::pool_discovery::{
    McsImplementation, PoolDiscoveryError, ReplicaEndpoint, ReplicaId,
};
use k8s_openapi::api::discovery::v1::EndpointSlice;
use kube::{Api, Client};
use tokio_util::sync::CancellationToken;

/// The Global Router's Service. It need not be headless: imported slices
/// list every router pod, and the Relay reads slices, not DNS.
#[derive(Clone, Debug)]
pub struct RouterService {
    pub namespace: String,
    pub name: String,
    /// Named port for the state stream.
    pub port_name: String,
}

pub struct EndpointSliceReplicaDirectory {
    slices: Api<EndpointSlice>,
    /// From `kubernetes.io/service-name` locally, or
    /// [`super::mcs::service_selector`] for an import.
    selector: String,
    port_name: String,
}

impl EndpointSliceReplicaDirectory {
    /// The router's own slices, in the hub.
    pub fn local(hub: Client, router: &RouterService) -> Self {
        todo!()
    }

    /// The router's slices as MCS imports them into a workload cluster.
    pub fn imported(workload: Client, router: &RouterService, mcs: McsImplementation) -> Self {
        todo!()
    }

    /// Ready endpoints only. The URL comes from the endpoint address and the
    /// named port, formatted through `SocketAddr` so IPv6 gets brackets.
    /// Per-pod DNS names are not used: they exist only for pods with
    /// hostnames.
    fn replicas(&self, slice: &EndpointSlice) -> Vec<ReplicaEndpoint> {
        todo!()
    }
}

#[async_trait]
impl Directory<ReplicaEndpoint, ReplicaId> for EndpointSliceReplicaDirectory {
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
