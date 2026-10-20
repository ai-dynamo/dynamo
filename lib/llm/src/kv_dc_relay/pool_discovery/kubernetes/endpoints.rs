// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Relay side: router replicas from the EndpointSlices MCS imports into the
//! Relay's cluster. The router itself lists its replicas through the hub's
//! Dynamo runtime discovery, like every mode.

// Sketch: fields and parameters are unused until the bodies exist.
#![allow(dead_code, unused_variables)]

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
    /// From [`super::mcs::service_selector`].
    selector: String,
    port_name: String,
}

impl EndpointSliceReplicaDirectory {
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
