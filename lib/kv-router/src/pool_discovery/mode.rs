// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;

use super::PoolDiscoveryError;
use super::identity::RelayAuthenticator;
use super::record::PoolDirectory;
use super::registrar::{LeasePolicy, PoolRegistrar};
use super::replicas::ReplicaDirectory;

/// Chosen once at startup, like `dynamo_runtime::distributed::DiscoveryBackend`.
/// Each mode's own settings are added with its implementation.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum PoolDiscoveryMode {
    /// Pools call `RegisterPool`. Records live in router memory.
    Grpc {
        lease: LeasePolicy,
    },
    /// Pools call `RegisterPool`. The Global Discovery Service writes records
    /// to the hub's Kubernetes API.
    Hybrid {
        lease: LeasePolicy,
    },
    /// SIG Multicluster: pools are MCS `ServiceExport`s. Liveness is endpoint
    /// readiness, so there is no lease.
    Kubernetes,
    File {
        path: PathBuf,
    },
}

/// Everything the router needs from the selected mode. The router code is
/// the same for every mode.
pub struct PoolDiscovery {
    pub pools: Arc<PoolDirectory>,
    pub replicas: Arc<ReplicaDirectory>,
    /// `None` in Kubernetes and File modes.
    pub registrar: Option<Arc<dyn PoolRegistrar>>,
    /// Incoming Relays go through `verify` with this, then
    /// [`super::admit`] with `pools`.
    pub authenticator: Arc<dyn RelayAuthenticator>,
}

#[async_trait]
pub trait BuildPoolDiscovery {
    async fn build(self) -> Result<PoolDiscovery, PoolDiscoveryError>;
}
