// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use super::PoolDiscoveryError;
use super::admission::PoolAdmission;
use super::identity::RelayAuthenticator;
use super::record::PoolDirectory;
use super::registrar::{LeasePolicy, PoolRegistrar};
use super::replicas::ReplicaDirectory;

/// Chosen once at startup, like `dynamo_runtime::distributed::DiscoveryBackend`.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum PoolDiscoveryMode {
    Grpc {
        lease: LeasePolicy,
        settle: SettlePolicy,
    },
    Hybrid {
        lease: LeasePolicy,
        hub_namespace: String,
    },
    /// SIG Multicluster: pools are MCS `ServiceExport`s. Liveness is endpoint
    /// readiness, so there is no lease.
    Kubernetes {
        /// Label selector on the hub's `ServiceImport`s.
        pool_selector: String,
        /// The Global Router's headless Service, exported to every cluster.
        router_service: String,
    },
    File {
        path: PathBuf,
    },
}

/// When a gRPC-mode replica treats its pool list as settled (soft readiness).
#[derive(Clone, Copy, Debug)]
pub enum SettlePolicy {
    FixedDelay(Duration),
    /// No new pool for this long.
    QuietPeriod(Duration),
    /// Copy a peer replica's list; fall back to a delay if none answers.
    PeerSnapshot {
        fallback: Duration,
    },
}

/// Everything the router needs from the selected mode. The router code is
/// the same for every mode.
pub struct PoolDiscovery {
    pub pools: Arc<PoolDirectory>,
    pub replicas: Arc<ReplicaDirectory>,
    /// `None` in Kubernetes and File modes.
    pub registrar: Option<Arc<dyn PoolRegistrar>>,
    pub admission: Arc<dyn PoolAdmission>,
    pub authenticator: Arc<dyn RelayAuthenticator>,
}

#[async_trait]
pub trait BuildPoolDiscovery {
    async fn build(self) -> Result<PoolDiscovery, PoolDiscoveryError>;
}
