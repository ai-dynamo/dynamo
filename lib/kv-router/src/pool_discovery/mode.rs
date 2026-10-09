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
    /// readiness, so there is no lease. Assumes a trusted network with pod
    /// addresses that are reachable between clusters.
    Kubernetes {
        mcs: McsImplementation,
        /// The operator names each PoolRelay Service `<DGD name><suffix>`.
        /// A name rule works with every MCS implementation; `exportedLabels`
        /// does not.
        pool_service_suffix: String,
        /// The Global Router's Service, exported to every workload cluster.
        router_service: String,
    },
    File {
        path: PathBuf,
    },
}

/// How the MCS implementation names imported EndpointSlices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum McsImplementation {
    /// The `multicluster.kubernetes.io` labels (Submariner, Cilium, GKE).
    Standard,
    /// Karmada's own names: a `derived-<service>` Service, and the source
    /// cluster in the `work.karmada.io/namespace` annotation.
    Karmada,
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
