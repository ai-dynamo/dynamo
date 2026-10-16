// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use async_trait::async_trait;

use super::PoolDiscoveryError;
use super::identity::VerifiedRelayIdentity;
use super::record::PoolRecord;
use super::replicas::ReplicaEndpoint;
use crate::global_view::PoolKey;

/// A record expires `lease_duration` after the last heartbeat this router saw,
/// measured on the router's clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LeasePolicy {
    pub heartbeat_interval: Duration,
    pub lease_duration: Duration,
}

/// Reply to `register` and `heartbeat`.
#[derive(Clone, Debug)]
pub struct Registration {
    pub lease: LeasePolicy,
    /// Every router replica. The Relay opens a state connection to each one.
    pub replicas: Vec<ReplicaEndpoint>,
}

/// Write side behind `RegisterPool`, `Heartbeat` and `UnregisterPool`.
/// gRPC mode keeps records in memory. Hybrid writes them to the hub's
/// Kubernetes API. Kubernetes and File modes have no registrar.
#[async_trait]
pub trait PoolRegistrar: Send + Sync {
    async fn register(
        &self,
        record: PoolRecord,
        caller: &VerifiedRelayIdentity,
    ) -> Result<Registration, PoolDiscoveryError>;

    async fn heartbeat(
        &self,
        key: &PoolKey,
        caller: &VerifiedRelayIdentity,
    ) -> Result<Registration, PoolDiscoveryError>;

    async fn unregister(
        &self,
        key: &PoolKey,
        caller: &VerifiedRelayIdentity,
    ) -> Result<(), PoolDiscoveryError>;
}
