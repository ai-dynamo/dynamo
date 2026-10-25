// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use async_trait::async_trait;

use super::PoolDiscoveryError;
use super::identity::VerifiedRelayIdentity;
use super::record::PoolRecord;
use dynamo_global_router_shared::PoolKey;

/// A PoolRelay registers again every `heartbeat_interval`. A record expires
/// `lease_duration` after the last registration the store saw, measured on
/// the store's clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LeasePolicy {
    pub heartbeat_interval: Duration,
    pub lease_duration: Duration,
}

/// Reply to `register`. It does not list router replicas: Relays get those
/// from a [`super::ReplicaDirectory`], served as `GET /routers`.
#[derive(Clone, Debug)]
pub struct Registration {
    pub lease: LeasePolicy,
}

/// Write side behind `RegisterPool` and `UnregisterPool`. A PoolRelay calls
/// `register` on a timer, and each call is also the pool's heartbeat:
/// registering an unchanged record again only renews it. gRPC mode keeps
/// records in memory. Hybrid writes them to the hub's Kubernetes API.
/// Kubernetes and File modes have no registrar.
#[async_trait]
pub trait PoolRegistrar: Send + Sync {
    async fn register(
        &self,
        record: PoolRecord,
        caller: &VerifiedRelayIdentity,
    ) -> Result<Registration, PoolDiscoveryError>;

    async fn unregister(
        &self,
        key: &PoolKey,
        caller: &VerifiedRelayIdentity,
    ) -> Result<(), PoolDiscoveryError>;
}
