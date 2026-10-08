// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dial-in replacement for [`super::RelayDgdSource`] (DEP-1276).
//!
//! Pool details come from discovery. The Relay-opened connection replaces the
//! Relay and stats channels the router used to dial.

use async_trait::async_trait;
use dynamo_kv_router::pool_discovery::{PoolDiscoveryError, PoolRecord, VerifiedRelayIdentity};
use tokio_util::sync::CancellationToken;

pub struct InboundPoolSource {
    pub record: PoolRecord,
    pub connection: Box<dyn RelayConnection>,
}

/// A connection a Relay opened to this router replica, already authenticated
/// and admitted. The state protocol on it is outside DEP-1276.
#[async_trait]
pub trait RelayConnection: Send + Sync {
    fn caller(&self) -> &VerifiedRelayIdentity;

    /// Resolves when the Relay disconnects, so the runtime can release the
    /// pool's `PlaneLease`s.
    async fn closed(&self);
}

/// The router's gRPC server side. Authenticates with `RelayAuthenticator` and
/// checks with `PoolAdmission` before yielding a source.
#[async_trait]
pub trait RelayListener: Send + Sync {
    async fn accept(
        &self,
        cancel_token: CancellationToken,
    ) -> Result<InboundPoolSource, PoolDiscoveryError>;
}
