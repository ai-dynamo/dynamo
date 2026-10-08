// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dial-in replacement for the Relay's gRPC server (DEP-1276).
//!
//! The Relay opens one connection per pool to each router replica listed by
//! its `Announcement`, and sends the same publications the server sends
//! today. Push RPC vs reverse tunnel is outside DEP-1276.

use std::sync::Arc;

use async_trait::async_trait;
use dynamo_kv_router::global_view::PoolKey;
use dynamo_kv_router::pool_discovery::{PoolDiscoveryError, ReplicaEndpoint, ReplicaId};
use tokio_util::sync::CancellationToken;

use crate::kv_dc_relay::RelayPublicationSource;

/// Counterpart of `global_view_consumer::inbound::RelayListener`.
#[async_trait]
pub trait RelayDialer: Send + Sync {
    async fn dial(
        &self,
        replica: &ReplicaEndpoint,
        pool: &PoolKey,
        publication: Arc<dyn RelayPublicationSource>,
        cancel_token: CancellationToken,
    ) -> Result<Box<dyn OutboundConnection>, PoolDiscoveryError>;
}

/// The router issues `PlaneLease`s when it accepts. A new Relay incarnation
/// must dial again.
#[async_trait]
pub trait OutboundConnection: Send + Sync {
    fn replica(&self) -> &ReplicaId;

    /// Resolves when the router replica disconnects. The Relay redials while
    /// the replica is still listed.
    async fn closed(&self);
}
