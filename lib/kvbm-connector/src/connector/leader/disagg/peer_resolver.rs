// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resolve a remote `InstanceId` to its `PeerInfo` and register it on the
//! local velo messenger.
//!
//! The conditional-disagg prefill coordinator needs to talk velo with the
//! decode peer that pushed it a request, but the only identifier carried in
//! `RemotePrefillRequest` is decode's `initiator_instance_id`. The hub holds
//! the `PeerInfo` (registered at decode-startup time); this trait closes the
//! loop by looking it up and feeding it to `messenger.register_peer`. The
//! coordinator caches successful resolves locally so repeat requests for the
//! same peer don't re-pay the round-trip.

use std::sync::Arc;

use anyhow::{Context, Result};
use futures::future::BoxFuture;
use kvbm_hub::HubClient;
use velo::{Messenger, discovery::PeerDiscovery};

use crate::InstanceId;

pub trait PeerResolver: Send + Sync {
    /// Resolve `instance_id` to `PeerInfo` and register it on the local
    /// messenger. Implementations should be safe to call once per remote
    /// peer; the coordinator de-duplicates upstream so repeat resolves
    /// for the same peer never reach the implementation.
    fn resolve_and_register(&self, instance_id: InstanceId) -> BoxFuture<'_, Result<()>>;
}

/// Production resolver: looks up `PeerInfo` via the kvbm-hub and registers
/// it on the local messenger.
pub struct HubPeerResolver {
    hub: Arc<HubClient>,
    messenger: Arc<Messenger>,
}

impl HubPeerResolver {
    pub fn new(hub: Arc<HubClient>, messenger: Arc<Messenger>) -> Arc<Self> {
        Arc::new(Self { hub, messenger })
    }
}

impl PeerResolver for HubPeerResolver {
    fn resolve_and_register(&self, instance_id: InstanceId) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move {
            let peer_info = self
                .hub
                .discover_by_instance_id(instance_id)
                .await
                .with_context(|| format!("hub lookup for instance {}", instance_id))?;
            self.messenger
                .register_peer(peer_info)
                .with_context(|| format!("messenger.register_peer({})", instance_id))?;
            Ok(())
        })
    }
}

/// Test-only no-op resolver. Use when the test harness wires peers
/// directly (e.g., shared in-process velo) and there's nothing to
/// resolve.
#[cfg(any(test, feature = "testing"))]
pub struct NoopPeerResolver;

#[cfg(any(test, feature = "testing"))]
impl PeerResolver for NoopPeerResolver {
    fn resolve_and_register(&self, _instance_id: InstanceId) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move { Ok(()) })
    }
}
