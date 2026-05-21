// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resolve a remote `InstanceId` to its `PeerInfo` and register it on the
//! local velo messenger.
//!
//! The conditional-disagg prefill coordinator needs to talk velo with the
//! decode peer that pushed it a request, but the only identifier carried in
//! `RemotePrefillRequest` is decode's `initiator_instance_id`. The hub holds
//! the `PeerInfo` (registered at decode-startup time); this trait closes the
//! loop by looking it up and feeding it to `velo.register_peer`. The
//! coordinator caches successful resolves locally so repeat requests for the
//! same peer don't re-pay the round-trip.
//!
//! The trait itself lives in `kvbm_engine::disagg::session` because
//! `VeloSessionFactory` also needs the hook (both the puller-side `attach`
//! and the holder-side `Frame::Attach` receive path call `attach_anchor`,
//! which fails if the peer isn't already in velo's streaming registry).
//! We re-export it here so connector callers keep the canonical
//! `crate::leader::disagg::peer_resolver::PeerResolver` import path.

use std::sync::Arc;

use anyhow::{Context, Result};
use futures::future::BoxFuture;
use kvbm_hub::HubClient;
use velo::{Velo, discovery::PeerDiscovery};

use crate::InstanceId;

pub use kvbm_engine::disagg::session::PeerResolver;

/// Production resolver: looks up `PeerInfo` via the kvbm-hub and registers
/// it on the local Velo instance. Velo::register_peer populates BOTH the
/// messenger registry AND the streaming-transport registry from a single
/// PeerInfo (provided the PeerInfo's WorkerAddress carries the streaming
/// endpoint — which `register_with_hub` ensures by using `velo.peer_info()`
/// rather than `messenger.peer_info()` at hub-registration time).
/// Calling `messenger.register_peer` directly would skip the streaming
/// registry and surface as "TCP streaming: peer <worker_id> not registered"
/// on the next `attach_anchor`.
pub struct HubPeerResolver {
    hub: Arc<HubClient>,
    velo: Arc<Velo>,
}

impl HubPeerResolver {
    pub fn new(hub: Arc<HubClient>, velo: Arc<Velo>) -> Arc<Self> {
        Arc::new(Self { hub, velo })
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
            self.velo
                .register_peer(peer_info)
                .with_context(|| format!("velo.register_peer({})", instance_id))?;
            Ok(())
        })
    }
}

/// Test-only no-op resolver. Use when the test harness wires peers
/// directly (e.g., shared in-process velo) and there's nothing to
/// resolve.
#[cfg(any(test, feature = "testing"))]
#[allow(dead_code)] // test/utility resolver; not constructed in this crate today
pub struct NoopPeerResolver;

#[cfg(any(test, feature = "testing"))]
impl PeerResolver for NoopPeerResolver {
    fn resolve_and_register(&self, _instance_id: InstanceId) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move { Ok(()) })
    }
}
