// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Swappable peer-registry backend for the hub.
//!
//! The HTTP handlers and the hub's own `velo::Velo` instance both delegate to
//! the same `Arc<dyn PeerRegistry>`. The default backend is
//! [`InMemoryRegistry`]; future backends (etcd, consul) implement the same
//! trait and plug into [`HubServerBuilder::registry`](crate::HubServerBuilder::registry).
//!
//! Because `PeerRegistry: PeerDiscovery`, an `Arc<dyn PeerRegistry>` coerces
//! to an `Arc<dyn PeerDiscovery>` via Rust stable trait upcasting (1.76+), so
//! the hub's Velo can share the same backend without a separate adapter.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use futures::future::BoxFuture;
use parking_lot::RwLock;
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use velo::discovery::PeerDiscovery;
use velo_ext::{InstanceId, PeerInfo, WorkerId};

/// Errors returned by [`PeerRegistry`] mutations.
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    /// A different `InstanceId` already holds this `worker_id`.
    #[error("worker_id {worker_id} already claimed by instance {existing}")]
    Conflict {
        /// The `WorkerId` that is already claimed.
        worker_id: WorkerId,
        /// The existing `InstanceId` holding that `WorkerId`.
        existing: InstanceId,
    },

    /// Target instance is not registered.
    #[error("instance {0} not found")]
    NotFound(InstanceId),

    /// Opaque backend failure (e.g. etcd connection, network).
    #[error(transparent)]
    Backend(#[from] anyhow::Error),
}

/// Registry trait consumed by the hub HTTP handlers and the hub's own Velo.
///
/// Extends `velo::discovery::PeerDiscovery` so `Arc<dyn PeerRegistry>` upcasts
/// cleanly to `Arc<dyn PeerDiscovery>`.
#[async_trait]
pub trait PeerRegistry: PeerDiscovery + Send + Sync {
    /// Register or re-register a peer.
    ///
    /// Re-registering the same `instance_id` is idempotent — the `PeerInfo`
    /// is updated and the liveness timestamp is reset. A different
    /// `instance_id` attempting to claim the same `worker_id` returns
    /// [`RegistryError::Conflict`].
    async fn register(&self, peer: PeerInfo) -> Result<(), RegistryError>;

    /// Remove a peer. Returns [`RegistryError::NotFound`] if the id was not
    /// registered.
    async fn unregister(&self, id: InstanceId) -> Result<(), RegistryError>;

    /// Refresh the liveness timestamp for a registered peer. Returns
    /// [`RegistryError::NotFound`] if the id was not registered.
    async fn touch(&self, id: InstanceId) -> Result<(), RegistryError>;

    /// Is this instance currently registered?
    fn contains(&self, id: InstanceId) -> bool;

    /// Snapshot all registered peers.
    fn list(&self) -> Vec<PeerInfo>;

    /// Spawn a backend-specific liveness/reaper task. Returns `None` for
    /// backends with native lease support (etcd). The caller owns the
    /// returned `JoinHandle` and is expected to cancel via the token at
    /// shutdown.
    fn spawn_liveness_task(self: Arc<Self>, _cancel: CancellationToken) -> Option<JoinHandle<()>> {
        None
    }
}

// ---------------------------------------------------------------------------
// In-memory backend
// ---------------------------------------------------------------------------

/// Default in-memory backend. Three maps behind one `parking_lot::RwLock`,
/// TTL-based eviction driven by [`InMemoryRegistry::spawn_liveness_task`].
///
/// Use [`InMemoryRegistry::protect`] to exempt an instance (e.g. the hub's
/// own self-entry) from TTL eviction.
pub struct InMemoryRegistry {
    inner: RwLock<RegistryInner>,
    ttl: Duration,
    prune_interval: Duration,
    eviction_cb: RwLock<Option<EvictionCallback>>,
}

/// Callback invoked whenever an instance leaves the registry (explicit
/// unregister or reaper-driven TTL eviction). Always called **outside** the
/// registry lock. Idempotency is the callback's responsibility.
pub type EvictionCallback = Arc<dyn Fn(InstanceId) + Send + Sync + 'static>;

#[derive(Default)]
struct RegistryInner {
    by_instance: HashMap<InstanceId, PeerInfo>,
    by_worker: HashMap<WorkerId, InstanceId>,
    last_seen: HashMap<InstanceId, Instant>,
    protected: HashSet<InstanceId>,
}

impl std::fmt::Debug for InMemoryRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InMemoryRegistry")
            .field("peers", &self.inner.read().by_instance.len())
            .field("ttl", &self.ttl)
            .field("prune_interval", &self.prune_interval)
            .finish()
    }
}

impl InMemoryRegistry {
    /// Builder entry point.
    pub fn builder() -> InMemoryRegistryBuilder {
        InMemoryRegistryBuilder::default()
    }

    /// TTL after which an unfresh entry is eligible for eviction.
    pub fn ttl(&self) -> Duration {
        self.ttl
    }

    /// Interval between reaper ticks.
    pub fn prune_interval(&self) -> Duration {
        self.prune_interval
    }

    /// Protect an instance from TTL eviction. Used by the hub to pin its
    /// own self-entry so it stays discoverable regardless of touches.
    pub fn protect(&self, id: InstanceId) {
        self.inner.write().protected.insert(id);
    }

    /// Install (or replace) the eviction callback. Called for every instance
    /// removed via explicit unregister or the reaper's TTL pass. Always
    /// invoked outside the registry lock.
    pub fn set_eviction_callback(&self, cb: EvictionCallback) {
        *self.eviction_cb.write() = Some(cb);
    }

    fn eviction_callback(&self) -> Option<EvictionCallback> {
        self.eviction_cb.read().as_ref().map(Arc::clone)
    }

    /// Immediately evict entries older than `ttl` that are not protected.
    /// Normally called from the reaper task; exposed for tests.
    pub fn prune_stale(&self) {
        let evicted = {
            let mut w = self.inner.write();
            let now = Instant::now();
            let ttl = self.ttl;
            let stale: Vec<InstanceId> = w
                .last_seen
                .iter()
                .filter_map(|(id, seen)| {
                    if w.protected.contains(id) {
                        None
                    } else if now.saturating_duration_since(*seen) > ttl {
                        Some(*id)
                    } else {
                        None
                    }
                })
                .collect();
            for id in &stale {
                if let Some(peer) = w.by_instance.remove(id) {
                    w.by_worker.remove(&peer.worker_id());
                }
                w.last_seen.remove(id);
            }
            stale
        };
        if !evicted.is_empty()
            && let Some(cb) = self.eviction_callback()
        {
            for id in evicted {
                cb(id);
            }
        }
    }
}

/// Builder for [`InMemoryRegistry`].
pub struct InMemoryRegistryBuilder {
    ttl: Duration,
    prune_interval: Duration,
}

impl Default for InMemoryRegistryBuilder {
    fn default() -> Self {
        Self {
            ttl: Duration::from_secs(30),
            prune_interval: Duration::from_secs(10),
        }
    }
}

impl InMemoryRegistryBuilder {
    /// Liveness TTL (default `30s`). Entries older than this are evicted.
    pub fn ttl(mut self, d: Duration) -> Self {
        self.ttl = d;
        self
    }

    /// Reaper tick interval (default `10s`).
    pub fn prune_interval(mut self, d: Duration) -> Self {
        self.prune_interval = d;
        self
    }

    /// Finalize the registry.
    pub fn build(self) -> InMemoryRegistry {
        InMemoryRegistry {
            inner: RwLock::new(RegistryInner::default()),
            ttl: self.ttl,
            prune_interval: self.prune_interval,
            eviction_cb: RwLock::new(None),
        }
    }
}

impl PeerDiscovery for InMemoryRegistry {
    fn discover_by_worker_id(&self, worker_id: WorkerId) -> BoxFuture<'_, Result<PeerInfo>> {
        let result = {
            let r = self.inner.read();
            match r.by_worker.get(&worker_id).copied() {
                Some(id) => r.by_instance.get(&id).cloned().ok_or_else(|| {
                    anyhow!(
                        "registry inconsistency: worker {} mapped to missing instance {}",
                        worker_id.as_u64(),
                        id
                    )
                }),
                None => Err(anyhow!("worker {} not registered", worker_id.as_u64())),
            }
        };
        Box::pin(async move { result })
    }

    fn discover_by_instance_id(&self, id: InstanceId) -> BoxFuture<'_, Result<PeerInfo>> {
        let result = self
            .inner
            .read()
            .by_instance
            .get(&id)
            .cloned()
            .ok_or_else(|| anyhow!("instance {id} not registered"));
        Box::pin(async move { result })
    }
}

#[async_trait]
impl PeerRegistry for InMemoryRegistry {
    async fn register(&self, peer: PeerInfo) -> Result<(), RegistryError> {
        let id = peer.instance_id();
        let wid = peer.worker_id();
        let mut w = self.inner.write();
        if let Some(existing) = w.by_worker.get(&wid).copied()
            && existing != id
        {
            return Err(RegistryError::Conflict {
                worker_id: wid,
                existing,
            });
        }
        w.by_worker.insert(wid, id);
        w.by_instance.insert(id, peer);
        w.last_seen.insert(id, Instant::now());
        Ok(())
    }

    async fn unregister(&self, id: InstanceId) -> Result<(), RegistryError> {
        {
            let mut w = self.inner.write();
            let peer = w
                .by_instance
                .remove(&id)
                .ok_or(RegistryError::NotFound(id))?;
            w.by_worker.remove(&peer.worker_id());
            w.last_seen.remove(&id);
            w.protected.remove(&id);
        }
        if let Some(cb) = self.eviction_callback() {
            cb(id);
        }
        Ok(())
    }

    async fn touch(&self, id: InstanceId) -> Result<(), RegistryError> {
        let mut w = self.inner.write();
        if !w.by_instance.contains_key(&id) {
            return Err(RegistryError::NotFound(id));
        }
        w.last_seen.insert(id, Instant::now());
        Ok(())
    }

    fn contains(&self, id: InstanceId) -> bool {
        self.inner.read().by_instance.contains_key(&id)
    }

    fn list(&self) -> Vec<PeerInfo> {
        self.inner.read().by_instance.values().cloned().collect()
    }

    fn spawn_liveness_task(self: Arc<Self>, cancel: CancellationToken) -> Option<JoinHandle<()>> {
        let prune_interval = self.prune_interval;
        Some(tokio::spawn(async move {
            let mut ticker = tokio::time::interval(prune_interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            ticker.tick().await; // discard the immediate first tick
            loop {
                tokio::select! {
                    _ = cancel.cancelled() => break,
                    _ = ticker.tick() => self.prune_stale(),
                }
            }
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use velo_ext::WorkerAddress;

    fn make_peer() -> PeerInfo {
        PeerInfo::new(
            InstanceId::new_v4(),
            WorkerAddress::from_encoded(b"test".to_vec()),
        )
    }

    #[tokio::test]
    async fn register_and_list() {
        let reg = InMemoryRegistry::builder().build();
        let peer = make_peer();
        reg.register(peer.clone()).await.unwrap();
        assert!(reg.contains(peer.instance_id()));
        assert_eq!(reg.list().len(), 1);
    }

    #[tokio::test]
    async fn reregister_same_instance_is_idempotent() {
        let reg = InMemoryRegistry::builder().build();
        let peer = make_peer();
        reg.register(peer.clone()).await.unwrap();
        reg.register(peer.clone()).await.unwrap();
        assert_eq!(reg.list().len(), 1);
    }

    #[tokio::test]
    async fn conflict_on_different_instance_same_worker() {
        let reg = InMemoryRegistry::builder().build();
        let a = make_peer();
        // b shares the worker_id of a (worker_id is derived from instance_id,
        // so we craft a PeerInfo that reuses a's instance_id → same worker_id).
        // Use a different instance_id but fake-shared worker: easiest is to
        // construct two peers whose worker ids happen to match — in practice
        // we exercise the path by having two distinct ids that share a worker.
        // Since WorkerId is derived from InstanceId, we test the conflict path
        // by first registering a, then constructing b with the SAME worker_id
        // via a different instance. This requires accessing internals, so we
        // do it by directly poking the by_worker map with a known id and then
        // calling register() with a *different* instance that reuses the worker.
        reg.register(a.clone()).await.unwrap();
        // Simulate a second instance claiming the same worker by writing the
        // conflict directly — this exercises the conflict-detection branch.
        let a_wid = a.worker_id();
        let different_instance = InstanceId::new_v4();
        {
            let mut w = reg.inner.write();
            w.by_worker.insert(a_wid, different_instance);
        }
        // Now a's register should see the conflict.
        let err = reg.register(a.clone()).await.unwrap_err();
        assert!(matches!(err, RegistryError::Conflict { .. }));
    }

    #[tokio::test]
    async fn unregister_removes_from_all_maps() {
        let reg = InMemoryRegistry::builder().build();
        let peer = make_peer();
        reg.register(peer.clone()).await.unwrap();
        reg.unregister(peer.instance_id()).await.unwrap();
        assert!(!reg.contains(peer.instance_id()));
        assert!(reg.discover_by_worker_id(peer.worker_id()).await.is_err());
    }

    #[tokio::test]
    async fn unregister_not_found() {
        let reg = InMemoryRegistry::builder().build();
        let err = reg.unregister(InstanceId::new_v4()).await.unwrap_err();
        assert!(matches!(err, RegistryError::NotFound(_)));
    }

    #[tokio::test]
    async fn touch_not_found() {
        let reg = InMemoryRegistry::builder().build();
        let err = reg.touch(InstanceId::new_v4()).await.unwrap_err();
        assert!(matches!(err, RegistryError::NotFound(_)));
    }

    #[tokio::test(start_paused = true)]
    async fn prune_stale_removes_old_entries() {
        let reg = InMemoryRegistry::builder()
            .ttl(Duration::from_millis(100))
            .prune_interval(Duration::from_millis(50))
            .build();
        let peer = make_peer();
        reg.register(peer.clone()).await.unwrap();
        tokio::time::advance(Duration::from_millis(200)).await;
        reg.prune_stale();
        assert!(!reg.contains(peer.instance_id()));
    }

    #[tokio::test(start_paused = true)]
    async fn touch_refreshes_ttl() {
        let reg = InMemoryRegistry::builder()
            .ttl(Duration::from_millis(100))
            .build();
        let peer = make_peer();
        reg.register(peer.clone()).await.unwrap();
        tokio::time::advance(Duration::from_millis(80)).await;
        reg.touch(peer.instance_id()).await.unwrap();
        tokio::time::advance(Duration::from_millis(80)).await;
        reg.prune_stale();
        assert!(reg.contains(peer.instance_id()));
    }

    #[tokio::test(start_paused = true)]
    async fn protect_exempts_from_prune() {
        let reg = InMemoryRegistry::builder()
            .ttl(Duration::from_millis(100))
            .build();
        let peer = make_peer();
        reg.register(peer.clone()).await.unwrap();
        reg.protect(peer.instance_id());
        tokio::time::advance(Duration::from_millis(500)).await;
        reg.prune_stale();
        assert!(reg.contains(peer.instance_id()));
    }
}
