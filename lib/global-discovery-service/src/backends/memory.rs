// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-memory store: one process keeps every registration. It needs no
//! Kubernetes, so it suits tests and a first deployment. A restarted process
//! starts empty and fills as PoolRelays register again, so its lists are
//! `Estimated` until one lease has passed: by then every live pool has
//! registered again. Router replicas come from configuration until routers
//! register too. Run one replica: each keeps only the registrations it
//! receives.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::Duration;

use crate::pool_discovery::{
    Completeness, Directory, DiscoveryError, Lease, PoolEvent, PoolListing, PoolRegistrar,
    PoolRegistration, ReplicaEndpoint, ReplicaEvent, ReplicaId, Snapshot, WatchEvent, WatchStream,
};
use async_trait::async_trait;
use dynamo_global_common::PoolKey;
use tokio::sync::broadcast;
use tokio::time::{Instant, MissedTickBehavior};
use tokio_util::sync::CancellationToken;

use super::stream::{EVENT_BUFFER, wake_at, watch_stream};

/// Pool registrations in memory. A pool expires the lease duration after its
/// last registration. Reads skip an expired pool at once; a sweep removes it,
/// with a `Removed` event, within one heartbeat interval.
///
/// Any caller may register any pool: V1 assumes a trusted network, and
/// several PoolRelay replicas may register the same pool.
pub struct MemoryPoolStore {
    lease: Lease,
    started: Instant,
    pools: Arc<Mutex<Pools>>,
    /// Sent while holding the `pools` lock, so a subscriber that snapshots
    /// and subscribes under it misses no event and sees none twice.
    events: broadcast::Sender<PoolEvent>,
    sweeping: OnceLock<()>,
    stop: CancellationToken,
}

#[derive(Default)]
struct Pools {
    registrations: HashMap<PoolKey, Registered>,
}

struct Registered {
    /// As the PoolRelay sent it, with its lists sorted.
    registration: PoolRegistration,
    expires_at: Instant,
}

impl Pools {
    fn snapshot(&self, now: Instant, completeness: Completeness) -> Snapshot<PoolListing> {
        Snapshot {
            items: self
                .registrations
                .values()
                .filter(|registered| registered.expires_at > now)
                .map(|registered| listing(&registered.registration))
                .collect(),
            completeness,
        }
    }
}

impl MemoryPoolStore {
    /// The service bounds the lease, since the store adds it to the current
    /// time.
    pub fn new(lease: Lease) -> Self {
        Self {
            lease,
            started: Instant::now(),
            pools: Arc::default(),
            events: broadcast::channel(EVENT_BUFFER).0,
            sweeping: OnceLock::new(),
            stop: CancellationToken::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Pools> {
        self.pools.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Every trait method runs in a Tokio runtime, so the spawn is safe.
    fn ensure_sweeping(&self) {
        self.sweeping.get_or_init(|| {
            tokio::spawn(sweep(
                Arc::clone(&self.pools),
                self.events.clone(),
                self.lease.heartbeat_interval(),
                self.started.checked_add(self.lease.lease_duration()),
                self.stop.clone(),
            ));
        });
    }
}

impl fmt::Debug for MemoryPoolStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryPoolStore")
            .field("lease", &self.lease)
            .finish_non_exhaustive()
    }
}

impl Drop for MemoryPoolStore {
    fn drop(&mut self) {
        self.stop.cancel();
    }
}

/// `Exact` once a lease has passed since the store started.
fn completeness(started: Instant, lease: Lease, now: Instant) -> Completeness {
    if now.saturating_duration_since(started) >= lease.lease_duration() {
        Completeness::Exact
    } else {
        Completeness::Estimated
    }
}

/// Removes expired pools and publishes `Removed` for each. At `complete_at`,
/// once a lease has passed since the store started, publishes a `Resync`, so
/// watchers learn that its lists are complete.
async fn sweep(
    pools: Arc<Mutex<Pools>>,
    events: broadcast::Sender<PoolEvent>,
    every: Duration,
    mut complete_at: Option<Instant>,
    stop: CancellationToken,
) {
    let mut ticks = tokio::time::interval(every);
    ticks.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            biased;
            () = stop.cancelled() => return,
            () = wake_at(complete_at) => {
                complete_at = None;
                let pools = pools.lock().unwrap_or_else(PoisonError::into_inner);
                let complete = pools.snapshot(Instant::now(), Completeness::Exact);
                // No subscribers is not an error.
                let _ = events.send(WatchEvent::Resync(complete));
                continue;
            }
            _ = ticks.tick() => {}
        }
        let now = Instant::now();
        let mut pools = pools.lock().unwrap_or_else(PoisonError::into_inner);
        pools.registrations.retain(|key, registered| {
            let live = registered.expires_at > now;
            if !live {
                // No subscribers is not an error.
                let _ = events.send(WatchEvent::Removed(key.clone()));
            }
            live
        });
    }
}

/// Sorts a registration's lists and drops repeats, so the same lists in
/// another order are not a change.
fn sorted(mut registration: PoolRegistration) -> PoolRegistration {
    registration.capabilities.sort_unstable();
    registration.capabilities.dedup();
    registration.frontends.sort_unstable();
    registration.frontends.dedup();
    registration.capacity.sort_unstable();
    registration.capacity.dedup();
    registration
}

/// A registration as listed. The store does not know the PoolRelay's
/// addresses.
fn listing(registration: &PoolRegistration) -> PoolListing {
    PoolListing {
        key: registration.key.clone(),
        location: registration.location.clone(),
        model: Some(registration.model.clone()),
        capabilities: registration.capabilities.clone(),
        frontends: registration.frontends.clone(),
        capacity: registration.capacity.clone(),
        relay_identities: Vec::new(),
    }
}

#[async_trait]
impl PoolRegistrar for MemoryPoolStore {
    /// An unchanged pool only renews its lease. A new or changed one replaces
    /// the old listing, with an `Added` event. Lists in another order, for
    /// example from another PoolRelay replica, are not a change.
    async fn register(&self, registration: PoolRegistration) -> Result<Lease, DiscoveryError> {
        let registration = sorted(registration);
        self.ensure_sweeping();
        let now = Instant::now();
        let expires_at = now + self.lease.lease_duration();
        let mut pools = self.lock();
        if let Some(registered) = pools.registrations.get_mut(&registration.key)
            && registered.expires_at > now
            && registered.registration == registration
        {
            registered.expires_at = expires_at;
            return Ok(self.lease);
        }
        let added = listing(&registration);
        pools.registrations.insert(
            registration.key.clone(),
            Registered {
                registration,
                expires_at,
            },
        );
        // No subscribers is not an error.
        let _ = self.events.send(WatchEvent::Added(added));
        Ok(self.lease)
    }
}

#[async_trait]
impl Directory<PoolListing, PoolKey> for MemoryPoolStore {
    async fn list(&self) -> Result<Snapshot<PoolListing>, DiscoveryError> {
        let now = Instant::now();
        let completeness = completeness(self.started, self.lease, now);
        Ok(self.lock().snapshot(now, completeness))
    }

    async fn list_and_watch(
        &self,
        cancel_token: Option<CancellationToken>,
    ) -> Result<WatchStream<PoolListing, PoolKey>, DiscoveryError> {
        self.ensure_sweeping();
        let pools = Arc::clone(&self.pools);
        let events = self.events.clone();
        let (started, lease) = (self.started, self.lease);
        Ok(watch_stream(
            move || {
                let pools = pools.lock().unwrap_or_else(PoisonError::into_inner);
                let now = Instant::now();
                let completeness = completeness(started, lease, now);
                (events.subscribe(), Some(pools.snapshot(now, completeness)))
            },
            cancel_token.unwrap_or_default(),
            self.stop.clone(),
        ))
    }
}

/// Router replicas from configuration. The set never changes, and lists are
/// `Exact`.
#[derive(Debug)]
pub struct StaticReplicaDirectory {
    replicas: Vec<ReplicaEndpoint>,
    /// Never sent on. Keeps watch streams open until the directory is
    /// dropped.
    events: broadcast::Sender<ReplicaEvent>,
    stop: CancellationToken,
}

impl StaticReplicaDirectory {
    pub fn new(replicas: Vec<ReplicaEndpoint>) -> Self {
        Self {
            replicas,
            events: broadcast::channel(1).0,
            stop: CancellationToken::new(),
        }
    }

    fn snapshot(&self) -> Snapshot<ReplicaEndpoint> {
        Snapshot {
            items: self.replicas.clone(),
            completeness: Completeness::Exact,
        }
    }
}

impl Drop for StaticReplicaDirectory {
    fn drop(&mut self) {
        self.stop.cancel();
    }
}

#[async_trait]
impl Directory<ReplicaEndpoint, ReplicaId> for StaticReplicaDirectory {
    async fn list(&self) -> Result<Snapshot<ReplicaEndpoint>, DiscoveryError> {
        Ok(self.snapshot())
    }

    async fn list_and_watch(
        &self,
        cancel_token: Option<CancellationToken>,
    ) -> Result<WatchStream<ReplicaEndpoint, ReplicaId>, DiscoveryError> {
        let events = self.events.clone();
        let snapshot = self.snapshot();
        Ok(watch_stream(
            move || (events.subscribe(), Some(snapshot.clone())),
            cancel_token.unwrap_or_default(),
            self.stop.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::pool_discovery::Capacity;
    use dynamo_global_common::PoolLocation;
    use futures::StreamExt;
    use url::Url;

    use super::*;

    fn lease() -> Lease {
        Lease::new(5, 15).unwrap()
    }

    fn key(dgd: &str) -> PoolKey {
        PoolKey::new("west", "dynamo", dgd).unwrap()
    }

    fn capacity(sku: &str, count: u32) -> Capacity {
        Capacity {
            sku: sku.to_owned(),
            count,
        }
    }

    fn registration(dgd: &str, model: &str) -> PoolRegistration {
        PoolRegistration {
            key: key(dgd),
            location: PoolLocation {
                region: "us-west-2".to_owned(),
                availability_zone: None,
                cluster: Some("west".to_owned()),
                datacenter: None,
            },
            model: model.to_owned(),
            capabilities: vec!["prefill".to_owned(), "decode".to_owned()],
            frontends: vec![
                Url::parse("http://10.43.1.3:8000/").unwrap(),
                Url::parse("http://10.43.1.2:8000/").unwrap(),
            ],
            capacity: vec![capacity("H100", 8), capacity("B200", 4)],
        }
    }

    async fn find(store: &MemoryPoolStore, dgd: &str) -> Option<PoolListing> {
        let snapshot = store.list().await.unwrap();
        snapshot
            .items
            .into_iter()
            .find(|listing| listing.key == key(dgd))
    }

    #[tokio::test(start_paused = true)]
    async fn registering_again_only_renews() {
        let store = MemoryPoolStore::new(lease());
        assert_eq!(
            store.register(registration("mocker", "m1")).await.unwrap(),
            lease()
        );
        let mut watch = store.list_and_watch(None).await.unwrap();
        let Some(Ok(WatchEvent::Added(first))) = watch.next().await else {
            panic!("expected the pool");
        };
        assert_eq!(first.capabilities, ["decode", "prefill"]);
        assert!(matches!(
            watch.next().await,
            Some(Ok(WatchEvent::Resync(_)))
        ));

        // Heartbeats every 10 s keep the 15 s lease alive past 15 s. Lists in
        // another order, or with an item twice, are no change.
        for _ in 0..3 {
            tokio::time::advance(Duration::from_secs(10)).await;
            let mut heartbeat = registration("mocker", "m1");
            heartbeat.capabilities.reverse();
            heartbeat.frontends.reverse();
            heartbeat.capacity.push(capacity("H100", 8));
            store.register(heartbeat).await.unwrap();
        }
        assert_eq!(find(&store, "mocker").await, Some(first));
        // A lease has passed, so the store's lists are complete; the renewals
        // published nothing.
        assert!(
            matches!(watch.next().await, Some(Ok(WatchEvent::Resync(s))) if s.completeness == Completeness::Exact)
        );

        // A changed pool is published.
        store.register(registration("mocker", "m2")).await.unwrap();
        let Some(Ok(WatchEvent::Added(changed))) = watch.next().await else {
            panic!("expected the change");
        };
        assert_eq!(changed.model.as_deref(), Some("m2"));
    }

    #[tokio::test(start_paused = true)]
    async fn a_missed_lease_hides_then_removes_the_pool() {
        let store = MemoryPoolStore::new(lease());
        store.register(registration("mocker", "m1")).await.unwrap();
        let mut watch = store.list_and_watch(None).await.unwrap();
        assert!(matches!(watch.next().await, Some(Ok(WatchEvent::Added(_)))));
        assert!(matches!(
            watch.next().await,
            Some(Ok(WatchEvent::Resync(_)))
        ));

        tokio::time::advance(Duration::from_secs(16)).await;
        // Reads skip it before the sweep runs.
        assert!(find(&store, "mocker").await.is_none());
        // A lease has passed, so the store's lists are complete. Then the
        // sweep removes the pool and tells watchers.
        assert!(matches!(
            watch.next().await,
            Some(Ok(WatchEvent::Resync(_)))
        ));
        assert!(
            matches!(watch.next().await, Some(Ok(WatchEvent::Removed(k))) if k == key("mocker"))
        );
        assert!(store.lock().registrations.is_empty());

        // Registering again brings it back.
        store.register(registration("mocker", "m1")).await.unwrap();
        assert!(
            matches!(watch.next().await, Some(Ok(WatchEvent::Added(listing))) if listing.key == key("mocker"))
        );
    }

    #[tokio::test(start_paused = true)]
    async fn lists_are_exact_once_a_lease_has_passed() {
        let store = MemoryPoolStore::new(lease());
        let completeness = || async { store.list().await.unwrap().completeness };
        let mut watch = store.list_and_watch(None).await.unwrap();
        let Some(Ok(WatchEvent::Resync(first))) = watch.next().await else {
            panic!("expected the first Resync");
        };
        assert_eq!(first.completeness, Completeness::Estimated);
        assert_eq!(completeness().await, Completeness::Estimated);

        tokio::time::advance(lease().lease_duration()).await;
        assert_eq!(completeness().await, Completeness::Exact);
        // Watchers learn it too. With no pools nothing else is published, so
        // the wait is bounded: a missing Resync fails the test, not hangs it.
        let next = tokio::time::timeout(Duration::from_secs(1), watch.next()).await;
        let Ok(Some(Ok(WatchEvent::Resync(complete)))) = next else {
            panic!("expected a Resync");
        };
        assert_eq!(complete.completeness, Completeness::Exact);
    }

    #[tokio::test]
    async fn static_replicas_list_and_watch() {
        let replica = ReplicaEndpoint {
            replica: ReplicaId("ohio/global-router-0".to_owned()),
            url: Url::parse("http://10.42.0.5:50051").unwrap(),
        };
        let directory = StaticReplicaDirectory::new(vec![replica.clone()]);
        let snapshot = directory.list().await.unwrap();
        assert_eq!(snapshot.items, std::slice::from_ref(&replica));
        assert_eq!(snapshot.completeness, Completeness::Exact);

        let cancel = CancellationToken::new();
        let mut watch = directory
            .list_and_watch(Some(cancel.clone()))
            .await
            .unwrap();
        assert!(matches!(watch.next().await, Some(Ok(WatchEvent::Added(r))) if r == replica));
        assert!(matches!(
            watch.next().await,
            Some(Ok(WatchEvent::Resync(_)))
        ));
        cancel.cancel();
        assert!(watch.next().await.is_none());
    }
}
