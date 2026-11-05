// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The memory backend: one process keeps every registration. It needs no
//! Kubernetes, so it suits tests and a first deployment. The MCS backend keeps
//! its registrations here too, with copies of the ones other service
//! instances received (see `crate::peers`). A restarted process starts empty
//! and fills as PoolRelays register again, so its lists are `Estimated` until
//! one lease has passed: by then every live pool has registered again. They
//! are `Estimated` too while copies from another instance may be missing. Router
//! replicas come from configuration until routers register too. Used alone,
//! run one replica: each keeps only the registrations it receives.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::Duration;

use crate::pool_discovery::{
    Completeness, Directory, DiscoveryError, Lease, PoolEvent, PoolListing, PoolRegistrar,
    PoolRegistration, RegistrationExchange, RouterReplica, RouterReplicaEvent, RouterReplicaId,
    Snapshot, WatchEvent, WatchStream,
};
use async_trait::async_trait;
use dynamo_global_common::PoolKey;
use tokio::sync::broadcast;
use tokio::time::{Instant, MissedTickBehavior};
use tokio_util::sync::CancellationToken;

use super::stream::{EVENT_BUFFER, wake_at, watch_stream};

/// The longest an expired registration waits for the sweep. Watchers, such
/// as the MCS backend's join, drop a pool only when the sweep removes it.
pub const SWEEP_EVERY: Duration = Duration::from_secs(1);

/// Pool registrations in memory. A pool expires the lease duration after its
/// last registration. Reads skip an expired pool at once; a sweep removes it,
/// with a `Removed` event, within [`SWEEP_EVERY`] or one heartbeat interval,
/// whichever is shorter.
///
/// Any caller may register any pool: V1 assumes a trusted network, and
/// several PoolRelay replicas may register the same pool. The latest
/// registration is the pool's listing, so replicas should send the same one:
/// replicas that disagree replace each other's listing until they agree.
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
    /// Registrations made with another service instance may be missing
    /// (see [`RegistrationExchange::set_copies_missing`]).
    copies_missing: bool,
}

struct Registered {
    /// As the PoolRelay sent it, with its lists sorted.
    registration: PoolRegistration,
    expires_at: Instant,
    /// The latest registration came from a PoolRelay, not from another
    /// service instance. Only these are shared.
    local: bool,
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
                self.lease.heartbeat_interval().min(SWEEP_EVERY),
                self.started.checked_add(self.lease.duration()),
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

impl RegistrationExchange for MemoryPoolStore {
    /// Keeps a registration copied from another service instance (see
    /// `crate::peers`) for the `remaining` time it had there, capped at this
    /// store's lease. The latest registration wins, wherever it was made. A
    /// copy with no time left is ignored, so it cannot list a lapsed pool
    /// again until the next sweep.
    fn adopt(&self, registration: PoolRegistration, remaining: Duration) {
        if remaining.is_zero() {
            return;
        }
        let registration = sorted(registration);
        self.ensure_sweeping();
        let now = Instant::now();
        let expires_at = now + remaining.min(self.lease.duration());
        let mut pools = self.lock();
        if let Some(registered) = pools.registrations.get_mut(&registration.key)
            && registered.expires_at > now
        {
            if registered.expires_at >= expires_at {
                return;
            }
            registered.local = false;
            if registered.registration == registration {
                registered.expires_at = expires_at;
                return;
            }
        }
        let added = listing(&registration);
        pools.registrations.insert(
            registration.key.clone(),
            Registered {
                registration,
                expires_at,
                local: false,
            },
        );
        // No subscribers is not an error.
        let _ = self.events.send(WatchEvent::Upserted(added));
    }

    /// The live registrations PoolRelays made here, with the time each has
    /// left. Copies are left out, so each registration is shared only by the
    /// instance that received it.
    fn local_registrations(&self) -> Vec<(PoolRegistration, Duration)> {
        let now = Instant::now();
        self.lock()
            .registrations
            .values()
            .filter(|registered| registered.local && registered.expires_at > now)
            .map(|registered| (registered.registration.clone(), registered.expires_at - now))
            .collect()
    }

    fn holds_every_local_registration(&self) -> bool {
        Instant::now().saturating_duration_since(self.started) >= self.lease.duration()
    }

    /// Watchers get a `Resync` when this changes whether lists are `Exact`.
    fn set_copies_missing(&self, missing: bool) {
        let now = Instant::now();
        let mut pools = self.lock();
        let before = completeness(self.started, self.lease, now, pools.copies_missing);
        pools.copies_missing = missing;
        let after = completeness(self.started, self.lease, now, missing);
        if after != before {
            // No subscribers is not an error.
            let _ = self
                .events
                .send(WatchEvent::Resync(pools.snapshot(now, after)));
        }
    }
}

/// `Exact` once a lease has passed since the store started, unless
/// registrations made with another instance may be missing.
fn completeness(
    started: Instant,
    lease: Lease,
    now: Instant,
    copies_missing: bool,
) -> Completeness {
    if !copies_missing && now.saturating_duration_since(started) >= lease.duration() {
        Completeness::Exact
    } else {
        Completeness::Estimated
    }
}

/// Removes expired pools and publishes `Removed` for each. At `complete_at`,
/// once a lease has passed since the store started, publishes a `Resync`, so
/// watchers learn that its lists are complete, unless copies may be missing.
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
                // Otherwise the lists stay `Estimated`, and
                // `set_copies_missing` announces the change.
                if !pools.copies_missing {
                    let complete = pools.snapshot(Instant::now(), Completeness::Exact);
                    // No subscribers is not an error.
                    let _ = events.send(WatchEvent::Resync(complete));
                }
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
    /// the old listing, with an `Upserted` event. Lists in another order, for
    /// example from another PoolRelay replica, are not a change.
    async fn register(&self, registration: PoolRegistration) -> Result<Lease, DiscoveryError> {
        let registration = sorted(registration);
        self.ensure_sweeping();
        let now = Instant::now();
        let expires_at = now + self.lease.duration();
        let mut pools = self.lock();
        if let Some(registered) = pools.registrations.get_mut(&registration.key)
            && registered.expires_at > now
            && registered.registration == registration
        {
            registered.expires_at = expires_at;
            registered.local = true;
            return Ok(self.lease);
        }
        let added = listing(&registration);
        pools.registrations.insert(
            registration.key.clone(),
            Registered {
                registration,
                expires_at,
                local: true,
            },
        );
        // No subscribers is not an error.
        let _ = self.events.send(WatchEvent::Upserted(added));
        Ok(self.lease)
    }
}

#[async_trait]
impl Directory<PoolListing, PoolKey> for MemoryPoolStore {
    async fn list(&self) -> Result<Snapshot<PoolListing>, DiscoveryError> {
        let now = Instant::now();
        let pools = self.lock();
        let completeness = completeness(self.started, self.lease, now, pools.copies_missing);
        Ok(pools.snapshot(now, completeness))
    }

    /// Starts from the view `list` returns, then follows registrations and
    /// expired leases, and sends a `Resync` when the lists become `Exact`.
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
                let completeness = completeness(started, lease, now, pools.copies_missing);
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
    replicas: Vec<RouterReplica>,
    /// Never sent on. Keeps watch streams open until the directory is
    /// dropped.
    events: broadcast::Sender<RouterReplicaEvent>,
    stop: CancellationToken,
}

impl StaticReplicaDirectory {
    pub fn new(replicas: Vec<RouterReplica>) -> Self {
        Self {
            replicas,
            events: broadcast::channel(1).0,
            stop: CancellationToken::new(),
        }
    }

    fn snapshot(&self) -> Snapshot<RouterReplica> {
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
impl Directory<RouterReplica, RouterReplicaId> for StaticReplicaDirectory {
    async fn list(&self) -> Result<Snapshot<RouterReplica>, DiscoveryError> {
        Ok(self.snapshot())
    }

    /// The configured replicas. The stream then stays open, with no events,
    /// until the caller cancels it or the directory is dropped.
    async fn list_and_watch(
        &self,
        cancel_token: Option<CancellationToken>,
    ) -> Result<WatchStream<RouterReplica, RouterReplicaId>, DiscoveryError> {
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
    use futures::{FutureExt, StreamExt};
    use url::Url;

    use super::*;

    fn lease() -> Lease {
        Lease::new(5, 15).unwrap()
    }

    fn key(dgd: &str) -> PoolKey {
        PoolKey::new("west", "dynamo", dgd).unwrap()
    }

    fn capacity(sku: &str, workers: u32) -> Capacity {
        Capacity {
            sku: sku.to_owned(),
            workers,
        }
    }

    fn registration(dgd: &str, model: &str) -> PoolRegistration {
        PoolRegistration {
            key: key(dgd),
            location: PoolLocation {
                region: "us-west-2".to_owned(),
                availability_zone: None,
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
        let Some(Ok(WatchEvent::Upserted(first))) = watch.next().await else {
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
            let _ = store.register(heartbeat).await.unwrap();
        }
        assert_eq!(find(&store, "mocker").await, Some(first));
        // A lease has passed, so the store's lists are complete; the renewals
        // published nothing.
        assert!(
            matches!(watch.next().await, Some(Ok(WatchEvent::Resync(s))) if s.completeness == Completeness::Exact)
        );

        // A changed pool is published.
        let _ = store.register(registration("mocker", "m2")).await.unwrap();
        let Some(Ok(WatchEvent::Upserted(changed))) = watch.next().await else {
            panic!("expected the change");
        };
        assert_eq!(changed.model.as_deref(), Some("m2"));
    }

    #[tokio::test(start_paused = true)]
    async fn a_missed_lease_hides_then_removes_the_pool() {
        let store = MemoryPoolStore::new(lease());
        let _ = store.register(registration("mocker", "m1")).await.unwrap();
        let mut watch = store.list_and_watch(None).await.unwrap();
        assert!(matches!(
            watch.next().await,
            Some(Ok(WatchEvent::Upserted(_)))
        ));
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
        let _ = store.register(registration("mocker", "m1")).await.unwrap();
        assert!(
            matches!(watch.next().await, Some(Ok(WatchEvent::Upserted(listing))) if listing.key == key("mocker"))
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

        tokio::time::advance(lease().duration()).await;
        assert_eq!(completeness().await, Completeness::Exact);
        // Watchers learn it too. With no pools nothing else is published, so
        // the wait is bounded: a missing Resync fails the test, not hangs it.
        let next = tokio::time::timeout(Duration::from_secs(1), watch.next()).await;
        let Ok(Some(Ok(WatchEvent::Resync(complete)))) = next else {
            panic!("expected a Resync");
        };
        assert_eq!(complete.completeness, Completeness::Exact);
    }

    #[tokio::test(start_paused = true)]
    async fn copies_follow_the_latest_registration() {
        let store = MemoryPoolStore::new(lease());
        let model = |store: &MemoryPoolStore| {
            store
                .lock()
                .registrations
                .get(&key("mocker"))
                .map(|registered| registered.registration.model.clone())
        };
        let _ = store.register(registration("mocker", "m1")).await.unwrap();

        // A copy older than the local registration changes nothing.
        store.adopt(registration("mocker", "m2"), Duration::from_secs(10));
        assert_eq!(model(&store).as_deref(), Some("m1"));
        assert_eq!(store.local_registrations().len(), 1);

        // A newer copy takes over, so it is not shared again from here, until
        // a PoolRelay registers here again.
        tokio::time::advance(Duration::from_secs(5)).await;
        store.adopt(registration("mocker", "m1"), Duration::from_secs(14));
        assert!(store.local_registrations().is_empty());
        let _ = store.register(registration("mocker", "m1")).await.unwrap();
        assert_eq!(store.local_registrations().len(), 1);

        // A still newer copy with another model replaces it.
        tokio::time::advance(Duration::from_secs(1)).await;
        store.adopt(registration("mocker", "m2"), Duration::from_secs(15));
        assert_eq!(model(&store).as_deref(), Some("m2"));

        // A copy never outlasts this store's lease, and one with no time left
        // is ignored.
        store.adopt(registration("mocker", "m2"), Duration::from_secs(3600));
        tokio::time::advance(lease().duration() + Duration::from_secs(1)).await;
        assert!(find(&store, "mocker").await.is_none());
        store.adopt(registration("other", "m1"), Duration::ZERO);
        assert!(!store.lock().registrations.contains_key(&key("other")));
    }

    #[tokio::test(start_paused = true)]
    async fn lists_are_estimated_while_copies_may_be_missing() {
        let store = MemoryPoolStore::new(lease());
        let mut watch = store.list_and_watch(None).await.unwrap();
        assert!(matches!(
            watch.next().await,
            Some(Ok(WatchEvent::Resync(_)))
        ));
        // Another instance has not answered, so the first lease passes
        // without the lists turning `Exact`, and the sweep announces nothing.
        store.set_copies_missing(true);
        tokio::time::advance(lease().duration()).await;
        tokio::time::sleep(Duration::from_millis(1)).await;
        assert!(watch.next().now_or_never().is_none());
        assert_eq!(
            store.list().await.unwrap().completeness,
            Completeness::Estimated
        );

        // Watchers learn when it answers, and when it stops again.
        for (missing, expected) in [
            (false, Completeness::Exact),
            (true, Completeness::Estimated),
        ] {
            store.set_copies_missing(missing);
            let next = tokio::time::timeout(Duration::from_secs(1), watch.next()).await;
            let Ok(Some(Ok(WatchEvent::Resync(snapshot)))) = next else {
                panic!("expected a Resync");
            };
            assert_eq!(snapshot.completeness, expected);
            assert_eq!(store.list().await.unwrap().completeness, expected);
        }
    }

    #[tokio::test]
    async fn static_replicas_list_and_watch() {
        let replica = RouterReplica {
            replica: RouterReplicaId("ohio/global-router-0".to_owned()),
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
        assert!(matches!(watch.next().await, Some(Ok(WatchEvent::Upserted(r))) if r == replica));
        assert!(matches!(
            watch.next().await,
            Some(Ok(WatchEvent::Resync(_)))
        ));
        cancel.cancel();
        assert!(watch.next().await.is_none());
    }
}
