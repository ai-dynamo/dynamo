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
    Completeness, Directory, DiscoveryError, ExchangeRound, Lease, PoolEvent, PoolListing,
    PoolRegistrar, PoolRegistration, RegistrationExchange, RouterReplica, RouterReplicaEvent,
    RouterReplicaId, Snapshot, WatchEvent, WatchStream, answer_timeout,
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

struct Pools {
    registrations: HashMap<PoolKey, Registered>,
    /// `Some` once this instance shares registrations with the others.
    exchange: Option<Exchange>,
    /// The completeness watchers were last told.
    announced: Completeness,
}

struct Registered {
    /// As the PoolRelay sent it, with its lists sorted.
    registration: PoolRegistration,
    expires_at: Instant,
    /// The latest registration came from a PoolRelay, not from another
    /// service instance. Only these are shared.
    local: bool,
}

impl Registered {
    /// When the pool's registration may have lapsed everywhere: at once for
    /// one made here, but a copy lapses up to [`answer_timeout`], the
    /// longest the other instance had to answer, before its original.
    fn lapsed_at(&self, lease: Lease) -> Instant {
        if self.local {
            self.expires_at
        } else {
            self.expires_at + answer_timeout(lease)
        }
    }
}

/// What this instance knows about the registrations the other instances
/// received.
#[derive(Default)]
struct Exchange {
    round: ExchangeRound,
    /// The latest time a registration lapsed here, of those the sweep removed.
    lapsed: Option<Instant>,
}

impl Exchange {
    /// No registration made with another instance can be missing: the last
    /// round vouches for now, and every instance has been asked since the
    /// latest lapse here, including the ones the sweep has not removed yet.
    fn complete(
        &self,
        registrations: &HashMap<PoolKey, Registered>,
        lease: Lease,
        now: Instant,
    ) -> bool {
        let lapsed = registrations
            .values()
            .filter(|registered| registered.expires_at <= now)
            .map(|registered| registered.lapsed_at(lease))
            .max()
            .max(self.lapsed);
        self.round.complete_until.is_some_and(|until| now < until)
            && lapsed.is_none_or(|lapsed| {
                self.round
                    .answered_since
                    .is_some_and(|since| since >= lapsed)
            })
    }
}

impl Pools {
    fn new() -> Self {
        Self {
            registrations: HashMap::new(),
            exchange: None,
            announced: Completeness::Estimated,
        }
    }

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

    /// `Exact` once a lease has passed since the store started, unless
    /// registrations made with another instance may be missing.
    fn completeness(&self, started: Instant, lease: Lease, now: Instant) -> Completeness {
        let copies_complete = self
            .exchange
            .as_ref()
            .is_none_or(|exchange| exchange.complete(&self.registrations, lease, now));
        if copies_complete && now.saturating_duration_since(started) >= lease.duration() {
            Completeness::Exact
        } else {
            Completeness::Estimated
        }
    }

    /// Sends watchers a `Resync` when the lists' completeness differs from
    /// what they were last told.
    fn announce(
        &mut self,
        started: Instant,
        lease: Lease,
        now: Instant,
        events: &broadcast::Sender<PoolEvent>,
    ) {
        let completeness = self.completeness(started, lease, now);
        if completeness != self.announced {
            self.announced = completeness;
            // No subscribers is not an error.
            let _ = events.send(WatchEvent::Resync(self.snapshot(now, completeness)));
        }
    }

    /// Keeps `registration` until `expires_at`. A live, unchanged pool only
    /// gets the new expiry. A new or changed one replaces the listing, with an
    /// `Upserted` event, and so does one whose lease ran out before the sweep
    /// removed it, since a `Resync` or a new watcher's first view may have
    /// left it out meanwhile.
    fn put(
        &mut self,
        events: &broadcast::Sender<PoolEvent>,
        registration: PoolRegistration,
        expires_at: Instant,
        local: bool,
        now: Instant,
    ) {
        if let Some(registered) = self.registrations.get_mut(&registration.key)
            && registered.expires_at > now
            && registered.registration == registration
        {
            registered.expires_at = expires_at;
            registered.local = local;
            return;
        }
        let added = listing(&registration);
        self.registrations.insert(
            registration.key.clone(),
            Registered {
                registration,
                expires_at,
                local,
            },
        );
        // No subscribers is not an error.
        let _ = events.send(WatchEvent::Upserted(added));
    }

    /// Removes the pools whose leases have passed, and returns their keys.
    /// Notes when the latest of them may have lapsed everywhere.
    fn remove_lapsed(&mut self, lease: Lease, now: Instant) -> Vec<PoolKey> {
        let mut removed = Vec::new();
        let mut lapsed = None;
        self.registrations.retain(|key, registered| {
            let live = registered.expires_at > now;
            if !live {
                lapsed = lapsed.max(Some(registered.lapsed_at(lease)));
                removed.push(key.clone());
            }
            live
        });
        if let Some(exchange) = &mut self.exchange {
            exchange.lapsed = exchange.lapsed.max(lapsed);
        }
        removed
    }
}

impl MemoryPoolStore {
    /// The service bounds the lease, since the store adds it to the current
    /// time.
    pub fn new(lease: Lease) -> Self {
        Self {
            lease,
            started: Instant::now(),
            pools: Arc::new(Mutex::new(Pools::new())),
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
                self.started,
                self.lease,
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
        if pools
            .registrations
            .get(&registration.key)
            .is_some_and(|registered| registered.expires_at >= expires_at)
        {
            return;
        }
        pools.put(&self.events, registration, expires_at, false, now);
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

    /// Watchers get a `Resync` when this changes whether lists are `Exact`.
    fn end_round(&self, round: ExchangeRound) {
        self.ensure_sweeping();
        let now = Instant::now();
        let mut pools = self.lock();
        pools.exchange.get_or_insert_default().round = round;
        pools.announce(self.started, self.lease, now, &self.events);
    }

    fn holds_every_local_registration(&self) -> bool {
        Instant::now().saturating_duration_since(self.started) >= self.lease.duration()
    }
}

/// Removes expired pools and publishes `Removed` for each, after a `Resync`
/// if the lists' completeness changed, so that watchers learn that a pool
/// may have been renewed with another instance before it goes. A lease after
/// `started`, publishes a `Resync`, so watchers learn that the lists are
/// complete, unless copies may be missing.
async fn sweep(
    pools: Arc<Mutex<Pools>>,
    events: broadcast::Sender<PoolEvent>,
    every: Duration,
    started: Instant,
    lease: Lease,
    stop: CancellationToken,
) {
    let mut complete_at = started.checked_add(lease.duration());
    let mut ticks = tokio::time::interval(every);
    ticks.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            biased;
            () = stop.cancelled() => return,
            () = wake_at(complete_at) => {
                complete_at = None;
                let mut pools = pools.lock().unwrap_or_else(PoisonError::into_inner);
                pools.announce(started, lease, Instant::now(), &events);
                continue;
            }
            _ = ticks.tick() => {}
        }
        let now = Instant::now();
        let mut pools = pools.lock().unwrap_or_else(PoisonError::into_inner);
        let removed = pools.remove_lapsed(lease, now);
        // Also announces the end of the last round's `complete_until`.
        pools.announce(started, lease, now, &events);
        for key in removed {
            // No subscribers is not an error.
            let _ = events.send(WatchEvent::Removed(key));
        }
    }
}

/// Sorts a registration's lists and drops repeats, so the same lists in
/// another order are not a change. The DynamoDB backend stores registrations
/// this way too.
pub(super) fn sorted(mut registration: PoolRegistration) -> PoolRegistration {
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
pub(super) fn listing(registration: &PoolRegistration) -> PoolListing {
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
        self.lock()
            .put(&self.events, registration, expires_at, true, now);
        Ok(self.lease)
    }
}

#[async_trait]
impl Directory<PoolListing, PoolKey> for MemoryPoolStore {
    async fn list(&self) -> Result<Snapshot<PoolListing>, DiscoveryError> {
        let now = Instant::now();
        let pools = self.lock();
        let completeness = pools.completeness(self.started, self.lease, now);
        Ok(pools.snapshot(now, completeness))
    }

    /// Starts from the view `list` returns, then follows registrations and
    /// expired leases, and sends a `Resync` whenever the lists' completeness
    /// changes.
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
                let mut pools = pools.lock().unwrap_or_else(PoisonError::into_inner);
                let now = Instant::now();
                // So that the new watcher hears of the next change, as the
                // others do.
                pools.announce(started, lease, now, &events);
                let completeness = pools.announced;
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

    /// The next event, which must come within a second, so a missing one
    /// fails the test rather than hanging it.
    async fn next_event(watch: &mut WatchStream<PoolListing, PoolKey>) -> PoolEvent {
        let next = tokio::time::timeout(Duration::from_secs(1), watch.next()).await;
        let Ok(Some(Ok(event))) = next else {
            panic!("expected an event within a second");
        };
        event
    }

    /// The completeness of the next event, which must be a `Resync`.
    async fn next_resync(watch: &mut WatchStream<PoolListing, PoolKey>) -> Completeness {
        let WatchEvent::Resync(snapshot) = next_event(watch).await else {
            panic!("expected a Resync");
        };
        snapshot.completeness
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
        assert_eq!(next_resync(&mut watch).await, Completeness::Exact);
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
    async fn lists_are_estimated_until_a_round_says_copies_are_complete_and_after_that() {
        let store = MemoryPoolStore::new(lease());
        let completeness = || async { store.list().await.unwrap().completeness };
        let mut watch = store.list_and_watch(None).await.unwrap();
        assert_eq!(next_resync(&mut watch).await, Completeness::Estimated);
        // Before the first round ends, registrations made with another
        // instance may be missing, so the first lease passes without the
        // lists turning `Exact`, and the sweep announces nothing.
        store.end_round(ExchangeRound::default());
        tokio::time::advance(lease().duration()).await;
        tokio::time::sleep(Duration::from_millis(1)).await;
        assert!(watch.next().now_or_never().is_none());
        assert_eq!(completeness().await, Completeness::Estimated);

        // A round vouches for the next 3 s, and watchers learn it.
        let now = Instant::now();
        store.end_round(ExchangeRound {
            answered_since: Some(now),
            complete_until: Some(now + Duration::from_secs(3)),
        });
        assert_eq!(next_resync(&mut watch).await, Completeness::Exact);
        assert_eq!(completeness().await, Completeness::Exact);

        // No other round ends, as when the task that asks the other instances
        // has stopped: once that time passes, the lists are `Estimated`, and
        // the sweep tells watchers.
        tokio::time::advance(Duration::from_secs(3)).await;
        assert_eq!(completeness().await, Completeness::Estimated);
        assert_eq!(next_resync(&mut watch).await, Completeness::Estimated);
    }

    #[tokio::test(start_paused = true)]
    async fn a_registration_that_lapses_keeps_lists_estimated_until_every_instance_is_asked_again()
    {
        let store = MemoryPoolStore::new(lease());
        // Rounds in which every instance answered in full when asked at
        // `answered_since`, vouching for an hour.
        let round = |answered_since: Instant| ExchangeRound {
            answered_since: Some(answered_since),
            complete_until: Some(answered_since + Duration::from_secs(3600)),
        };
        store.end_round(ExchangeRound::default());
        tokio::time::advance(lease().duration()).await;
        // The sweep takes the end of the first lease now, not with the lapse.
        tokio::time::sleep(Duration::from_millis(1)).await;
        // A copy with 5 s left, and a registration made here, which lasts a
        // lease.
        store.adopt(registration("copied", "m1"), Duration::from_secs(5));
        let _ = store.register(registration("local", "m1")).await.unwrap();
        let asked = Instant::now();
        store.end_round(round(asked));
        let mut watch = store.list_and_watch(None).await.unwrap();
        for _ in 0..2 {
            assert!(matches!(
                next_event(&mut watch).await,
                WatchEvent::Upserted(_)
            ));
        }
        assert_eq!(next_resync(&mut watch).await, Completeness::Exact);

        // The copy lapses before the instances are asked again, but its
        // PoolRelay may have renewed it late since, with any of them. The
        // pool is not listed, and neither are the lists complete. Watchers
        // learn that before the pool goes.
        tokio::time::advance(Duration::from_secs(5)).await;
        let listed = store.list().await.unwrap();
        assert!(find(&store, "copied").await.is_none());
        assert_eq!(listed.completeness, Completeness::Estimated);
        assert_eq!(next_resync(&mut watch).await, Completeness::Estimated);
        assert!(
            matches!(next_event(&mut watch).await, WatchEvent::Removed(k) if k == key("copied"))
        );

        // Answers to requests sent before the lapse settle nothing, and nor
        // do ones sent within the answer timeout after it: a copy lapses up
        // to that much before its original, the longest an instance takes to
        // answer.
        for sent in [asked + Duration::from_secs(4), Instant::now()] {
            store.end_round(round(sent));
            assert_eq!(
                store.list().await.unwrap().completeness,
                Completeness::Estimated
            );
        }
        // A later round brings the renewed pool back, and since every
        // instance was asked after its original may have lapsed, the lists
        // are complete again.
        tokio::time::advance(answer_timeout(lease())).await;
        store.adopt(registration("copied", "m1"), Duration::from_secs(15));
        store.end_round(round(Instant::now()));
        assert!(
            matches!(next_event(&mut watch).await, WatchEvent::Upserted(listing) if listing.key == key("copied"))
        );
        assert_eq!(next_resync(&mut watch).await, Completeness::Exact);

        // So does a registration made here: its PoolRelay may have renewed it
        // late with another instance. Its lapse is known exactly, so the
        // next round settles it.
        tokio::time::advance(Duration::from_millis(7500)).await;
        assert!(find(&store, "local").await.is_none());
        assert_eq!(next_resync(&mut watch).await, Completeness::Estimated);
        store.end_round(round(Instant::now()));
        assert_eq!(
            store.list().await.unwrap().completeness,
            Completeness::Exact
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_watcher_hears_of_a_renewal_after_the_lease_ran_out() {
        // A round vouches for every copy for an hour, from when the store
        // started. The sweep looks every second, from then.
        let store = MemoryPoolStore::new(lease());
        let start = Instant::now();
        store.end_round(ExchangeRound {
            answered_since: Some(start),
            complete_until: Some(start + Duration::from_secs(3600)),
        });
        tokio::time::sleep_until(start + Duration::from_millis(500)).await;
        let _ = store.register(registration("mocker", "m1")).await.unwrap();
        // The lease runs out at 15.5 s, and the next sweep is at 16 s. A
        // watcher that starts meanwhile does not see the pool, and sees an
        // incomplete list, since the pool may have been renewed with another
        // instance.
        tokio::time::sleep_until(start + Duration::from_millis(15_600)).await;
        let mut watch = store.list_and_watch(None).await.unwrap();
        assert_eq!(next_resync(&mut watch).await, Completeness::Estimated);
        // The PoolRelay renews it late, here: the watcher hears of the pool
        // again, and that the list is complete, at the next sweep.
        let _ = store.register(registration("mocker", "m1")).await.unwrap();
        assert!(
            matches!(next_event(&mut watch).await, WatchEvent::Upserted(listing) if listing.key == key("mocker"))
        );
        assert_eq!(next_resync(&mut watch).await, Completeness::Exact);
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
