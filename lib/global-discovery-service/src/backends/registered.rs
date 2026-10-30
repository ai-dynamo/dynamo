// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The MCS backend with registrations. A pool is listed only while both
//! sources agree: MCS imports a Ready PoolRelay endpoint for it, and its
//! PoolRelay has registered within the lease (`POST /pools`).
//!
//! Each source gives what it knows best. The registration says what the pool
//! serves: model, capabilities, capacity and location. MCS gives the
//! addresses other clusters can reach: the Relay's, and the frontends'.
//!
//! Each source also covers a gap in the other. MCS drops a pool within
//! seconds when its Relay stops or its export is removed. The lease drops a
//! pool that MCS still shows as Ready, for example when the pool's cluster
//! loses contact with the MCS implementation.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::Duration;

use crate::pool_discovery::{
    Completeness, Directory, DiscoveryError, PoolDirectory, PoolEvent, PoolListing, Snapshot,
    WatchEvent, WatchStream,
};
use async_trait::async_trait;
use dynamo_global_common::{Lease, PoolKey};
use futures::StreamExt;
use tokio::sync::broadcast;
use tokio::time::{Instant, MissedTickBehavior};
use tokio_util::sync::CancellationToken;

use super::stream::{EVENT_BUFFER, watch_stream};

/// Joins an MCS pool directory with a registration store. Lists are `Exact`
/// while both sources' latest lists are, and `Estimated` otherwise, for
/// example until a lease has passed since the registration store started.
///
/// One task watches both sources and keeps the join, which serves `list`
/// and every `list_and_watch` stream. Each event from either source
/// updates one pool. The task starts on first use and stops when the
/// directory is dropped.
pub struct RegisteredPoolDirectory {
    mcs: Arc<PoolDirectory>,
    registrations: Arc<PoolDirectory>,
    lease: Lease,
    join: Arc<Join>,
    joining: OnceLock<()>,
    stop: CancellationToken,
}

impl RegisteredPoolDirectory {
    /// `lease` is the registrations' lease. A pool that only one source has
    /// listed for two lease durations is logged once as a warning, since it
    /// usually means its names do not match.
    pub fn new(mcs: Arc<PoolDirectory>, registrations: Arc<PoolDirectory>, lease: Lease) -> Self {
        Self {
            mcs,
            registrations,
            lease,
            join: Arc::new(Join {
                state: Mutex::default(),
                events: broadcast::channel(EVENT_BUFFER).0,
            }),
            joining: OnceLock::new(),
            stop: CancellationToken::new(),
        }
    }

    /// Every `Directory` method runs in a Tokio runtime, so the spawn is safe.
    fn ensure_joining(&self) {
        self.joining.get_or_init(|| {
            tokio::spawn(join_sources(
                Arc::clone(&self.mcs),
                Arc::clone(&self.registrations),
                Arc::clone(&self.join),
                self.lease,
                self.stop.clone(),
            ));
        });
    }
}

impl Drop for RegisteredPoolDirectory {
    fn drop(&mut self) {
        self.stop.cancel();
    }
}

impl fmt::Debug for RegisteredPoolDirectory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RegisteredPoolDirectory")
            .field("lease", &self.lease)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl Directory<PoolListing, PoolKey> for RegisteredPoolDirectory {
    /// From the join once both sources have synced. Before that, joins one
    /// list of each, so a caller never waits for the watches to start.
    async fn list(&self) -> Result<Snapshot<PoolListing>, DiscoveryError> {
        self.ensure_joining();
        if let Some(snapshot) = self.join.snapshot() {
            return Ok(snapshot);
        }
        let (mcs, registered) = tokio::try_join!(self.mcs.list(), self.registrations.list())?;
        let both_exact = mcs.completeness == Completeness::Exact
            && registered.completeness == Completeness::Exact;
        let registered: HashMap<PoolKey, PoolListing> = registered
            .items
            .into_iter()
            .map(|listing| (listing.key.clone(), listing))
            .collect();
        Ok(Snapshot {
            items: mcs
                .items
                .iter()
                .filter_map(|mcs| Some(merge(mcs, registered.get(&mcs.key)?)))
                .collect(),
            completeness: completeness(both_exact),
        })
    }

    async fn list_and_watch(
        &self,
        cancel_token: Option<CancellationToken>,
    ) -> Result<WatchStream<PoolListing, PoolKey>, DiscoveryError> {
        self.ensure_joining();
        let join = Arc::clone(&self.join);
        Ok(watch_stream(
            move || join.subscribe_now(),
            cancel_token.unwrap_or_default(),
            self.stop.clone(),
        ))
    }
}

/// A pool missing from either source is missing from the join, so the join
/// is `Exact` only when both sources are.
fn completeness(both_exact: bool) -> Completeness {
    if both_exact {
        Completeness::Exact
    } else {
        Completeness::Estimated
    }
}

/// What the pool serves comes from its registration. Where to reach it comes
/// from MCS: the registration's own `frontends` are not used, because a
/// PoolRelay cannot know which of its addresses other clusters can reach.
fn merge(mcs: &PoolListing, registered: &PoolListing) -> PoolListing {
    PoolListing {
        key: mcs.key.clone(),
        location: registered.location.clone(),
        model: registered.model.clone(),
        capabilities: registered.capabilities.clone(),
        frontends: mcs.frontends.clone(),
        capacity: registered.capacity.clone(),
        relay_identities: mcs.relay_identities.clone(),
    }
}

/// Single consumer of both sources' watches, so the join applies their events
/// in order.
async fn join_sources(
    mcs: Arc<PoolDirectory>,
    registrations: Arc<PoolDirectory>,
    join: Arc<Join>,
    lease: Lease,
    stop: CancellationToken,
) {
    let watches = tokio::try_join!(
        mcs.list_and_watch(Some(stop.child_token())),
        registrations.list_and_watch(Some(stop.child_token())),
    );
    let (mut from_mcs, mut from_registrations) = match watches {
        Ok(watches) => watches,
        Err(error) => {
            // The backends in this crate subscribe without contacting anything.
            tracing::error!(
                ?error,
                "cannot watch the pool sources; joining on each read"
            );
            // End watch streams instead of leaving them waiting for a Resync.
            stop.cancel();
            return;
        }
    };
    let warn_after = lease.duration() * 2;
    let mut ticks = tokio::time::interval(lease.heartbeat_interval());
    ticks.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        let (source, event) = tokio::select! {
            biased;
            () = stop.cancelled() => return,
            event = from_mcs.next() => (Source::Mcs, event),
            event = from_registrations.next() => (Source::Registrations, event),
            _ = ticks.tick() => {
                join.warn_one_sided(warn_after);
                continue;
            }
        };
        match event {
            Some(Ok(event)) => join.apply(source, event),
            Some(Err(error)) => tracing::warn!(?error, ?source, "pool source watch error"),
            None => {
                // Only a dropped source ends its stream. Serve fresh joins
                // rather than a view that no longer changes, and end watch
                // streams rather than leave them waiting.
                join.unsync();
                stop.cancel();
                return;
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Source {
    Mcs,
    Registrations,
}

struct Join {
    state: Mutex<JoinState>,
    /// Sent while holding the `state` lock, so a subscriber that snapshots
    /// and subscribes under it misses no event and sees none twice.
    events: broadcast::Sender<PoolEvent>,
}

impl Join {
    fn lock(&self) -> MutexGuard<'_, JoinState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn apply(&self, source: Source, event: PoolEvent) {
        let mut state = self.lock();
        for event in state.apply(source, event, Instant::now()) {
            // No subscribers is not an error.
            let _ = self.events.send(event);
        }
    }

    /// `None` until both sources have synced.
    fn snapshot(&self) -> Option<Snapshot<PoolListing>> {
        let state = self.lock();
        state.synced().then(|| state.snapshot())
    }

    fn subscribe_now(
        &self,
    ) -> (
        broadcast::Receiver<PoolEvent>,
        Option<Snapshot<PoolListing>>,
    ) {
        let state = self.lock();
        (
            self.events.subscribe(),
            state.synced().then(|| state.snapshot()),
        )
    }

    fn unsync(&self) {
        let mut state = self.lock();
        state.mcs.synced = false;
        state.registered.synced = false;
    }

    /// Logs each pool that only one source has listed for `after`, once.
    fn warn_one_sided(&self, after: Duration) {
        let now = Instant::now();
        let mut state = self.lock();
        let JoinState { mcs, one_sided, .. } = &mut *state;
        for (key, one_sided) in one_sided.iter_mut() {
            if one_sided.warned || now.duration_since(one_sided.since) < after {
                continue;
            }
            one_sided.warned = true;
            if mcs.listings.contains_key(key) {
                tracing::warn!(
                    ?key,
                    "MCS lists a Ready PoolRelay for this pool, but the pool has no live \
                     registration here, so it is not listed. Check that its PoolRelay calls \
                     POST /pools on this service with this key."
                );
            } else {
                tracing::warn!(
                    ?key,
                    "this pool is registered, but MCS lists no Ready PoolRelay for it, so it is \
                     not listed. Check that cluster_id is the pool's MCS cluster name and that its \
                     PoolRelay Service is exported."
                );
            }
        }
    }
}

#[derive(Default)]
struct JoinState {
    mcs: SourceView,
    registered: SourceView,
    /// The join: pools both sources list.
    pools: HashMap<PoolKey, PoolListing>,
    /// Pools only one source lists, after both have synced.
    one_sided: HashMap<PoolKey, OneSided>,
}

#[derive(Default)]
struct SourceView {
    /// The source has sent its first `Resync`.
    synced: bool,
    /// Its latest `Resync` was `Exact`.
    exact: bool,
    listings: HashMap<PoolKey, PoolListing>,
}

struct OneSided {
    since: Instant,
    warned: bool,
}

impl JoinState {
    fn synced(&self) -> bool {
        self.mcs.synced && self.registered.synced
    }

    fn snapshot(&self) -> Snapshot<PoolListing> {
        Snapshot {
            items: self.pools.values().cloned().collect(),
            completeness: completeness(self.mcs.exact && self.registered.exact),
        }
    }

    /// Returns the events to publish. Nothing is published until both
    /// sources have synced; then one `Resync` starts the join. Each later
    /// `Resync` from a source, such as one that changes its completeness,
    /// publishes another.
    fn apply(&mut self, source: Source, event: PoolEvent, now: Instant) -> Vec<PoolEvent> {
        let was_synced = self.synced();
        let view = match source {
            Source::Mcs => &mut self.mcs,
            Source::Registrations => &mut self.registered,
        };
        let key = match event {
            WatchEvent::Upserted(listing) => {
                let key = listing.key.clone();
                view.listings.insert(key.clone(), listing);
                key
            }
            WatchEvent::Removed(key) => {
                view.listings.remove(&key);
                key
            }
            WatchEvent::Resync(snapshot) => {
                view.exact = snapshot.completeness == Completeness::Exact;
                view.listings = snapshot
                    .items
                    .into_iter()
                    .map(|listing| (listing.key.clone(), listing))
                    .collect();
                view.synced = true;
                if !self.synced() {
                    return Vec::new();
                }
                self.rebuild(now);
                return vec![WatchEvent::Resync(self.snapshot())];
            }
        };
        if !was_synced {
            return Vec::new();
        }
        self.update(&key, now).into_iter().collect()
    }

    /// Joins every pool either source lists again after a source (re)listed.
    /// Only then is the whole join rebuilt. A joined pool is still in one
    /// source after one source relists, so `update` drops it from the join.
    fn rebuild(&mut self, now: Instant) {
        let JoinState {
            mcs,
            registered,
            one_sided,
            ..
        } = &mut *self;
        one_sided.retain(|key, _| {
            mcs.listings.contains_key(key) != registered.listings.contains_key(key)
        });
        let keys: HashSet<PoolKey> = mcs
            .listings
            .keys()
            .chain(registered.listings.keys())
            .cloned()
            .collect();
        for key in &keys {
            // The `Resync` replaces per-pool events.
            let _ = self.update(key, now);
        }
    }

    /// Joins one pool again after one source changed it. Returns the event to
    /// publish if the joined listing changed.
    fn update(&mut self, key: &PoolKey, now: Instant) -> Option<PoolEvent> {
        let mcs = self.mcs.listings.get(key);
        let registered = self.registered.listings.get(key);
        if mcs.is_some() == registered.is_some() {
            self.one_sided.remove(key);
        } else {
            self.one_sided.entry(key.clone()).or_insert(OneSided {
                since: now,
                warned: false,
            });
        }
        let Some((mcs, registered)) = mcs.zip(registered) else {
            return self
                .pools
                .remove(key)
                .map(|_| WatchEvent::Removed(key.clone()));
        };
        let listing = merge(mcs, registered);
        if self.pools.get(key) == Some(&listing) {
            return None;
        }
        self.pools.insert(key.clone(), listing.clone());
        Some(WatchEvent::Upserted(listing))
    }
}

#[cfg(test)]
mod tests {
    use crate::backends::memory::{MemoryPoolStore, SWEEP_EVERY};
    use crate::pool_discovery::{Capacity, PoolRegistrar, PoolRegistration};
    use dynamo_global_common::PoolLocation;
    use url::Url;

    use super::*;

    fn lease() -> Lease {
        Lease::new(5, 15).unwrap()
    }

    fn key(dgd: &str) -> PoolKey {
        PoolKey::new("west", "dynamo", dgd).unwrap()
    }

    /// What the MCS backend lists: addresses, and nothing about the model.
    fn from_mcs(dgd: &str) -> PoolListing {
        PoolListing {
            key: key(dgd),
            location: PoolLocation {
                region: "west".to_owned(),
                availability_zone: None,
                datacenter: None,
            },
            model: None,
            capabilities: Vec::new(),
            frontends: vec![Url::parse("http://10.43.0.2:8000/").unwrap()],
            capacity: Vec::new(),
            relay_identities: vec!["ip:10.43.0.1".to_owned()],
        }
    }

    /// What a PoolRelay registers.
    fn registered(dgd: &str, model: &str) -> PoolRegistration {
        PoolRegistration {
            key: key(dgd),
            location: PoolLocation {
                region: "us-west-2".to_owned(),
                availability_zone: Some("us-west-2a".to_owned()),
                datacenter: None,
            },
            model: model.to_owned(),
            capabilities: Vec::new(),
            frontends: vec![Url::parse("http://mocker-frontend:8000/").unwrap()],
            capacity: vec![Capacity {
                sku: "H100".to_owned(),
                workers: 8,
            }],
        }
    }

    /// What the registration store lists for `registered(dgd, model)`.
    fn listed(dgd: &str, model: &str) -> PoolListing {
        let registration = registered(dgd, model);
        PoolListing {
            key: registration.key,
            location: registration.location,
            model: Some(registration.model),
            capabilities: registration.capabilities,
            frontends: registration.frontends,
            capacity: registration.capacity,
            relay_identities: Vec::new(),
        }
    }

    /// Stands in for the MCS backend: pools stay until removed.
    struct FakeMcs {
        listings: Arc<Mutex<HashMap<PoolKey, PoolListing>>>,
        events: broadcast::Sender<PoolEvent>,
    }

    impl FakeMcs {
        fn new() -> Self {
            Self {
                listings: Arc::default(),
                events: broadcast::channel(EVENT_BUFFER).0,
            }
        }

        fn put(&self, listing: PoolListing) {
            let mut listings = self.listings.lock().unwrap();
            listings.insert(listing.key.clone(), listing.clone());
            let _ = self.events.send(WatchEvent::Upserted(listing));
        }

        /// Lists every pool again, with `completeness`, as the MCS backend
        /// does when its watch keeps failing and when it recovers.
        fn relist(&self, completeness: Completeness) {
            let mut snapshot = exact(&self.listings.lock().unwrap());
            snapshot.completeness = completeness;
            let _ = self.events.send(WatchEvent::Resync(snapshot));
        }

        fn remove(&self, key: &PoolKey) {
            let mut listings = self.listings.lock().unwrap();
            if listings.remove(key).is_some() {
                let _ = self.events.send(WatchEvent::Removed(key.clone()));
            }
        }
    }

    fn exact(listings: &HashMap<PoolKey, PoolListing>) -> Snapshot<PoolListing> {
        Snapshot {
            items: listings.values().cloned().collect(),
            completeness: Completeness::Exact,
        }
    }

    #[async_trait]
    impl Directory<PoolListing, PoolKey> for FakeMcs {
        async fn list(&self) -> Result<Snapshot<PoolListing>, DiscoveryError> {
            Ok(exact(&self.listings.lock().unwrap()))
        }

        async fn list_and_watch(
            &self,
            cancel_token: Option<CancellationToken>,
        ) -> Result<WatchStream<PoolListing, PoolKey>, DiscoveryError> {
            let listings = Arc::clone(&self.listings);
            let events = self.events.clone();
            Ok(watch_stream(
                move || {
                    let listings = listings.lock().unwrap();
                    (events.subscribe(), Some(exact(&listings)))
                },
                cancel_token.unwrap_or_default(),
                CancellationToken::new(),
            ))
        }
    }

    struct Fixture {
        directory: RegisteredPoolDirectory,
        mcs: Arc<FakeMcs>,
        registrations: Arc<MemoryPoolStore>,
    }

    fn fixture() -> Fixture {
        let mcs = Arc::new(FakeMcs::new());
        let registrations = Arc::new(MemoryPoolStore::new(lease()));
        Fixture {
            directory: RegisteredPoolDirectory::new(mcs.clone(), registrations.clone(), lease()),
            mcs,
            registrations,
        }
    }

    /// The listing for `key`, if any.
    async fn find(directory: &PoolDirectory, key: &PoolKey) -> Option<PoolListing> {
        let snapshot = directory.list().await.unwrap();
        snapshot
            .items
            .into_iter()
            .find(|listing| &listing.key == key)
    }

    impl Fixture {
        fn in_mcs(&self, dgd: &str) {
            self.mcs.put(from_mcs(dgd));
        }

        async fn register(&self, dgd: &str, model: &str) {
            let _ = self
                .registrations
                .register(registered(dgd, model))
                .await
                .unwrap();
        }

        async fn keys(&self) -> Vec<PoolKey> {
            let mut keys: Vec<PoolKey> = self
                .directory
                .list()
                .await
                .unwrap()
                .items
                .into_iter()
                .map(|listing| listing.key)
                .collect();
            keys.sort_by(|a, b| a.dgd_name().cmp(b.dgd_name()));
            keys
        }
    }

    /// Lets the join task run.
    async fn settle() {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }

    #[tokio::test(start_paused = true)]
    async fn joins_only_pools_both_sources_list() {
        let fx = fixture();
        fx.in_mcs("relay-only");
        fx.register("registered-only", "m").await;
        fx.in_mcs("both");
        fx.register("both", "qwen").await;

        // Before the join has synced, each read joins the sources itself.
        assert_eq!(fx.keys().await, [key("both")]);
        settle().await;
        assert!(fx.directory.join.snapshot().is_some());
        assert_eq!(fx.keys().await, [key("both")]);

        // What the pool serves comes from its registration, and where to reach
        // it from MCS.
        let listing = find(&fx.directory, &key("both")).await.unwrap();
        let registration = registered("both", "qwen");
        assert_eq!(listing.model.as_deref(), Some("qwen"));
        assert_eq!(listing.location, registration.location);
        assert_eq!(listing.capacity, registration.capacity);
        assert_eq!(listing.frontends, from_mcs("both").frontends);
        assert_eq!(listing.relay_identities, from_mcs("both").relay_identities);
    }

    #[tokio::test(start_paused = true)]
    async fn watch_follows_both_sources() {
        let fx = fixture();
        fx.in_mcs("mocker");
        fx.register("mocker", "m1").await;
        let mut watch = fx.directory.list_and_watch(None).await.unwrap();
        assert!(matches!(
            watch.next().await,
            Some(Ok(WatchEvent::Upserted(_)))
        ));
        assert!(
            matches!(watch.next().await, Some(Ok(WatchEvent::Resync(s))) if s.items.len() == 1)
        );

        // A heartbeat with a new model is published; an unchanged one
        // publishes nothing.
        fx.register("mocker", "m2").await;
        fx.register("mocker", "m2").await;
        let Some(Ok(WatchEvent::Upserted(second))) = watch.next().await else {
            panic!("expected Added");
        };
        assert_eq!(second.model.as_deref(), Some("m2"));

        // MCS dropping the Relay removes the pool at once, though it is still
        // registered.
        fx.mcs.remove(&key("mocker"));
        assert!(
            matches!(watch.next().await, Some(Ok(WatchEvent::Removed(k))) if k == key("mocker"))
        );
        fx.in_mcs("mocker");
        assert!(matches!(
            watch.next().await,
            Some(Ok(WatchEvent::Upserted(_)))
        ));

        // A heartbeat keeps the pool while the registration store's first
        // lease passes, and then its lists are complete.
        tokio::time::advance(Duration::from_secs(10)).await;
        fx.register("mocker", "m2").await;
        tokio::time::advance(lease().duration() - Duration::from_secs(10)).await;
        assert!(
            matches!(watch.next().await, Some(Ok(WatchEvent::Resync(s))) if s.items.len() == 1)
        );

        // No heartbeat for longer than the lease, as when the pool's cluster
        // is cut off but its imported slices still say Ready: the pool goes
        // at the first sweep after its lease ends. The last heartbeat falls
        // between two heartbeat intervals, so a sweep that ran only once per
        // heartbeat interval would be seconds late. `sleep`, unlike
        // `advance`, lets each sweep run at its own time, before the lease
        // ends.
        tokio::time::advance(Duration::from_secs(2)).await;
        fx.register("mocker", "m2").await;
        tokio::time::sleep(lease().duration() + Duration::from_millis(1)).await;
        let removed = tokio::time::timeout(SWEEP_EVERY + Duration::from_millis(100), watch.next())
            .await
            .expect("removed within one sweep of the lease's end");
        assert!(matches!(removed, Some(Ok(WatchEvent::Removed(k))) if k == key("mocker")));
        assert!(find(&*fx.mcs, &key("mocker")).await.is_some());
    }

    #[tokio::test(start_paused = true)]
    async fn the_join_is_complete_once_both_sources_are() {
        let fx = fixture();
        fx.in_mcs("mocker");
        fx.register("mocker", "m1").await;
        let completeness = || async { fx.directory.list().await.unwrap().completeness };
        // MCS lists every pool, but until a lease has passed since the
        // registration store started, a PoolRelay may not have registered.
        assert_eq!(completeness().await, Completeness::Estimated);
        let mut watch = fx.directory.list_and_watch(None).await.unwrap();
        assert!(matches!(
            watch.next().await,
            Some(Ok(WatchEvent::Upserted(_)))
        ));
        assert!(
            matches!(watch.next().await, Some(Ok(WatchEvent::Resync(s))) if s.completeness == Completeness::Estimated)
        );

        // The PoolRelay keeps registering while the first lease passes.
        tokio::time::advance(Duration::from_secs(10)).await;
        fx.register("mocker", "m1").await;
        tokio::time::advance(lease().duration() - Duration::from_secs(10)).await;
        assert!(
            matches!(watch.next().await, Some(Ok(WatchEvent::Resync(s))) if s.completeness == Completeness::Exact && s.items.len() == 1)
        );
        assert_eq!(completeness().await, Completeness::Exact);

        // A stale MCS view makes the join `Estimated` again.
        fx.mcs.relist(Completeness::Estimated);
        assert!(
            matches!(watch.next().await, Some(Ok(WatchEvent::Resync(s))) if s.completeness == Completeness::Estimated)
        );
        assert_eq!(completeness().await, Completeness::Estimated);
    }

    #[tokio::test(start_paused = true)]
    async fn tracks_pools_only_one_source_lists() {
        let join = Join {
            state: Mutex::default(),
            events: broadcast::channel(16).0,
        };
        let resync = |items: Vec<PoolListing>| {
            WatchEvent::Resync(Snapshot {
                items,
                completeness: Completeness::Exact,
            })
        };
        let one_sided = |join: &Join| {
            let mut one_sided: Vec<(String, bool)> = join
                .lock()
                .one_sided
                .iter()
                .map(|(key, one_sided)| (key.dgd_name().to_owned(), one_sided.warned))
                .collect();
            one_sided.sort();
            one_sided
        };
        join.apply(
            Source::Mcs,
            resync(vec![from_mcs("relay-only"), from_mcs("both")]),
        );
        join.apply(
            Source::Registrations,
            resync(vec![listed("registered-only", "m"), listed("both", "m")]),
        );
        let after = lease().duration() * 2;
        join.warn_one_sided(after);
        assert_eq!(
            one_sided(&join),
            [
                ("registered-only".to_owned(), false),
                ("relay-only".to_owned(), false)
            ]
        );

        // Each is logged once, `after` from when it became one-sided, though
        // its one source changed it meanwhile.
        tokio::time::advance(after / 2).await;
        join.apply(Source::Mcs, WatchEvent::Upserted(from_mcs("relay-only")));
        tokio::time::advance(after / 2).await;
        join.warn_one_sided(after);
        assert_eq!(
            one_sided(&join),
            [
                ("registered-only".to_owned(), true),
                ("relay-only".to_owned(), true)
            ]
        );

        // A pool both sources list, or neither, is no longer one-sided, also
        // after a relist.
        join.apply(
            Source::Registrations,
            WatchEvent::Upserted(listed("relay-only", "m")),
        );
        join.apply(
            Source::Registrations,
            resync(vec![listed("both", "m"), listed("relay-only", "m")]),
        );
        assert!(one_sided(&join).is_empty());
    }
}
