// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The MCS store with registrations. A pool is listed only while both
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
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::Duration;

use crate::pool_discovery::watch::{Directory, Snapshot, WatchStream};
use crate::pool_discovery::{
    Completeness, LeasePolicy, PoolDirectory, PoolDiscoveryError, PoolEvent, PoolRecord, Revision,
    WatchEvent,
};
use async_trait::async_trait;
use dynamo_global_router_shared::PoolKey;
use futures::StreamExt;
use tokio::sync::broadcast;
use tokio::time::{Instant, MissedTickBehavior};
use tokio_util::sync::CancellationToken;

use super::stream::{EVENT_BUFFER, watch_stream};

/// Joins an MCS pool directory with a registration store. Lists are
/// `Estimated`, because registrations are learned from PoolRelays, not read
/// from a shared store.
///
/// One task watches both sources and keeps the join, which serves `get`,
/// `list` and every `list_and_watch` stream. Each event from either source
/// updates one pool. The task starts on first use and stops when the
/// directory is dropped.
pub struct RegisteredPoolDirectory {
    mcs: Arc<PoolDirectory>,
    registrations: Arc<PoolDirectory>,
    lease: LeasePolicy,
    join: Arc<Join>,
    joining: OnceLock<()>,
    stop: CancellationToken,
}

impl RegisteredPoolDirectory {
    /// `lease` is the registrations' lease. A pool that only one source has
    /// listed for two lease durations is logged once as a warning, since it
    /// usually means its names do not match.
    pub fn new(
        mcs: Arc<PoolDirectory>,
        registrations: Arc<PoolDirectory>,
        lease: LeasePolicy,
    ) -> Self {
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

#[async_trait]
impl Directory<PoolRecord, PoolKey> for RegisteredPoolDirectory {
    async fn check_connection(&self) -> Result<(), PoolDiscoveryError> {
        tokio::try_join!(
            self.mcs.check_connection(),
            self.registrations.check_connection()
        )
        .map(|_| ())
    }

    /// From the join once both sources have synced. Before that, joins one
    /// list of each, so a caller never waits for the watches to start.
    async fn list(&self) -> Result<Snapshot<PoolRecord>, PoolDiscoveryError> {
        self.ensure_joining();
        if let Some(snapshot) = self.join.snapshot() {
            return Ok(snapshot);
        }
        let (mcs, registered) = tokio::try_join!(self.mcs.list(), self.registrations.list())?;
        let registered: HashMap<PoolKey, PoolRecord> = registered
            .items
            .into_iter()
            .map(|record| (record.key.clone(), record))
            .collect();
        Ok(Snapshot {
            items: mcs
                .items
                .iter()
                .filter_map(|mcs| Some(merge(mcs, registered.get(&mcs.key)?)))
                .collect(),
            completeness: Completeness::Estimated,
        })
    }

    async fn get(&self, key: &PoolKey) -> Result<Option<PoolRecord>, PoolDiscoveryError> {
        self.ensure_joining();
        if let Some(record) = self.join.get(key) {
            return Ok(record);
        }
        let (mcs, registered) = tokio::try_join!(self.mcs.get(key), self.registrations.get(key))?;
        Ok(mcs
            .zip(registered)
            .map(|(mcs, registered)| merge(&mcs, &registered)))
    }

    async fn list_and_watch(
        &self,
        cancel_token: Option<CancellationToken>,
    ) -> Result<WatchStream<PoolRecord, PoolKey>, PoolDiscoveryError> {
        self.ensure_joining();
        let join = Arc::clone(&self.join);
        Ok(watch_stream(
            move || join.subscribe_now(),
            cancel_token.unwrap_or_default(),
            self.stop.clone(),
        ))
    }
}

/// What the pool serves comes from its registration. Where to reach it comes
/// from MCS: the registration's own `frontends` are not used, because a
/// PoolRelay cannot know which of its addresses other clusters can reach.
fn merge(mcs: &PoolRecord, registered: &PoolRecord) -> PoolRecord {
    PoolRecord {
        key: mcs.key.clone(),
        location: registered.location.clone(),
        runtime_namespace: registered.runtime_namespace.clone(),
        frontend_endpoint: registered.frontend_endpoint.clone(),
        model: registered.model.clone(),
        capabilities: registered.capabilities.clone(),
        frontends: mcs.frontends.clone(),
        capacity: registered.capacity.clone(),
        relay_identity: mcs.relay_identity.clone(),
        revision: Revision::default(),
    }
}

/// Single consumer of both sources' watches, so the join applies their events
/// in order.
async fn join_sources(
    mcs: Arc<PoolDirectory>,
    registrations: Arc<PoolDirectory>,
    join: Arc<Join>,
    lease: LeasePolicy,
    stop: CancellationToken,
) {
    let watches = tokio::try_join!(
        mcs.list_and_watch(Some(stop.child_token())),
        registrations.list_and_watch(Some(stop.child_token())),
    );
    let (mut from_mcs, mut from_registrations) = match watches {
        Ok(watches) => watches,
        Err(error) => {
            // The stores in this crate subscribe without contacting anything.
            tracing::error!(
                ?error,
                "cannot watch the pool sources; joining on each read"
            );
            // End watch streams instead of leaving them waiting for a Resync.
            stop.cancel();
            return;
        }
    };
    let warn_after = lease.lease_duration * 2;
    let mut ticks = tokio::time::interval(lease.heartbeat_interval);
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
    fn snapshot(&self) -> Option<Snapshot<PoolRecord>> {
        let state = self.lock();
        state.synced().then(|| state.snapshot())
    }

    /// `None` until both sources have synced.
    fn get(&self, key: &PoolKey) -> Option<Option<PoolRecord>> {
        let state = self.lock();
        state.synced().then(|| state.pools.get(key).cloned())
    }

    fn subscribe_now(&self) -> (broadcast::Receiver<PoolEvent>, Option<Snapshot<PoolRecord>>) {
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
            if mcs.records.contains_key(key) {
                tracing::warn!(
                    ?key,
                    "MCS lists a Ready PoolRelay for this pool, but the pool has no live \
                     registration, so it is not listed. Check that its PoolRelay calls \
                     POST /pools with this key."
                );
            } else {
                tracing::warn!(
                    ?key,
                    "this pool is registered, but MCS lists no Ready PoolRelay for it, so it is \
                     not listed. Check that site_id is the pool's MCS cluster name and that its \
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
    pools: HashMap<PoolKey, PoolRecord>,
    last_revision: u64,
    /// Pools only one source lists, after both have synced.
    one_sided: HashMap<PoolKey, OneSided>,
}

#[derive(Default)]
struct SourceView {
    /// The source has sent its first `Resync`.
    synced: bool,
    records: HashMap<PoolKey, PoolRecord>,
}

struct OneSided {
    since: Instant,
    warned: bool,
}

impl JoinState {
    fn synced(&self) -> bool {
        self.mcs.synced && self.registered.synced
    }

    fn snapshot(&self) -> Snapshot<PoolRecord> {
        Snapshot {
            items: self.pools.values().cloned().collect(),
            completeness: Completeness::Estimated,
        }
    }

    /// Returns the events to publish. Nothing is published until both
    /// sources have synced; then one `Resync` starts the join.
    fn apply(&mut self, source: Source, event: PoolEvent, now: Instant) -> Vec<PoolEvent> {
        let was_synced = self.synced();
        let view = match source {
            Source::Mcs => &mut self.mcs,
            Source::Registrations => &mut self.registered,
        };
        let key = match event {
            WatchEvent::Added(record) => {
                let key = record.key.clone();
                view.records.insert(key.clone(), record);
                key
            }
            WatchEvent::Removed(key) => {
                view.records.remove(&key);
                key
            }
            WatchEvent::Resync(snapshot) => {
                view.records = snapshot
                    .items
                    .into_iter()
                    .map(|record| (record.key.clone(), record))
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

    /// Joins every pool again after a source (re)listed. Only then is the
    /// whole join rebuilt.
    fn rebuild(&mut self, now: Instant) {
        let JoinState {
            mcs,
            registered,
            pools,
            one_sided,
            ..
        } = &mut *self;
        let both =
            |key: &PoolKey| mcs.records.contains_key(key) && registered.records.contains_key(key);
        pools.retain(|key, _| both(key));
        one_sided
            .retain(|key, _| mcs.records.contains_key(key) != registered.records.contains_key(key));
        let keys: HashSet<PoolKey> = mcs
            .records
            .keys()
            .chain(registered.records.keys())
            .cloned()
            .collect();
        for key in &keys {
            // The `Resync` replaces per-pool events.
            let _ = self.update(key, now);
        }
    }

    /// Joins one pool again after one source changed it. Returns the event to
    /// publish if the joined record changed.
    fn update(&mut self, key: &PoolKey, now: Instant) -> Option<PoolEvent> {
        let mcs = self.mcs.records.get(key);
        let registered = self.registered.records.get(key);
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
        let mut record = merge(mcs, registered);
        if let Some(old) = self.pools.get(key) {
            record.revision = old.revision;
            if *old == record {
                return None;
            }
        }
        self.last_revision += 1;
        record.revision = Revision(self.last_revision);
        self.pools.insert(key.clone(), record.clone());
        Some(WatchEvent::Added(record))
    }
}

#[cfg(test)]
mod tests {
    use crate::backends::memory::{MemoryPoolStore, SWEEP_EVERY};
    use crate::pool_discovery::{
        Capacity, PeerCredentials, PoolRecordBuilder, PoolRegistrar, RelayAuthenticator,
        RelayIdentity, VerifiedRelayIdentity, verify,
    };
    use dynamo_global_router_shared::PoolLocation;
    use url::Url;

    use super::*;

    const LEASE: LeasePolicy = LeasePolicy {
        heartbeat_interval: Duration::from_secs(5),
        lease_duration: Duration::from_secs(15),
    };

    /// Stands in for the MCS store: its records never expire here.
    const NO_EXPIRY: LeasePolicy = LeasePolicy {
        heartbeat_interval: Duration::from_secs(3600),
        lease_duration: Duration::from_secs(7200),
    };

    struct AnyCaller;

    #[async_trait]
    impl RelayAuthenticator for AnyCaller {
        async fn authenticate(
            &self,
            _credentials: &PeerCredentials,
        ) -> Result<RelayIdentity, PoolDiscoveryError> {
            Ok(RelayIdentity("test".to_owned()))
        }
    }

    async fn caller() -> VerifiedRelayIdentity {
        verify(
            &AnyCaller,
            &PeerCredentials::PeerAddress("10.0.0.1".parse().unwrap()),
        )
        .await
        .unwrap()
    }

    fn key(dgd: &str) -> PoolKey {
        PoolKey::new("west", "dynamo", dgd).unwrap()
    }

    /// What the MCS store lists: addresses, and nothing about the model.
    fn from_mcs(dgd: &str) -> PoolRecord {
        PoolRecordBuilder::default()
            .key(key(dgd))
            .location(PoolLocation {
                region: "west".to_owned(),
                availability_zone: None,
                cluster: Some("west".to_owned()),
                datacenter: None,
            })
            .frontends(vec![Url::parse("http://10.43.0.2:8000/").unwrap()])
            .relay_identity(Some(RelayIdentity("ip:10.43.0.1".to_owned())))
            .build()
            .unwrap()
    }

    /// What a PoolRelay registers.
    fn registered(dgd: &str, model: &str) -> PoolRecord {
        PoolRecordBuilder::default()
            .key(key(dgd))
            .location(PoolLocation {
                region: "us-west-2".to_owned(),
                availability_zone: Some("us-west-2a".to_owned()),
                cluster: Some("west".to_owned()),
                datacenter: None,
            })
            .model(Some(model.to_owned()))
            .capacity(vec![Capacity {
                sku: "H100".to_owned(),
                count: 8,
            }])
            .frontends(vec![Url::parse("http://mocker-frontend:8000/").unwrap()])
            .build()
            .unwrap()
    }

    struct Fixture {
        directory: RegisteredPoolDirectory,
        mcs: Arc<MemoryPoolStore>,
        registrations: Arc<MemoryPoolStore>,
        caller: VerifiedRelayIdentity,
    }

    async fn fixture(registration_lease: LeasePolicy) -> Fixture {
        let mcs = Arc::new(MemoryPoolStore::new(NO_EXPIRY).unwrap());
        let registrations = Arc::new(MemoryPoolStore::new(registration_lease).unwrap());
        Fixture {
            directory: RegisteredPoolDirectory::new(mcs.clone(), registrations.clone(), LEASE),
            mcs,
            registrations,
            caller: caller().await,
        }
    }

    impl Fixture {
        async fn in_mcs(&self, dgd: &str) {
            self.mcs
                .register(from_mcs(dgd), &self.caller)
                .await
                .unwrap();
        }

        async fn register(&self, dgd: &str, model: &str) {
            self.registrations
                .register(registered(dgd, model), &self.caller)
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
                .map(|record| record.key)
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
    async fn lists_only_pools_both_sources_have() {
        let fx = fixture(LEASE).await;
        fx.in_mcs("relay-only").await;
        fx.register("registered-only", "m").await;
        fx.in_mcs("both").await;
        fx.register("both", "m").await;

        // Before the join has synced, each read joins the sources itself.
        assert_eq!(fx.keys().await, [key("both")]);
        settle().await;
        assert!(fx.directory.join.snapshot().is_some());
        assert_eq!(fx.keys().await, [key("both")]);
    }

    #[tokio::test(start_paused = true)]
    async fn the_registration_says_what_mcs_says_where() {
        let fx = fixture(LEASE).await;
        fx.in_mcs("mocker").await;
        fx.register("mocker", "qwen").await;
        let record = fx.directory.get(&key("mocker")).await.unwrap().unwrap();
        assert_eq!(record.model.as_deref(), Some("qwen"));
        assert_eq!(record.location, registered("mocker", "qwen").location);
        assert_eq!(
            record.capacity,
            [Capacity {
                sku: "H100".to_owned(),
                count: 8
            }]
        );
        // The frontends and the Relay come from MCS, not the registration.
        assert_eq!(record.frontends, from_mcs("mocker").frontends);
        assert_eq!(record.relay_identity, from_mcs("mocker").relay_identity);
    }

    #[tokio::test(start_paused = true)]
    async fn watch_follows_changes_from_both_sources() {
        let fx = fixture(LEASE).await;
        fx.in_mcs("mocker").await;
        fx.register("mocker", "m1").await;
        let mut watch = fx.directory.list_and_watch(None).await.unwrap();
        let Some(Ok(WatchEvent::Added(first))) = watch.next().await else {
            panic!("expected Added");
        };
        assert!(
            matches!(watch.next().await, Some(Ok(WatchEvent::Resync(s))) if s.items.len() == 1)
        );

        // A heartbeat with a new model is a new revision.
        fx.register("mocker", "m2").await;
        let Some(Ok(WatchEvent::Added(second))) = watch.next().await else {
            panic!("expected Added");
        };
        assert_eq!(second.model.as_deref(), Some("m2"));
        assert!(second.revision > first.revision);

        // An unchanged heartbeat publishes nothing; MCS dropping the Relay
        // removes the pool at once, though it is still registered.
        fx.register("mocker", "m2").await;
        fx.mcs.unregister(&key("mocker"), &fx.caller).await.unwrap();
        assert!(
            matches!(watch.next().await, Some(Ok(WatchEvent::Removed(k))) if k == key("mocker"))
        );
        assert!(fx.keys().await.is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn a_lapsed_registration_drops_a_pool_mcs_still_lists() {
        let fx = fixture(LEASE).await;
        fx.in_mcs("mocker").await;
        fx.register("mocker", "m1").await;
        let mut watch = fx.directory.list_and_watch(None).await.unwrap();
        assert!(matches!(watch.next().await, Some(Ok(WatchEvent::Added(_)))));
        assert!(matches!(
            watch.next().await,
            Some(Ok(WatchEvent::Resync(_)))
        ));

        // No heartbeat for longer than the lease, as when the pool's cluster
        // is cut off but its imported slices still say Ready. The pool goes
        // within one sweep.
        tokio::time::advance(LEASE.lease_duration + SWEEP_EVERY).await;
        assert!(
            matches!(watch.next().await, Some(Ok(WatchEvent::Removed(k))) if k == key("mocker"))
        );
        assert!(fx.mcs.get(&key("mocker")).await.unwrap().is_some());
        assert!(fx.keys().await.is_empty());

        // Registering again brings it back.
        fx.register("mocker", "m1").await;
        assert!(matches!(watch.next().await, Some(Ok(WatchEvent::Added(_)))));
    }

    #[tokio::test]
    async fn a_relist_forgets_pools_neither_source_lists() {
        let snapshot = |items| {
            WatchEvent::Resync(Snapshot {
                items,
                completeness: Completeness::Exact,
            })
        };
        let now = Instant::now();
        let mut state = JoinState::default();
        state.apply(Source::Mcs, snapshot(Vec::new()), now);
        state.apply(
            Source::Registrations,
            snapshot(vec![registered("gone", "m")]),
            now,
        );
        assert!(state.one_sided.contains_key(&key("gone")));
        state.apply(Source::Registrations, snapshot(Vec::new()), now);
        assert!(state.one_sided.is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn warns_once_about_pools_only_one_source_lists() {
        let fx = fixture(NO_EXPIRY).await;
        fx.in_mcs("relay-only").await;
        fx.register("registered-only", "m").await;
        fx.in_mcs("both").await;
        fx.register("both", "m").await;
        fx.directory.list().await.unwrap();
        settle().await;
        let warned = |fx: &Fixture| {
            let mut warned: Vec<(String, bool)> = fx
                .directory
                .join
                .lock()
                .one_sided
                .iter()
                .map(|(key, one_sided)| (key.dgd_name().to_owned(), one_sided.warned))
                .collect();
            warned.sort();
            warned
        };
        assert_eq!(
            warned(&fx),
            [
                ("registered-only".to_owned(), false),
                ("relay-only".to_owned(), false)
            ]
        );
        tokio::time::advance(LEASE.lease_duration * 2 + LEASE.heartbeat_interval).await;
        settle().await;
        assert_eq!(
            warned(&fx),
            [
                ("registered-only".to_owned(), true),
                ("relay-only".to_owned(), true)
            ]
        );

        // Once both sources list it, the pool is no longer one-sided.
        fx.register("relay-only", "m").await;
        settle().await;
        assert_eq!(warned(&fx), [("registered-only".to_owned(), true)]);
    }
}
