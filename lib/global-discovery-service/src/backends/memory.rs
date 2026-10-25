// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-memory store: one process keeps every registration. It needs no
//! Kubernetes, so it suits tests and a first deployment. A restarted process
//! starts empty and fills as PoolRelays register again, so its lists are
//! `Estimated`. Router replicas come from configuration until routers
//! register too.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::Duration;

use crate::pool_discovery::watch::{Directory, Snapshot, WatchStream};
use crate::pool_discovery::{
    Completeness, LeasePolicy, PoolDiscoveryError, PoolEvent, PoolRecord, PoolRegistrar,
    Registration, ReplicaEndpoint, ReplicaEvent, ReplicaId, ReplicaRegistrar, Revision,
    VerifiedRelayIdentity, WatchEvent,
};
use async_trait::async_trait;
use dynamo_global_router_shared::PoolKey;
use tokio::sync::broadcast;
use tokio::time::{Instant, MissedTickBehavior};
use tokio_util::sync::CancellationToken;

use super::stream::{EVENT_BUFFER, watch_stream};

/// Pool registrations in memory. A record expires `lease_duration` after its
/// last registration. Reads skip an expired record at once; a sweep removes
/// it, with a `Removed` event, within one `heartbeat_interval`.
///
/// Any authenticated caller may register any pool: V1 assumes a trusted
/// network, and several PoolRelay replicas may register the same pool.
pub struct MemoryPoolStore {
    lease: LeasePolicy,
    pools: Arc<Mutex<Pools>>,
    /// Sent while holding the `pools` lock, so a subscriber that snapshots
    /// and subscribes under it misses no event and sees none twice.
    events: broadcast::Sender<PoolEvent>,
    sweeping: OnceLock<()>,
    stop: CancellationToken,
}

#[derive(Default)]
struct Pools {
    records: HashMap<PoolKey, Registered>,
    last_revision: u64,
}

struct Registered {
    record: PoolRecord,
    expires_at: Instant,
}

impl Pools {
    fn snapshot(&self, now: Instant) -> Snapshot<PoolRecord> {
        Snapshot {
            items: self
                .records
                .values()
                .filter(|registered| registered.expires_at > now)
                .map(|registered| registered.record.clone())
                .collect(),
            completeness: Completeness::Estimated,
        }
    }
}

impl MemoryPoolStore {
    /// Fails unless `heartbeat_interval` is above zero and `lease_duration`
    /// is longer than it.
    pub fn new(lease: LeasePolicy) -> anyhow::Result<Self> {
        check_lease(&lease)?;
        Ok(Self {
            lease,
            pools: Arc::default(),
            events: broadcast::channel(EVENT_BUFFER).0,
            sweeping: OnceLock::new(),
            stop: CancellationToken::new(),
        })
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
                self.lease.heartbeat_interval,
                self.stop.clone(),
            ));
        });
    }
}

impl Drop for MemoryPoolStore {
    fn drop(&mut self) {
        self.stop.cancel();
    }
}

/// `heartbeat_interval` must be above zero and `lease_duration` longer than
/// it, so the sweep has a period and a registration outlives one interval.
fn check_lease(lease: &LeasePolicy) -> anyhow::Result<()> {
    anyhow::ensure!(
        !lease.heartbeat_interval.is_zero() && lease.lease_duration > lease.heartbeat_interval,
        "the lease duration must be longer than the heartbeat interval, which must be above zero"
    );
    Ok(())
}

/// Removes expired records and publishes `Removed` for each.
async fn sweep(
    pools: Arc<Mutex<Pools>>,
    events: broadcast::Sender<PoolEvent>,
    every: Duration,
    stop: CancellationToken,
) {
    let mut ticks = tokio::time::interval(every);
    ticks.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            biased;
            () = stop.cancelled() => return,
            _ = ticks.tick() => {}
        }
        let now = Instant::now();
        let mut pools = pools.lock().unwrap_or_else(PoisonError::into_inner);
        pools.records.retain(|key, registered| {
            let live = registered.expires_at > now;
            if !live {
                // No subscribers is not an error.
                let _ = events.send(WatchEvent::Removed(key.clone()));
            }
            live
        });
    }
}

#[async_trait]
impl PoolRegistrar for MemoryPoolStore {
    /// An unchanged record only renews its lease. A new or changed one gets
    /// the next revision and an `Added` event.
    async fn register(
        &self,
        mut record: PoolRecord,
        _caller: &VerifiedRelayIdentity,
    ) -> Result<Registration, PoolDiscoveryError> {
        self.ensure_sweeping();
        let now = Instant::now();
        let expires_at = now + self.lease.lease_duration;
        let mut pools = self.lock();
        if let Some(registered) = pools.records.get_mut(&record.key)
            && registered.expires_at > now
        {
            record.revision = registered.record.revision;
            if registered.record == record {
                registered.expires_at = expires_at;
                return Ok(Registration { lease: self.lease });
            }
        }
        pools.last_revision += 1;
        record.revision = Revision(pools.last_revision);
        pools.records.insert(
            record.key.clone(),
            Registered {
                record: record.clone(),
                expires_at,
            },
        );
        // No subscribers is not an error.
        let _ = self.events.send(WatchEvent::Added(record));
        Ok(Registration { lease: self.lease })
    }

    async fn unregister(
        &self,
        key: &PoolKey,
        _caller: &VerifiedRelayIdentity,
    ) -> Result<(), PoolDiscoveryError> {
        let mut pools = self.lock();
        if pools.records.remove(key).is_none() {
            return Err(PoolDiscoveryError::UnknownPool(key.clone()));
        }
        let _ = self.events.send(WatchEvent::Removed(key.clone()));
        Ok(())
    }
}

#[async_trait]
impl Directory<PoolRecord, PoolKey> for MemoryPoolStore {
    async fn check_connection(&self) -> Result<(), PoolDiscoveryError> {
        Ok(())
    }

    async fn list(&self) -> Result<Snapshot<PoolRecord>, PoolDiscoveryError> {
        Ok(self.lock().snapshot(Instant::now()))
    }

    async fn get(&self, key: &PoolKey) -> Result<Option<PoolRecord>, PoolDiscoveryError> {
        let now = Instant::now();
        Ok(self
            .lock()
            .records
            .get(key)
            .filter(|registered| registered.expires_at > now)
            .map(|registered| registered.record.clone()))
    }

    async fn list_and_watch(
        &self,
        cancel_token: Option<CancellationToken>,
    ) -> Result<WatchStream<PoolRecord, PoolKey>, PoolDiscoveryError> {
        self.ensure_sweeping();
        let pools = Arc::clone(&self.pools);
        let events = self.events.clone();
        Ok(watch_stream(
            move || {
                let pools = pools.lock().unwrap_or_else(PoisonError::into_inner);
                (events.subscribe(), Some(pools.snapshot(Instant::now())))
            },
            cancel_token.unwrap_or_default(),
            self.stop.clone(),
        ))
    }
}

/// Router replicas from configuration. The set never changes, and lists are
/// `Exact`.
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
    async fn check_connection(&self) -> Result<(), PoolDiscoveryError> {
        Ok(())
    }

    async fn list(&self) -> Result<Snapshot<ReplicaEndpoint>, PoolDiscoveryError> {
        Ok(self.snapshot())
    }

    async fn get(&self, key: &ReplicaId) -> Result<Option<ReplicaEndpoint>, PoolDiscoveryError> {
        Ok(self
            .replicas
            .iter()
            .find(|replica| &replica.replica == key)
            .cloned())
    }

    async fn list_and_watch(
        &self,
        cancel_token: Option<CancellationToken>,
    ) -> Result<WatchStream<ReplicaEndpoint, ReplicaId>, PoolDiscoveryError> {
        let events = self.events.clone();
        let snapshot = self.snapshot();
        Ok(watch_stream(
            move || (events.subscribe(), Some(snapshot.clone())),
            cancel_token.unwrap_or_default(),
            self.stop.clone(),
        ))
    }
}

/// Router replicas in memory: the ones from configuration, and the ones
/// that register (`POST /routers`). A registration expires `lease_duration`
/// after its last renewal, and while it lasts it replaces a configured
/// replica with the same ID. A restarted process lists only the configured
/// replicas until the others register again, so lists are `Estimated`.
///
/// Any authenticated caller may register any replica ID: like
/// [`MemoryPoolStore`], V1 assumes a trusted network.
pub struct MemoryReplicaStore {
    lease: LeasePolicy,
    replicas: Arc<Mutex<Replicas>>,
    /// Sent while holding the `replicas` lock, as in [`MemoryPoolStore`].
    events: broadcast::Sender<ReplicaEvent>,
    sweeping: OnceLock<()>,
    stop: CancellationToken,
}

struct Replicas {
    configured: HashMap<ReplicaId, ReplicaEndpoint>,
    registered: HashMap<ReplicaId, RegisteredReplica>,
}

struct RegisteredReplica {
    endpoint: ReplicaEndpoint,
    expires_at: Instant,
}

impl Replicas {
    /// A live registration, else the configured replica.
    fn served(&self, id: &ReplicaId, now: Instant) -> Option<&ReplicaEndpoint> {
        self.registered
            .get(id)
            .filter(|registered| registered.expires_at > now)
            .map(|registered| &registered.endpoint)
            .or_else(|| self.configured.get(id))
    }

    fn snapshot(&self, now: Instant) -> Snapshot<ReplicaEndpoint> {
        let registered = |id: &ReplicaId| {
            self.registered
                .get(id)
                .is_some_and(|registered| registered.expires_at > now)
        };
        Snapshot {
            items: self
                .registered
                .values()
                .filter(|registered| registered.expires_at > now)
                .map(|registered| registered.endpoint.clone())
                .chain(
                    self.configured
                        .values()
                        .filter(|endpoint| !registered(&endpoint.replica))
                        .cloned(),
                )
                .collect(),
            completeness: Completeness::Estimated,
        }
    }
}

impl MemoryReplicaStore {
    /// `configured` replicas are listed until shutdown. Fails, like
    /// [`MemoryPoolStore::new`], for a lease without room for heartbeats.
    pub fn new(configured: Vec<ReplicaEndpoint>, lease: LeasePolicy) -> anyhow::Result<Self> {
        check_lease(&lease)?;
        let configured = configured
            .into_iter()
            .map(|endpoint| (endpoint.replica.clone(), endpoint))
            .collect();
        Ok(Self {
            lease,
            replicas: Arc::new(Mutex::new(Replicas {
                configured,
                registered: HashMap::new(),
            })),
            events: broadcast::channel(EVENT_BUFFER).0,
            sweeping: OnceLock::new(),
            stop: CancellationToken::new(),
        })
    }

    fn lock(&self) -> MutexGuard<'_, Replicas> {
        self.replicas.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Every trait method runs in a Tokio runtime, so the spawn is safe.
    fn ensure_sweeping(&self) {
        self.sweeping.get_or_init(|| {
            tokio::spawn(sweep_replicas(
                Arc::clone(&self.replicas),
                self.events.clone(),
                self.lease.heartbeat_interval,
                self.stop.clone(),
            ));
        });
    }
}

impl Drop for MemoryReplicaStore {
    fn drop(&mut self) {
        self.stop.cancel();
    }
}

/// Removes expired registrations. A configured replica goes back to its
/// configured address; any other replica is `Removed`.
async fn sweep_replicas(
    replicas: Arc<Mutex<Replicas>>,
    events: broadcast::Sender<ReplicaEvent>,
    every: Duration,
    stop: CancellationToken,
) {
    let mut ticks = tokio::time::interval(every);
    ticks.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            biased;
            () = stop.cancelled() => return,
            _ = ticks.tick() => {}
        }
        let now = Instant::now();
        let mut replicas = replicas.lock().unwrap_or_else(PoisonError::into_inner);
        let Replicas {
            configured,
            registered,
        } = &mut *replicas;
        registered.retain(|id, registered| {
            let live = registered.expires_at > now;
            if !live {
                // No subscribers is not an error.
                match configured.get(id) {
                    None => {
                        let _ = events.send(WatchEvent::Removed(id.clone()));
                    }
                    Some(endpoint) if *endpoint != registered.endpoint => {
                        let _ = events.send(WatchEvent::Added(endpoint.clone()));
                    }
                    Some(_) => {}
                }
            }
            live
        });
    }
}

#[async_trait]
impl ReplicaRegistrar for MemoryReplicaStore {
    /// An unchanged registration only renews its lease. A new or changed one
    /// is an `Added` event.
    async fn register(
        &self,
        endpoint: ReplicaEndpoint,
    ) -> Result<Registration, PoolDiscoveryError> {
        self.ensure_sweeping();
        let now = Instant::now();
        let mut replicas = self.lock();
        let changed = replicas.served(&endpoint.replica, now) != Some(&endpoint);
        replicas.registered.insert(
            endpoint.replica.clone(),
            RegisteredReplica {
                endpoint: endpoint.clone(),
                expires_at: now + self.lease.lease_duration,
            },
        );
        if changed {
            // No subscribers is not an error.
            let _ = self.events.send(WatchEvent::Added(endpoint));
        }
        Ok(Registration { lease: self.lease })
    }
}

#[async_trait]
impl Directory<ReplicaEndpoint, ReplicaId> for MemoryReplicaStore {
    async fn check_connection(&self) -> Result<(), PoolDiscoveryError> {
        Ok(())
    }

    async fn list(&self) -> Result<Snapshot<ReplicaEndpoint>, PoolDiscoveryError> {
        Ok(self.lock().snapshot(Instant::now()))
    }

    async fn get(&self, key: &ReplicaId) -> Result<Option<ReplicaEndpoint>, PoolDiscoveryError> {
        Ok(self.lock().served(key, Instant::now()).cloned())
    }

    async fn list_and_watch(
        &self,
        cancel_token: Option<CancellationToken>,
    ) -> Result<WatchStream<ReplicaEndpoint, ReplicaId>, PoolDiscoveryError> {
        self.ensure_sweeping();
        let replicas = Arc::clone(&self.replicas);
        let events = self.events.clone();
        Ok(watch_stream(
            move || {
                let replicas = replicas.lock().unwrap_or_else(PoisonError::into_inner);
                (events.subscribe(), Some(replicas.snapshot(Instant::now())))
            },
            cancel_token.unwrap_or_default(),
            self.stop.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::pool_discovery::{
        PeerCredentials, PoolRecordBuilder, RelayAuthenticator, RelayIdentity, verify,
    };
    use dynamo_global_router_shared::PoolLocation;
    use futures::StreamExt;
    use url::Url;

    use super::*;

    const LEASE: LeasePolicy = LeasePolicy {
        heartbeat_interval: Duration::from_secs(5),
        lease_duration: Duration::from_secs(15),
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

    fn record(dgd: &str, model: &str) -> PoolRecord {
        PoolRecordBuilder::default()
            .key(key(dgd))
            .location(PoolLocation {
                region: "us-west-2".to_owned(),
                availability_zone: None,
                cluster: Some("west".to_owned()),
                datacenter: None,
            })
            .model(Some(model.to_owned()))
            .build()
            .unwrap()
    }

    #[tokio::test(start_paused = true)]
    async fn registering_again_renews_without_a_new_revision() {
        let store = MemoryPoolStore::new(LEASE).unwrap();
        let caller = caller().await;
        let registration = store
            .register(record("mocker", "m1"), &caller)
            .await
            .unwrap();
        assert_eq!(registration.lease, LEASE);
        let first = store.get(&key("mocker")).await.unwrap().unwrap();
        assert_eq!(first.revision, Revision(1));

        // Heartbeats every 10 s keep the 15 s lease alive past 15 s.
        for _ in 0..3 {
            tokio::time::advance(Duration::from_secs(10)).await;
            store
                .register(record("mocker", "m1"), &caller)
                .await
                .unwrap();
        }
        let renewed = store.get(&key("mocker")).await.unwrap().unwrap();
        assert_eq!(renewed.revision, Revision(1));

        // A changed record is a new revision.
        store
            .register(record("mocker", "m2"), &caller)
            .await
            .unwrap();
        let changed = store.get(&key("mocker")).await.unwrap().unwrap();
        assert_eq!(changed.revision, Revision(2));
        assert_eq!(changed.model.as_deref(), Some("m2"));
    }

    #[tokio::test(start_paused = true)]
    async fn a_missed_lease_hides_then_removes_the_pool() {
        let store = MemoryPoolStore::new(LEASE).unwrap();
        let caller = caller().await;
        store
            .register(record("mocker", "m1"), &caller)
            .await
            .unwrap();
        let mut watch = store.list_and_watch(None).await.unwrap();
        assert!(matches!(watch.next().await, Some(Ok(WatchEvent::Added(_)))));
        assert!(matches!(
            watch.next().await,
            Some(Ok(WatchEvent::Resync(_)))
        ));

        tokio::time::advance(Duration::from_secs(16)).await;
        // Reads skip it before the sweep runs.
        assert!(store.get(&key("mocker")).await.unwrap().is_none());
        assert!(store.list().await.unwrap().items.is_empty());
        // The sweep removes it and tells watchers.
        assert!(
            matches!(watch.next().await, Some(Ok(WatchEvent::Removed(k))) if k == key("mocker"))
        );
        assert!(store.lock().records.is_empty());

        // Registering again brings it back as a new revision.
        store
            .register(record("mocker", "m1"), &caller)
            .await
            .unwrap();
        assert!(
            matches!(watch.next().await, Some(Ok(WatchEvent::Added(r))) if r.revision == Revision(2))
        );
    }

    #[tokio::test]
    async fn unregister_removes_the_pool() {
        let store = MemoryPoolStore::new(LEASE).unwrap();
        let caller = caller().await;
        store
            .register(record("mocker", "m1"), &caller)
            .await
            .unwrap();
        store.unregister(&key("mocker"), &caller).await.unwrap();
        assert!(store.list().await.unwrap().items.is_empty());
        assert!(matches!(
            store.unregister(&key("mocker"), &caller).await,
            Err(PoolDiscoveryError::UnknownPool(_))
        ));
    }

    #[tokio::test]
    async fn lists_are_estimated() {
        let store = MemoryPoolStore::new(LEASE).unwrap();
        store
            .register(record("mocker", "m1"), &caller().await)
            .await
            .unwrap();
        let snapshot = store.list().await.unwrap();
        assert_eq!(snapshot.items.len(), 1);
        assert_eq!(snapshot.completeness, Completeness::Estimated);
    }

    #[test]
    fn rejects_a_lease_without_room_for_heartbeats() {
        for (heartbeat_interval, lease_duration) in [(0, 15), (5, 5), (10, 5)] {
            let lease = LeasePolicy {
                heartbeat_interval: Duration::from_secs(heartbeat_interval),
                lease_duration: Duration::from_secs(lease_duration),
            };
            assert!(
                MemoryPoolStore::new(lease).is_err(),
                "{heartbeat_interval}s / {lease_duration}s"
            );
            assert!(
                MemoryReplicaStore::new(Vec::new(), lease).is_err(),
                "{heartbeat_interval}s / {lease_duration}s"
            );
        }
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
        assert_eq!(
            directory.get(&replica.replica).await.unwrap(),
            Some(replica.clone())
        );

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

    fn router(id: &str, url: &str) -> ReplicaEndpoint {
        ReplicaEndpoint {
            replica: ReplicaId(id.to_owned()),
            url: Url::parse(url).unwrap(),
        }
    }

    #[tokio::test(start_paused = true)]
    async fn a_registered_router_is_listed_until_its_lease_ends() {
        let store = MemoryReplicaStore::new(Vec::new(), LEASE).unwrap();
        let replica = router("ohio/global-router-0", "http://10.42.0.5:50051");
        let mut watch = store.list_and_watch(None).await.unwrap();
        assert!(
            matches!(watch.next().await, Some(Ok(WatchEvent::Resync(s))) if s.items.is_empty())
        );

        let registration = store.register(replica.clone()).await.unwrap();
        assert_eq!(registration.lease, LEASE);
        assert!(matches!(watch.next().await, Some(Ok(WatchEvent::Added(r))) if r == replica));

        // Heartbeats every 10 s keep the 15 s lease alive; renewing an
        // unchanged registration is not a change.
        for _ in 0..3 {
            tokio::time::advance(Duration::from_secs(10)).await;
            store.register(replica.clone()).await.unwrap();
        }
        assert_eq!(
            store.list().await.unwrap().items,
            std::slice::from_ref(&replica)
        );

        // Without heartbeats it ends: reads skip it, then the sweep removes it.
        tokio::time::advance(Duration::from_secs(16)).await;
        assert!(store.get(&replica.replica).await.unwrap().is_none());
        assert!(
            matches!(watch.next().await, Some(Ok(WatchEvent::Removed(id))) if id == replica.replica)
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_registration_replaces_a_configured_router_while_it_lasts() {
        let configured = router("ohio/global-router-0", "http://10.42.0.5:50051");
        let moved = router("ohio/global-router-0", "http://10.42.0.9:50051");
        let other = router("west/global-router-0", "http://10.43.0.5:50051");
        let store = MemoryReplicaStore::new(vec![configured.clone()], LEASE).unwrap();
        let mut watch = store.list_and_watch(None).await.unwrap();
        assert!(matches!(watch.next().await, Some(Ok(WatchEvent::Added(r))) if r == configured));
        assert!(matches!(
            watch.next().await,
            Some(Ok(WatchEvent::Resync(_)))
        ));

        // Registering the configured address is not a change.
        store.register(configured.clone()).await.unwrap();
        store.register(moved.clone()).await.unwrap();
        store.register(other.clone()).await.unwrap();
        assert!(matches!(watch.next().await, Some(Ok(WatchEvent::Added(r))) if r == moved));
        assert!(matches!(watch.next().await, Some(Ok(WatchEvent::Added(r))) if r == other));
        assert_eq!(
            store.get(&configured.replica).await.unwrap(),
            Some(moved.clone())
        );

        // When the registrations end, the configured address comes back and
        // the other replica goes.
        tokio::time::advance(Duration::from_secs(16)).await;
        let mut changes = Vec::new();
        for _ in 0..2 {
            changes.push(watch.next().await.unwrap().unwrap());
        }
        assert!(
            changes
                .iter()
                .any(|event| matches!(event, WatchEvent::Added(r) if *r == configured))
        );
        assert!(
            changes
                .iter()
                .any(|event| matches!(event, WatchEvent::Removed(id) if *id == other.replica))
        );
        assert_eq!(store.list().await.unwrap().items, [configured]);
    }
}
