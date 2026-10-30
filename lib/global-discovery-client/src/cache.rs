// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-memory copies of the service's lists, refreshed on an interval:
//! [`RouterCache`] for `GET /routers` and [`PoolCache`] for `GET /pools`. A
//! failed refresh keeps the last list and marks it stale, so a discovery
//! outage does not drop what a caller already knows.

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::future::Future;
use std::time::Duration;

use dynamo_global_common::{PoolKey, PoolListing, PoolsResponse, RouterReplica};
use tokio::sync::watch;
use tokio::time::{Interval, MissedTickBehavior};
use tokio_util::sync::CancellationToken;
use tokio_util::task::AbortOnDropHandle;

use crate::{DiscoveryClientError, GlobalDiscoveryClient};

/// Whether a cache holds the service's current list. A cache that has stopped
/// keeps the value from its last refresh, so check that its channel is open
/// before trusting `Current`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Freshness {
    /// No refresh has finished yet, so the list is empty.
    Unread,
    /// The last refresh succeeded.
    Current,
    /// The last refresh failed with this error. The cache keeps the last list.
    Stale(String),
}

/// The Global Router replicas from `GET /routers`, refreshed until cancelled
/// or dropped.
#[derive(Debug)]
#[must_use = "dropping the cache stops its refresh task"]
pub struct RouterCache {
    routers: watch::Receiver<HashSet<RouterReplica>>,
    freshness: watch::Receiver<Freshness>,
    _task: AbortOnDropHandle<()>,
}

impl RouterCache {
    /// Reads `GET /routers` immediately, then every `interval`, until `cancel`
    /// fires or the cache is dropped. The set starts empty, so nothing waits
    /// on the first read. Call it inside a Tokio runtime.
    ///
    /// # Panics
    ///
    /// If `interval` is zero.
    pub fn spawn(
        client: GlobalDiscoveryClient,
        interval: Duration,
        cancel: CancellationToken,
    ) -> Self {
        Self::spawn_from(client, interval, cancel)
    }

    fn spawn_from(
        source: impl RouterSource + 'static,
        interval: Duration,
        cancel: CancellationToken,
    ) -> Self {
        let mut ticks = ticks(interval);
        let (routers, routers_rx) = watch::channel(HashSet::new());
        let (freshness, freshness_rx) = watch::channel(Freshness::Unread);
        let task = tokio::spawn(async move {
            cancel
                .run_until_cancelled(async {
                    loop {
                        ticks.tick().await;
                        refresh_routers(&source, &routers, &freshness).await;
                    }
                })
                .await;
        });
        Self {
            routers: routers_rx,
            freshness: freshness_rx,
            _task: AbortOnDropHandle::new(task),
        }
    }

    /// The current set. A refresh that lists the same routers publishes
    /// nothing. The channel closes when the cache stops.
    pub fn subscribe(&self) -> watch::Receiver<HashSet<RouterReplica>> {
        self.routers.clone()
    }

    /// Whether [`Self::subscribe`] holds the service's current list.
    pub fn freshness(&self) -> Freshness {
        self.freshness.borrow().clone()
    }
}

/// What a [`RouterCache`] reads. The client implements it; tests use a fake.
trait RouterSource: Send + Sync {
    fn routers(
        &self,
    ) -> impl Future<Output = Result<Vec<RouterReplica>, DiscoveryClientError>> + Send;
}

impl RouterSource for GlobalDiscoveryClient {
    fn routers(
        &self,
    ) -> impl Future<Output = Result<Vec<RouterReplica>, DiscoveryClientError>> + Send {
        GlobalDiscoveryClient::routers(self)
    }
}

async fn refresh_routers(
    source: &impl RouterSource,
    routers: &watch::Sender<HashSet<RouterReplica>>,
    freshness: &watch::Sender<Freshness>,
) {
    let listed = match source.routers().await {
        Ok(listed) => listed.into_iter().collect::<HashSet<_>>(),
        Err(error) => {
            let reason = error_chain(&error);
            if mark_stale(freshness, &reason) {
                let known = routers.borrow().len();
                tracing::warn!(error = %reason, known, "cannot list Global Routers; keeping the known ones");
            }
            return;
        }
    };
    // Publish the set before marking it current, so a reader never sees
    // `Current` next to the previous set.
    let mut added = Vec::new();
    let mut removed = Vec::new();
    routers.send_if_modified(|current| {
        if *current == listed {
            return false;
        }
        added = described(listed.difference(current));
        removed = described(current.difference(&listed));
        *current = listed;
        true
    });
    if mark_current(freshness) {
        tracing::info!("listing Global Routers again");
    }
    if !added.is_empty() || !removed.is_empty() {
        tracing::info!(?added, ?removed, "Global Router set changed");
    }
}

/// The pools from `GET /pools` by key, refreshed until cancelled or dropped.
///
/// The service may list only some pools (`complete: false`), for example while
/// it starts. A complete list replaces the known pools. A partial list adds new
/// pools and updates changed ones, but keeps known pools that are missing, so
/// pools are removed only on a complete list. A cache that knows no pools
/// takes a partial list too: some pools beat none.
#[derive(Debug)]
#[must_use = "dropping the cache stops its refresh task"]
pub struct PoolCache {
    pools: watch::Receiver<HashMap<PoolKey, PoolListing>>,
    freshness: watch::Receiver<Freshness>,
    _task: AbortOnDropHandle<()>,
}

impl PoolCache {
    /// Reads `GET /pools` immediately, then every `interval`, until `cancel`
    /// fires or the cache is dropped. The pools start empty, so nothing waits
    /// on the first read. Call it inside a Tokio runtime.
    ///
    /// # Panics
    ///
    /// If `interval` is zero.
    pub fn spawn(
        client: GlobalDiscoveryClient,
        interval: Duration,
        cancel: CancellationToken,
    ) -> Self {
        Self::spawn_from(client, interval, cancel)
    }

    fn spawn_from(
        source: impl PoolSource + 'static,
        interval: Duration,
        cancel: CancellationToken,
    ) -> Self {
        let mut ticks = ticks(interval);
        let (pools, pools_rx) = watch::channel(HashMap::new());
        let (freshness, freshness_rx) = watch::channel(Freshness::Unread);
        let task = tokio::spawn(async move {
            // Assumed complete, so a first list that is partial is logged.
            let mut complete = true;
            cancel
                .run_until_cancelled(async {
                    loop {
                        ticks.tick().await;
                        refresh_pools(&source, &pools, &freshness, &mut complete).await;
                    }
                })
                .await;
        });
        Self {
            pools: pools_rx,
            freshness: freshness_rx,
            _task: AbortOnDropHandle::new(task),
        }
    }

    /// The known pools. A refresh that adds, updates and removes nothing
    /// publishes nothing. The channel closes when the cache stops.
    pub fn subscribe(&self) -> watch::Receiver<HashMap<PoolKey, PoolListing>> {
        self.pools.clone()
    }

    /// Whether the last refresh succeeded. A partial list is a success: it is
    /// what the service knows.
    pub fn freshness(&self) -> Freshness {
        self.freshness.borrow().clone()
    }
}

/// What a [`PoolCache`] reads. The client implements it; tests use a fake.
trait PoolSource: Send + Sync {
    fn pools(&self) -> impl Future<Output = Result<PoolsResponse, DiscoveryClientError>> + Send;
}

impl PoolSource for GlobalDiscoveryClient {
    fn pools(&self) -> impl Future<Output = Result<PoolsResponse, DiscoveryClientError>> + Send {
        GlobalDiscoveryClient::pools(self)
    }
}

/// `was_complete` is whether the last list read was complete. It starts
/// `true`, so a partial first list is logged.
async fn refresh_pools(
    source: &impl PoolSource,
    pools: &watch::Sender<HashMap<PoolKey, PoolListing>>,
    freshness: &watch::Sender<Freshness>,
    was_complete: &mut bool,
) {
    let PoolsResponse {
        pools: listed,
        complete,
    } = match source.pools().await {
        Ok(response) => response,
        Err(error) => {
            let reason = error_chain(&error);
            if mark_stale(freshness, &reason) {
                let known = pools.borrow().len();
                tracing::warn!(error = %reason, known, "cannot list pools; keeping the known ones");
            }
            return;
        }
    };
    let mut added = Vec::new();
    let mut updated = Vec::new();
    let mut removed = Vec::new();
    let changed = pools.send_if_modified(|known| {
        // A partial list can miss a pool that still exists, so only a
        // complete one shows that a missing pool is gone.
        if complete {
            let listed_keys = listed.iter().map(|pool| &pool.key).collect::<HashSet<_>>();
            known.retain(|key, _| {
                let kept = listed_keys.contains(key);
                if !kept {
                    removed.push(key.to_string());
                }
                kept
            });
        }
        // Clones a key only for a new pool. If the service lists a key twice,
        // the last listing wins.
        for pool in listed {
            match known.get_mut(&pool.key) {
                Some(known_pool) => {
                    if *known_pool != pool {
                        updated.push(pool.key.to_string());
                        *known_pool = pool;
                    }
                }
                None => {
                    added.push(pool.key.to_string());
                    known.insert(pool.key.clone(), pool);
                }
            }
        }
        !(added.is_empty() && updated.is_empty() && removed.is_empty())
    });
    // Marked current only once the pools are published, so a reader never
    // sees `Current` next to the previous pools.
    if mark_current(freshness) {
        tracing::info!("listing pools again");
    }
    // Logged when it flips, even if no pool changes: a service whose
    // Kubernetes watch keeps failing serves a partial list for as long as it
    // fails, and the cache keeps any pool removed meanwhile.
    if complete != *was_complete {
        *was_complete = complete;
        if complete {
            tracing::info!("Global Discovery lists every pool; removing the pools it leaves out");
        } else {
            tracing::info!(
                "Global Discovery lists only some pools; keeping the pools it leaves out"
            );
        }
    }
    if changed {
        for names in [&mut added, &mut updated, &mut removed] {
            names.sort();
        }
        tracing::info!(?added, ?updated, ?removed, complete, "pool list changed");
    }
}

/// Ticks immediately, then every `interval`. A refresh that overruns delays
/// the next tick instead of causing a burst. Built before the task is
/// spawned, so a zero interval panics in the caller.
fn ticks(interval: Duration) -> Interval {
    assert!(!interval.is_zero(), "the refresh interval must be non-zero");
    let mut ticks = tokio::time::interval(interval);
    ticks.set_missed_tick_behavior(MissedTickBehavior::Delay);
    ticks
}

/// Records a failed refresh. Returns whether `reason` differs from the last
/// one, so a caller logs each new reason once instead of every interval.
fn mark_stale(freshness: &watch::Sender<Freshness>, reason: &str) -> bool {
    freshness.send_if_modified(|current| {
        if matches!(current, Freshness::Stale(last) if last == reason) {
            return false;
        }
        *current = Freshness::Stale(reason.to_owned());
        true
    })
}

/// Records a successful refresh. Returns whether it ends an outage, so a
/// caller logs the recovery once. Call it after publishing the list, so a
/// reader never sees `Current` next to the previous list.
fn mark_current(freshness: &watch::Sender<Freshness>) -> bool {
    let mut recovered = false;
    freshness.send_if_modified(|current| {
        if *current == Freshness::Current {
            return false;
        }
        recovered = matches!(current, Freshness::Stale(_));
        *current = Freshness::Current;
        true
    });
    recovered
}

/// Each router as `<id>=<url>`, sorted, so a URL change under the same ID
/// shows in the log.
fn described<'a>(routers: impl Iterator<Item = &'a RouterReplica>) -> Vec<String> {
    let mut described = routers
        .map(|router| format!("{}={}", router.replica.0, router.url))
        .collect::<Vec<_>>();
    described.sort();
    described
}

/// The error and each of its causes, joined with `: `.
fn error_chain(error: &dyn Error) -> String {
    let mut chain = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        chain.push_str(": ");
        chain.push_str(&cause.to_string());
        source = cause.source();
    }
    chain
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use dynamo_global_common::{Capacity, PoolLocation, RouterReplicaId};
    use reqwest::StatusCode;

    use super::*;

    /// A source whose answer the test sets; `None` is an outage.
    #[derive(Clone, Default)]
    struct FakeSource(Arc<Mutex<Option<Vec<RouterReplica>>>>);

    impl FakeSource {
        fn answer(&self, ids: Option<&[&str]>) {
            *self.0.lock().unwrap() = ids.map(|ids| ids.iter().map(|id| router(id)).collect());
        }
    }

    impl RouterSource for FakeSource {
        async fn routers(&self) -> Result<Vec<RouterReplica>, DiscoveryClientError> {
            self.0
                .lock()
                .unwrap()
                .clone()
                .ok_or_else(|| DiscoveryClientError::Status {
                    url: "http://discovery/routers".parse().unwrap(),
                    status: StatusCode::SERVICE_UNAVAILABLE,
                    body: "discovery is down".to_owned(),
                })
        }
    }

    fn router(id: &str) -> RouterReplica {
        RouterReplica {
            replica: RouterReplicaId(id.to_owned()),
            url: format!("http://{id}:9000/").parse().unwrap(),
        }
    }

    fn ids<'a>(routers: impl Iterator<Item = &'a RouterReplica>) -> Vec<String> {
        let mut ids = routers
            .map(|router| router.replica.0.clone())
            .collect::<Vec<_>>();
        ids.sort();
        ids
    }

    #[test]
    fn error_chain_joins_every_cause() {
        #[derive(Debug, thiserror::Error)]
        #[error("request failed")]
        struct Outer(#[source] std::io::Error);
        let error = Outer(std::io::Error::new(
            std::io::ErrorKind::ConnectionRefused,
            "connection refused",
        ));
        assert_eq!(error_chain(&error), "request failed: connection refused");
    }

    async fn next_set(routers: &mut watch::Receiver<HashSet<RouterReplica>>) -> Vec<String> {
        tokio::time::timeout(Duration::from_secs(5), routers.changed())
            .await
            .expect("the router set did not change")
            .unwrap();
        ids(routers.borrow_and_update().iter())
    }

    #[tokio::test(start_paused = true)]
    async fn keeps_routers_through_an_outage_and_reports_them_stale() {
        let source = FakeSource::default();
        source.answer(Some(&["a"]));
        let cancel = CancellationToken::new();
        let cache = RouterCache::spawn_from(source.clone(), Duration::from_secs(1), cancel.clone());
        assert_eq!(cache.freshness(), Freshness::Unread);
        let mut routers = cache.subscribe();
        assert_eq!(next_set(&mut routers).await, ["a"]);
        assert_eq!(cache.freshness(), Freshness::Current);

        source.answer(Some(&["a", "b"]));
        assert_eq!(next_set(&mut routers).await, ["a", "b"]);

        // An outage keeps the routers and marks them stale.
        source.answer(None);
        tokio::time::sleep(Duration::from_secs(3)).await;
        assert_eq!(ids(routers.borrow().iter()), ["a", "b"]);
        assert_eq!(
            cache.freshness(),
            Freshness::Stale(
                "http://discovery/routers returned 503 Service Unavailable: discovery is down"
                    .to_owned()
            )
        );

        // Recovery clears it, and an unchanged list publishes nothing.
        source.answer(Some(&["b", "a"]));
        tokio::time::sleep(Duration::from_secs(3)).await;
        assert_eq!(cache.freshness(), Freshness::Current);
        assert!(!routers.has_changed().unwrap());

        source.answer(Some(&["b"]));
        assert_eq!(next_set(&mut routers).await, ["b"]);

        // Cancelling stops the refresh and closes the channel, which stops the
        // broadcaster.
        cancel.cancel();
        let closed = tokio::time::timeout(Duration::from_secs(5), routers.changed()).await;
        assert!(closed.expect("the channel did not close").is_err());
    }

    #[tokio::test(start_paused = true)]
    async fn dropping_the_cache_stops_it() {
        let source = FakeSource::default();
        source.answer(Some(&["a"]));
        let cache = RouterCache::spawn_from(
            source.clone(),
            Duration::from_secs(1),
            CancellationToken::new(),
        );
        let mut routers = cache.subscribe();
        assert_eq!(next_set(&mut routers).await, ["a"]);

        // A subscriber outliving the cache sees its channel close, and the
        // task releases the source.
        drop(cache);
        let closed = tokio::time::timeout(Duration::from_secs(5), routers.changed()).await;
        assert!(closed.expect("the channel did not close").is_err());
        assert_eq!(Arc::strong_count(&source.0), 1);
    }

    #[test]
    #[should_panic(expected = "the refresh interval must be non-zero")]
    fn rejects_a_zero_interval() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        runtime.block_on(async {
            let _cache = RouterCache::spawn_from(
                FakeSource::default(),
                Duration::ZERO,
                CancellationToken::new(),
            );
        });
    }

    /// A pool source whose answer the test sets; `None` is an outage.
    #[derive(Clone, Default)]
    struct FakePools(Arc<Mutex<Option<PoolsResponse>>>);

    impl FakePools {
        /// Lists each pool as `(DGD name, workers)`.
        fn answer(&self, complete: bool, pools: &[(&str, u32)]) {
            let pools = pools
                .iter()
                .map(|&(name, workers)| pool(name, workers))
                .collect();
            *self.0.lock().unwrap() = Some(PoolsResponse { pools, complete });
        }

        fn fail(&self) {
            *self.0.lock().unwrap() = None;
        }
    }

    impl PoolSource for FakePools {
        async fn pools(&self) -> Result<PoolsResponse, DiscoveryClientError> {
            self.0
                .lock()
                .unwrap()
                .clone()
                .ok_or_else(|| DiscoveryClientError::Status {
                    url: "http://discovery/pools".parse().unwrap(),
                    status: StatusCode::SERVICE_UNAVAILABLE,
                    body: "discovery is down".to_owned(),
                })
        }
    }

    fn pool(name: &str, workers: u32) -> PoolListing {
        PoolListing {
            key: PoolKey::new("aws-west", "dynamo", name).unwrap(),
            location: PoolLocation {
                region: "us-west-2".to_owned(),
                availability_zone: None,
                datacenter: None,
            },
            model: Some("Qwen/Qwen3-0.6B".to_owned()),
            capabilities: Vec::new(),
            frontends: Vec::new(),
            capacity: vec![Capacity {
                sku: "H100".to_owned(),
                workers,
            }],
            relay_identities: Vec::new(),
        }
    }

    /// Each pool as `name=workers`, sorted.
    fn pool_workers(pools: &HashMap<PoolKey, PoolListing>) -> Vec<String> {
        let mut pools = pools
            .values()
            .map(|pool| format!("{}={}", pool.key.dgd_name(), pool.capacity[0].workers))
            .collect::<Vec<_>>();
        pools.sort();
        pools
    }

    async fn next_pools(pools: &mut watch::Receiver<HashMap<PoolKey, PoolListing>>) -> Vec<String> {
        tokio::time::timeout(Duration::from_secs(5), pools.changed())
            .await
            .expect("the pools did not change")
            .unwrap();
        pool_workers(&pools.borrow_and_update())
    }

    fn spawn_pool_cache(source: &FakePools) -> PoolCache {
        PoolCache::spawn_from(
            source.clone(),
            Duration::from_secs(1),
            CancellationToken::new(),
        )
    }

    #[tokio::test(start_paused = true)]
    async fn a_complete_list_replaces_the_known_pools() {
        let source = FakePools::default();
        source.answer(true, &[("a", 8), ("b", 8)]);
        let cache = spawn_pool_cache(&source);
        let mut pools = cache.subscribe();
        assert_eq!(next_pools(&mut pools).await, ["a=8", "b=8"]);

        source.answer(true, &[("b", 16), ("c", 8)]);
        assert_eq!(next_pools(&mut pools).await, ["b=16", "c=8"]);
    }

    #[tokio::test(start_paused = true)]
    async fn a_partial_list_removes_nothing_until_a_complete_one() {
        let source = FakePools::default();
        source.answer(true, &[("a", 8), ("b", 8)]);
        let cache = spawn_pool_cache(&source);
        let mut pools = cache.subscribe();
        assert_eq!(next_pools(&mut pools).await, ["a=8", "b=8"]);

        // Adds c and updates a, but keeps b, which may still exist.
        source.answer(false, &[("a", 16), ("c", 8)]);
        assert_eq!(next_pools(&mut pools).await, ["a=16", "b=8", "c=8"]);

        // A partial list with nothing new publishes nothing.
        source.answer(false, &[("c", 8)]);
        tokio::time::sleep(Duration::from_secs(3)).await;
        assert!(!pools.has_changed().unwrap());

        // The next complete list without b removes it.
        source.answer(true, &[("a", 16), ("c", 8)]);
        assert_eq!(next_pools(&mut pools).await, ["a=16", "c=8"]);
    }

    #[tokio::test(start_paused = true)]
    async fn the_last_listing_of_a_key_wins() {
        let source = FakePools::default();
        source.answer(true, &[("a", 8), ("a", 16)]);
        let cache = spawn_pool_cache(&source);
        let mut pools = cache.subscribe();
        assert_eq!(next_pools(&mut pools).await, ["a=16"]);
    }

    #[tokio::test(start_paused = true)]
    async fn an_empty_cache_takes_a_partial_list() {
        let source = FakePools::default();
        source.answer(false, &[("a", 8)]);
        let cache = spawn_pool_cache(&source);
        let mut pools = cache.subscribe();
        assert_eq!(next_pools(&mut pools).await, ["a=8"]);
        assert_eq!(cache.freshness(), Freshness::Current);
    }

    #[tokio::test(start_paused = true)]
    async fn keeps_pools_through_an_outage_and_reports_them_stale() {
        let source = FakePools::default();
        source.answer(true, &[("a", 8)]);
        let cancel = CancellationToken::new();
        let cache = PoolCache::spawn_from(source.clone(), Duration::from_secs(1), cancel.clone());
        assert_eq!(cache.freshness(), Freshness::Unread);
        let mut pools = cache.subscribe();
        assert_eq!(next_pools(&mut pools).await, ["a=8"]);

        source.fail();
        tokio::time::sleep(Duration::from_secs(3)).await;
        assert_eq!(pool_workers(&pools.borrow()), ["a=8"]);
        assert_eq!(
            cache.freshness(),
            Freshness::Stale(
                "http://discovery/pools returned 503 Service Unavailable: discovery is down"
                    .to_owned()
            )
        );

        // Recovery clears it, and an unchanged list publishes nothing.
        source.answer(true, &[("a", 8)]);
        tokio::time::sleep(Duration::from_secs(3)).await;
        assert_eq!(cache.freshness(), Freshness::Current);
        assert!(!pools.has_changed().unwrap());

        // Cancelling stops the refresh and closes the channel.
        cancel.cancel();
        let closed = tokio::time::timeout(Duration::from_secs(5), pools.changed()).await;
        assert!(closed.expect("the channel did not close").is_err());
    }

    #[tokio::test(start_paused = true)]
    async fn a_partial_list_after_an_outage_keeps_the_missing_pools() {
        let source = FakePools::default();
        source.answer(true, &[("a", 8), ("b", 8)]);
        let cache = spawn_pool_cache(&source);
        let mut pools = cache.subscribe();
        assert_eq!(next_pools(&mut pools).await, ["a=8", "b=8"]);

        source.fail();
        tokio::time::sleep(Duration::from_secs(3)).await;
        assert!(matches!(cache.freshness(), Freshness::Stale(_)));

        // A restarted service may list only some pools at first.
        source.answer(false, &[("a", 8)]);
        tokio::time::sleep(Duration::from_secs(3)).await;
        assert_eq!(cache.freshness(), Freshness::Current);
        assert!(!pools.has_changed().unwrap());
        assert_eq!(pool_workers(&pools.borrow()), ["a=8", "b=8"]);
    }

    #[tokio::test(start_paused = true)]
    async fn a_complete_empty_list_removes_every_pool() {
        let source = FakePools::default();
        source.answer(true, &[("a", 8), ("b", 8)]);
        let cache = spawn_pool_cache(&source);
        let mut pools = cache.subscribe();
        assert_eq!(next_pools(&mut pools).await, ["a=8", "b=8"]);

        source.answer(true, &[]);
        assert!(next_pools(&mut pools).await.is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn dropping_the_pool_cache_stops_it() {
        let source = FakePools::default();
        source.answer(true, &[("a", 8)]);
        let cache = spawn_pool_cache(&source);
        let mut pools = cache.subscribe();
        assert_eq!(next_pools(&mut pools).await, ["a=8"]);

        // A subscriber outliving the cache sees its channel close, and the
        // task releases the source.
        drop(cache);
        let closed = tokio::time::timeout(Duration::from_secs(5), pools.changed()).await;
        assert!(closed.expect("the channel did not close").is_err());
        assert_eq!(Arc::strong_count(&source.0), 1);
    }
}
