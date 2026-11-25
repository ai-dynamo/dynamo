// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One EndpointSlice watch feeding a derived index, shared by the pool and
//! router replica directories. Each slice event updates only what that slice
//! affects; a relist rebuilds the index once, at `InitDone`.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, OnceLock, PoisonError, RwLock, RwLockReadGuard};
use std::time::Duration;

use crate::pool_discovery::{
    Completeness, Directory, DiscoveryError, Snapshot, WatchEvent, WatchStream,
};
use async_trait::async_trait;
use futures::{Stream, StreamExt};
use k8s_openapi::api::discovery::v1::{Endpoint, EndpointSlice};
use kube::Api;
use kube::api::ListParams;
use kube::runtime::{WatchStreamExt, watcher};
use tokio::sync::broadcast;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use super::super::stream::{EVENT_BUFFER, wake_at, watch_stream};

/// How long the watch may keep failing before the cache reports its last view
/// as `Estimated`. It keeps serving that view, and the service stays ready.
pub(super) const STALE_AFTER: Duration = Duration::from_secs(60);

/// How long without an error before a failing watch counts as working again,
/// since kube resumes a watch without an event. While it fails, errors come
/// at least every 30 s of backoff (60 s should kube ever apply its jitter)
/// plus up to 85 s for an attempt to fail within the client's 10 s connect
/// and 75 s read timeouts.
pub(super) const RECOVERED_AFTER: Duration = Duration::from_secs(180);

/// How long the view stays `Estimated` after a pool or router leaves it, so
/// readers keep what left until a complete list shows it is still gone. An
/// MCS implementation can drop healthy clusters' imports for a while, as when
/// its registry restarts or a cluster's lease lapses.
pub(super) const REMOVAL_HOLD: Duration = Duration::from_secs(60);

/// Each watch asks the server to end it after this long and starts again, so
/// a server that stops answering fails within the client's read timeout
/// rather than kube's default of nearly five minutes.
pub(super) const WATCH_TIMEOUT_SECS: u32 = 50;

/// Slices are keyed by (namespace, name).
pub(super) type SliceKey = (String, String);

pub(super) fn slice_key(slice: &EndpointSlice) -> SliceKey {
    (
        slice.metadata.namespace.clone().unwrap_or_default(),
        slice.metadata.name.clone().unwrap_or_default(),
    )
}

/// A slice's named TCP port, unless it is 0. One slice lists one set of ports
/// for all its endpoints.
pub(super) fn tcp_port(slice: &EndpointSlice, name: &str) -> Option<u16> {
    slice
        .ports
        .iter()
        .flatten()
        .find(|port| {
            port.name.as_deref() == Some(name)
                && port
                    .protocol
                    .as_deref()
                    .is_none_or(|protocol| protocol == "TCP")
        })
        .and_then(|port| port.port)
        .and_then(|port| u16::try_from(port).ok())
        .filter(|port| *port != 0)
}

/// The slice's Ready endpoints. The EndpointSlice API reads a missing `ready`
/// as ready.
pub(super) fn ready_endpoints(slice: &EndpointSlice) -> impl Iterator<Item = &Endpoint> {
    slice.endpoints.iter().filter(|endpoint| {
        endpoint
            .conditions
            .as_ref()
            .and_then(|conditions| conditions.ready)
            .unwrap_or(true)
    })
}

/// What a directory derives from slices.
pub(super) trait SliceIndex: Default + Send + Sync + 'static {
    type Item: Clone + Send + Sync + 'static;
    type Key: Clone + Send + Sync + 'static;
    /// Turns a slice into an entry.
    type Rules: Send + Sync + 'static;
    /// What one slice contributes.
    type Entry: Send + Sync + 'static;

    /// `None` for a slice this index does not use.
    fn entry(rules: &Self::Rules, slice: &EndpointSlice) -> Option<Self::Entry>;

    /// One slice changed, or went away (`None`). Touches only what that slice
    /// affects and returns the events to publish.
    fn upsert(
        &mut self,
        rules: &Self::Rules,
        key: SliceKey,
        entry: Option<Self::Entry>,
    ) -> Vec<WatchEvent<Self::Item, Self::Key>>;

    /// A relist: every slice at once. The cache publishes one `Resync`.
    /// Returns whether an item listed before is gone.
    fn replace(&mut self, rules: &Self::Rules, entries: HashMap<SliceKey, Self::Entry>) -> bool;

    fn snapshot(&self) -> Snapshot<Self::Item>;

    /// Lists no item.
    fn is_empty(&self) -> bool;
}

pub(super) struct CacheState<I: SliceIndex> {
    pub(super) synced: bool,
    /// The current run of watch errors. `None` while the watch works.
    failures: Option<Failures>,
    /// When an item last left the index, until [`REMOVAL_HOLD`] after.
    removed_at: Option<Instant>,
    /// The completeness watchers last learned from a `Resync`.
    published: Completeness,
    /// Slices listed since the current relist began; they replace the index
    /// at `InitDone`.
    relist: Option<HashMap<SliceKey, I::Entry>>,
    pub(super) index: I,
}

impl<I: SliceIndex> Default for CacheState<I> {
    fn default() -> Self {
        Self {
            synced: false,
            failures: None,
            removed_at: None,
            published: Completeness::Exact,
            relist: None,
            index: I::default(),
        }
    }
}

impl<I: SliceIndex> CacheState<I> {
    /// `Estimated` while the watch keeps failing, for [`REMOVAL_HOLD`] after
    /// an item left, and while nothing is listed: an MCS implementation that
    /// dropped every import looks the same as no pools or routers at all.
    fn completeness(&self, now: Instant) -> Completeness {
        let stale = self.failures.is_some_and(|failures| failures.stale(now));
        let held = self.removed_at.is_some_and(|at| now < at + REMOVAL_HOLD);
        if stale || held || self.index.is_empty() {
            Completeness::Estimated
        } else {
            Completeness::Exact
        }
    }

    /// `None` until the first list completes.
    fn snapshot(&self) -> Option<Snapshot<I::Item>> {
        let mut snapshot = self.synced.then(|| self.index.snapshot())?;
        snapshot.completeness = self.completeness(Instant::now());
        Some(snapshot)
    }

    /// The index as a `Resync`, which tells watchers its completeness. `None`
    /// until the first list completes.
    fn resync(&mut self, now: Instant) -> Option<WatchEvent<I::Item, I::Key>> {
        let mut snapshot = self.synced.then(|| self.index.snapshot())?;
        snapshot.completeness = self.completeness(now);
        self.published = snapshot.completeness;
        Some(WatchEvent::Resync(snapshot))
    }

    /// A `Resync` if the completeness changed since watchers last learned it.
    fn resync_if_changed(&mut self, now: Instant) -> Option<WatchEvent<I::Item, I::Key>> {
        if !self.synced || self.completeness(now) == self.published {
            return None;
        }
        self.resync(now)
    }

    /// Ends the run of errors if it has been quiet for `RECOVERED_AFTER`.
    fn end_quiet_run(&mut self, now: Instant) {
        if self
            .failures
            .is_some_and(|failures| now.saturating_duration_since(failures.last) >= RECOVERED_AFTER)
        {
            self.failures = None;
        }
    }
}

/// Watch errors no more than `RECOVERED_AFTER` apart.
#[derive(Clone, Copy)]
struct Failures {
    first: Instant,
    last: Instant,
}

impl Failures {
    /// The errors span `STALE_AFTER`, so the view turned `Estimated`.
    fn long(&self) -> bool {
        self.last.saturating_duration_since(self.first) >= STALE_AFTER
    }

    /// A long run whose last error is recent. One error followed by silence
    /// is a watch that resumed.
    fn stale(&self, now: Instant) -> bool {
        now.saturating_duration_since(self.last) < RECOVERED_AFTER && self.long()
    }
}

/// The index kept current from slice events. Shared by the watch task, `list`
/// and every `list_and_watch` stream.
pub(super) struct SliceCache<I: SliceIndex> {
    rules: I::Rules,
    state: RwLock<CacheState<I>>,
    /// Sent while holding the `state` write lock. A subscriber that takes a
    /// snapshot and subscribes under the read lock therefore misses no event
    /// and sees none twice.
    pub(super) events: broadcast::Sender<WatchEvent<I::Item, I::Key>>,
}

impl<I: SliceIndex> SliceCache<I> {
    pub(super) fn new(rules: I::Rules) -> Self {
        Self {
            rules,
            state: RwLock::default(),
            events: broadcast::channel(EVENT_BUFFER).0,
        }
    }

    pub(super) fn read(&self) -> RwLockReadGuard<'_, CacheState<I>> {
        self.state.read().unwrap_or_else(PoisonError::into_inner)
    }

    /// `None` until the first list completes.
    pub(super) fn snapshot(&self) -> Option<Snapshot<I::Item>> {
        self.read().snapshot()
    }

    /// Applies one watch event. Returns `true` for a relist that found
    /// nothing.
    pub(super) fn apply(&self, event: watcher::Event<EndpointSlice>) -> bool {
        let now = Instant::now();
        let mut state = self.state.write().unwrap_or_else(PoisonError::into_inner);
        // `Apply`, `Delete` and `InitDone` show that the watch works and the
        // index is current. A relist's slices replace the index only at
        // `InitDone`, so `InitApply` does not end a run of errors, and kube
        // sends `Init` before each relist attempt, before any request.
        if !matches!(event, watcher::Event::Init | watcher::Event::InitApply(_)) {
            state.failures = None;
        }
        let mut listed_nothing = false;
        let mut events = match event {
            watcher::Event::Init => {
                state.relist = Some(HashMap::new());
                Vec::new()
            }
            watcher::Event::InitApply(slice) => {
                // The watch drops `Init` (see `retried`), and a watcher that
                // streams its lists sends none, so the first slice starts the
                // relist.
                if let Some(entry) = I::entry(&self.rules, &slice) {
                    state
                        .relist
                        .get_or_insert_with(HashMap::new)
                        .insert(slice_key(&slice), entry);
                }
                Vec::new()
            }
            watcher::Event::InitDone => {
                let entries = state.relist.take().unwrap_or_default();
                if state.index.replace(&self.rules, entries) {
                    state.removed_at = Some(now);
                }
                state.synced = true;
                listed_nothing = state.index.is_empty();
                // A relist tells watchers everything.
                state.resync(now).into_iter().collect()
            }
            watcher::Event::Apply(slice) => {
                let entry = I::entry(&self.rules, &slice);
                state.index.upsert(&self.rules, slice_key(&slice), entry)
            }
            watcher::Event::Delete(slice) => {
                state.index.upsert(&self.rules, slice_key(&slice), None)
            }
        };
        if events
            .iter()
            .any(|event| matches!(event, WatchEvent::Removed(_)))
        {
            state.removed_at = Some(now);
        }
        // Watchers learn when the view turns `Estimated` or back.
        events.extend(state.resync_if_changed(now));
        for event in events {
            // No subscribers is not an error.
            let _ = self.events.send(event);
        }
        listed_nothing
    }

    /// The watch failed. Returns how long it has been failing.
    pub(super) fn failed(&self) -> Duration {
        let now = Instant::now();
        let mut state = self.state.write().unwrap_or_else(PoisonError::into_inner);
        // A relist that failed starts over, so its slices so far are dropped.
        state.relist = None;
        // A run that has been quiet for `RECOVERED_AFTER` is over, though
        // `recheck` may not have run yet; this error starts a new one.
        state.end_quiet_run(now);
        let failures = match state.failures {
            Some(failures) => Failures {
                last: now,
                ..failures
            },
            None => Failures {
                first: now,
                last: now,
            },
        };
        state.failures = Some(failures);
        // Watchers learn when the view turns `Estimated`.
        if let Some(event) = state.resync_if_changed(now) {
            // No subscribers is not an error.
            let _ = self.events.send(event);
        }
        now.saturating_duration_since(failures.first)
    }

    /// Ends a run of errors that has been quiet for `RECOVERED_AFTER`, as kube
    /// resumes a watch without an event, and a hold that has run out.
    /// Watchers learn if the view is complete again.
    pub(super) fn recheck(&self) {
        let now = Instant::now();
        let mut state = self.state.write().unwrap_or_else(PoisonError::into_inner);
        state.end_quiet_run(now);
        if state.removed_at.is_some_and(|at| now >= at + REMOVAL_HOLD) {
            state.removed_at = None;
        }
        if let Some(event) = state.resync_if_changed(now) {
            // No subscribers is not an error.
            let _ = self.events.send(event);
        }
    }

    /// When the view's completeness can next change without a watch event:
    /// a run of errors falls quiet, or a hold runs out.
    pub(super) fn next_recheck(&self) -> Option<Instant> {
        let state = self.read();
        let quiet = state
            .failures
            .map(|failures| failures.last + RECOVERED_AFTER);
        let held = state.removed_at.map(|at| at + REMOVAL_HOLD);
        quiet.into_iter().chain(held).min()
    }

    /// A watch stream that starts at the first full listing and ends when
    /// `cancel` (the caller's) or `stop` (the directory's) is cancelled.
    pub(super) fn subscribe(
        self: &Arc<Self>,
        cancel: CancellationToken,
        stop: CancellationToken,
    ) -> WatchStream<I::Item, I::Key> {
        let cache = Arc::clone(self);
        watch_stream(
            move || {
                let state = cache.read();
                let receiver = cache.events.subscribe();
                (receiver, state.snapshot())
            },
            cancel,
            stop,
        )
    }
}

/// Single consumer of the slice watch, so the cache applies events in order.
async fn watch_slices<I: SliceIndex>(
    slices: Api<EndpointSlice>,
    selector: String,
    cache: Arc<SliceCache<I>>,
    stop: CancellationToken,
) {
    // Lists come from the API server's watch cache (`resourceVersion=0`), as
    // client-go's reflector does, rather than paged quorum reads from etcd:
    // the watch catches up from there, and discovery needs no more. kube sends
    // `resourceVersion=0` only without a page size, so paging is off.
    let config = watcher::Config {
        page_size: None,
        ..watcher::Config::default()
            .labels(&selector)
            .any_semantic()
            .timeout(WATCH_TIMEOUT_SECS)
    };
    follow(retried(watcher(slices, config)), &selector, &cache, &stop).await;
}

/// Applies `events` to `cache` until they end or `stop` is cancelled.
pub(super) async fn follow<I: SliceIndex>(
    events: impl Stream<Item = watcher::Result<watcher::Event<EndpointSlice>>>,
    selector: &str,
    cache: &SliceCache<I>,
    stop: &CancellationToken,
) {
    let mut events = std::pin::pin!(events);
    // When the view's completeness can change without an event.
    let mut recheck_at = None;
    loop {
        let event = tokio::select! {
            biased;
            () = stop.cancelled() => return,
            () = wake_at(recheck_at) => {
                cache.recheck();
                recheck_at = cache.next_recheck();
                continue;
            }
            event = events.next() => event,
        };
        match event {
            Some(Ok(event)) => {
                if cache.apply(event) {
                    tracing::warn!(
                        %selector,
                        "the EndpointSlices list nothing: check that MCS imports the expected Services into this cluster, under the expected names"
                    );
                }
            }
            // The cache keeps the last good view, and the watcher retries with
            // backoff. The service serves that view and stays ready; after
            // `STALE_AFTER`, the view is `Estimated`.
            Some(Err(error)) => {
                let failing_for = cache.failed();
                if failing_for < STALE_AFTER {
                    tracing::warn!(%error, %selector, "EndpointSlice watch error");
                } else {
                    tracing::error!(
                        %error,
                        %selector,
                        ?failing_for,
                        "EndpointSlice watch still failing"
                    );
                }
            }
            None => return,
        }
        recheck_at = cache.next_recheck();
    }
}

/// The watch with kube's backoff. kube resets its backoff on every item, and
/// sends `Init` before each relist attempt, before any request, so a list that
/// keeps failing would be retried every 800 ms. `Init` is dropped: the cache
/// starts a relist at its first `InitApply` instead, which works because, with
/// paging off, a list's objects arrive together.
pub(super) fn retried<K>(
    events: impl Stream<Item = watcher::Result<watcher::Event<K>>>,
) -> impl Stream<Item = watcher::Result<watcher::Event<K>>> {
    events
        .filter(|event| std::future::ready(!matches!(event, Ok(watcher::Event::Init))))
        .default_backoff()
}

/// A [`Directory`] served from one EndpointSlice watch. The watch starts on
/// first use and stops when the directory is dropped.
pub(super) struct SliceDirectory<I: SliceIndex> {
    slices: Api<EndpointSlice>,
    selector: String,
    cache: Arc<SliceCache<I>>,
    watching: OnceLock<()>,
    stop: CancellationToken,
}

impl<I: SliceIndex> SliceDirectory<I> {
    pub(super) fn new(slices: Api<EndpointSlice>, selector: String, rules: I::Rules) -> Self {
        Self {
            slices,
            selector,
            cache: Arc::new(SliceCache::new(rules)),
            watching: OnceLock::new(),
            stop: CancellationToken::new(),
        }
    }

    /// Every `Directory` method runs in a Tokio runtime, so the spawn is safe.
    fn ensure_watching(&self) {
        self.watching.get_or_init(|| {
            tokio::spawn(watch_slices(
                self.slices.clone(),
                self.selector.clone(),
                Arc::clone(&self.cache),
                self.stop.clone(),
            ));
        });
    }
}

impl<I: SliceIndex> fmt::Debug for SliceDirectory<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SliceDirectory")
            .field("selector", &self.selector)
            .finish_non_exhaustive()
    }
}

impl<I: SliceIndex> Drop for SliceDirectory<I> {
    fn drop(&mut self) {
        self.stop.cancel();
    }
}

#[async_trait]
impl<I: SliceIndex> Directory<I::Item, I::Key> for SliceDirectory<I> {
    /// From the cache once it has synced. Before that, one list from the API
    /// server's cache, so a caller never waits for the watch to start.
    async fn list(&self) -> Result<Snapshot<I::Item>, DiscoveryError> {
        self.ensure_watching();
        if let Some(snapshot) = self.cache.snapshot() {
            return Ok(snapshot);
        }
        let listed = self
            .slices
            .list(&ListParams::default().labels(&self.selector).match_any())
            .await
            .map_err(unavailable)?;
        let entries = listed
            .items
            .iter()
            .filter_map(|slice| Some((slice_key(slice), I::entry(&self.cache.rules, slice)?)))
            .collect();
        let mut index = I::default();
        index.replace(&self.cache.rules, entries);
        let mut snapshot = index.snapshot();
        if index.is_empty() {
            snapshot.completeness = Completeness::Estimated;
        }
        Ok(snapshot)
    }

    /// The stream starts at the first full listing. Until then it waits while
    /// the watch retries and logs its failures, as `dynamo_runtime`'s
    /// Kubernetes discovery waits to be ready; `cancel_token` ends the wait.
    async fn list_and_watch(
        &self,
        cancel_token: Option<CancellationToken>,
    ) -> Result<WatchStream<I::Item, I::Key>, DiscoveryError> {
        self.ensure_watching();
        Ok(self
            .cache
            .subscribe(cancel_token.unwrap_or_default(), self.stop.clone()))
    }
}

fn unavailable(error: impl Into<anyhow::Error>) -> DiscoveryError {
    DiscoveryError::Unavailable(error.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(start_paused = true)]
    async fn a_relist_that_keeps_failing_backs_off_without_its_init() {
        let attempts = (0..4).flat_map(|_| {
            [
                Ok(watcher::Event::<EndpointSlice>::Init),
                Err(watcher::Error::NoResourceVersion),
            ]
        });
        let mut events = std::pin::pin!(retried(futures::stream::iter(attempts)));
        let mut last = Instant::now();
        let mut gaps = Vec::new();
        while let Some(event) = events.next().await {
            assert!(event.is_err(), "`Init` is dropped");
            gaps.push(last.elapsed());
            last = Instant::now();
        }
        // Each retry waits longer, as kube's backoff grows from 800 ms.
        assert_eq!(gaps.len(), 4);
        assert_eq!(gaps[0], Duration::ZERO);
        assert!(
            gaps.windows(2).skip(1).all(|pair| pair[1] > pair[0]),
            "{gaps:?}"
        );
    }
}
