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
use futures::StreamExt;
use k8s_openapi::api::discovery::v1::{Endpoint, EndpointSlice};
use kube::Api;
use kube::api::ListParams;
use kube::runtime::{WatchStreamExt, watcher};
use tokio::sync::broadcast;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use super::super::stream::{EVENT_BUFFER, watch_stream};

/// How long the watch may keep failing before the cache reports its last view
/// as `Estimated`. It keeps serving that view, and the service stays ready.
pub(super) const STALE_AFTER: Duration = Duration::from_secs(60);

/// How long without an error before a failing watch counts as working again.
/// kube resumes a watch without an event, so silence is the only sign of it.
/// While the watch fails, errors come at least every 30 s backoff plus 30 s
/// connect timeout.
pub(super) const RECOVERED_AFTER: Duration = Duration::from_secs(120);

/// Slices are keyed by (namespace, name).
pub(super) type SliceKey = (String, String);

pub(super) fn slice_key(slice: &EndpointSlice) -> SliceKey {
    (
        slice.metadata.namespace.clone().unwrap_or_default(),
        slice.metadata.name.clone().unwrap_or_default(),
    )
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
    fn replace(&mut self, rules: &Self::Rules, entries: HashMap<SliceKey, Self::Entry>);

    fn snapshot(&self) -> Snapshot<Self::Item>;
}

pub(super) struct CacheState<I: SliceIndex> {
    pub(super) synced: bool,
    /// The current run of watch errors. `None` while the watch works.
    failures: Option<Failures>,
    /// Slices listed since the last `Init`; they replace the index at
    /// `InitDone`.
    relist: Option<HashMap<SliceKey, I::Entry>>,
    pub(super) index: I,
}

impl<I: SliceIndex> Default for CacheState<I> {
    fn default() -> Self {
        Self {
            synced: false,
            failures: None,
            relist: None,
            index: I::default(),
        }
    }
}

impl<I: SliceIndex> CacheState<I> {
    /// `None` until the first list completes.
    fn snapshot(&self) -> Option<Snapshot<I::Item>> {
        let mut snapshot = self.synced.then(|| self.index.snapshot())?;
        if self
            .failures
            .is_some_and(|failures| failures.stale(Instant::now()))
        {
            snapshot.completeness = Completeness::Estimated;
        }
        Some(snapshot)
    }
}

/// Watch errors no more than `RECOVERED_AFTER` apart.
#[derive(Clone, Copy)]
struct Failures {
    first: Instant,
    last: Instant,
}

impl Failures {
    /// The errors span `STALE_AFTER`, and the last one is recent. One error
    /// followed by silence is a watch that resumed.
    fn stale(&self, now: Instant) -> bool {
        now.saturating_duration_since(self.last) < RECOVERED_AFTER
            && self.last.saturating_duration_since(self.first) >= STALE_AFTER
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
        let mut state = self.state.write().unwrap_or_else(PoisonError::into_inner);
        // Every event but `Init` came from the API server, so the watch works.
        // kube sends `Init` before each relist attempt, before any request.
        if !matches!(event, watcher::Event::Init) {
            state.failures = None;
        }
        let mut listed_nothing = false;
        let events = match event {
            watcher::Event::Init => {
                state.relist = Some(HashMap::new());
                Vec::new()
            }
            watcher::Event::InitApply(slice) => {
                // A watcher that streams its lists sends no `Init`, so the
                // first slice starts the relist too.
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
                state.index.replace(&self.rules, entries);
                state.synced = true;
                let snapshot = state.index.snapshot();
                listed_nothing = snapshot.items.is_empty();
                vec![WatchEvent::Resync(snapshot)]
            }
            watcher::Event::Apply(slice) => {
                let entry = I::entry(&self.rules, &slice);
                state.index.upsert(&self.rules, slice_key(&slice), entry)
            }
            watcher::Event::Delete(slice) => {
                state.index.upsert(&self.rules, slice_key(&slice), None)
            }
        };
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
        let failures = match state.failures {
            Some(failures) if now.saturating_duration_since(failures.last) < RECOVERED_AFTER => {
                Failures {
                    last: now,
                    ..failures
                }
            }
            _ => Failures {
                first: now,
                last: now,
            },
        };
        state.failures = Some(failures);
        now.saturating_duration_since(failures.first)
    }

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
        ..watcher::Config::default().labels(&selector).any_semantic()
    };
    // kube resets its backoff on every item, and sends `Init` before each
    // relist attempt, before any request, so a list that keeps failing would
    // be retried every 800 ms. The cache starts a relist at its first
    // `InitApply` instead: with paging off, a list's slices arrive together.
    let mut events = std::pin::pin!(
        watcher(slices, config)
            .filter(|event| std::future::ready(!matches!(event, Ok(watcher::Event::Init))))
            .default_backoff()
    );
    loop {
        let event = tokio::select! {
            biased;
            () = stop.cancelled() => return,
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
    }
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
        Ok(index.snapshot())
    }

    /// The stream starts at the first complete list. Until then it waits while
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
