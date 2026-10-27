// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One EndpointSlice watch feeding a derived index, shared by the pool and
//! router replica directories. Each slice event updates only what that slice
//! affects; a relist rebuilds the index once, at `InitDone`.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, PoisonError, RwLock, RwLockReadGuard};
use std::time::Duration;

use crate::pool_discovery::{
    Completeness, Directory, PoolDiscoveryError, Snapshot, WatchEvent, WatchStream,
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

/// How long the watch may fail before the cache reports its last view as
/// `Estimated`. It keeps serving that view, and the service stays ready.
pub(super) const STALE_AFTER: Duration = Duration::from_secs(60);

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
    /// When the watch started failing. `None` while it works.
    failing_since: Option<Instant>,
    /// Slices listed since the last `Init`; they replace the index at
    /// `InitDone`.
    relist: Option<HashMap<SliceKey, I::Entry>>,
    pub(super) index: I,
}

impl<I: SliceIndex> Default for CacheState<I> {
    fn default() -> Self {
        Self {
            synced: false,
            failing_since: None,
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
            .failing_since
            .is_some_and(|since| since.elapsed() >= STALE_AFTER)
        {
            snapshot.completeness = Completeness::Estimated;
        }
        Some(snapshot)
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

    pub(super) fn apply(&self, event: watcher::Event<EndpointSlice>) {
        let mut state = self.state.write().unwrap_or_else(PoisonError::into_inner);
        // Any event means the watch works.
        state.failing_since = None;
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
                vec![WatchEvent::Resync(state.index.snapshot())]
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
    }

    /// The watch failed again. Returns how long it has been failing.
    pub(super) fn failing(&self) -> Duration {
        let mut state = self.state.write().unwrap_or_else(PoisonError::into_inner);
        state
            .failing_since
            .get_or_insert_with(Instant::now)
            .elapsed()
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
    let mut events = std::pin::pin!(watcher(slices, config).default_backoff());
    loop {
        let event = tokio::select! {
            biased;
            () = stop.cancelled() => return,
            event = events.next() => event,
        };
        match event {
            Some(Ok(event)) => {
                let relisted = matches!(event, watcher::Event::InitDone);
                cache.apply(event);
                if relisted
                    && cache
                        .snapshot()
                        .is_some_and(|listed| listed.items.is_empty())
                {
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
                let failing_for = cache.failing();
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

impl<I: SliceIndex> Drop for SliceDirectory<I> {
    fn drop(&mut self) {
        self.stop.cancel();
    }
}

#[async_trait]
impl<I: SliceIndex> Directory<I::Item, I::Key> for SliceDirectory<I> {
    /// From the cache once it has synced. Before that, one list from the API
    /// server's cache, so a caller never waits for the watch to start.
    async fn list(&self) -> Result<Snapshot<I::Item>, PoolDiscoveryError> {
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
    ) -> Result<WatchStream<I::Item, I::Key>, PoolDiscoveryError> {
        self.ensure_watching();
        Ok(self
            .cache
            .subscribe(cancel_token.unwrap_or_default(), self.stop.clone()))
    }
}

fn unavailable(error: impl Into<anyhow::Error>) -> PoolDiscoveryError {
    PoolDiscoveryError::Unavailable(error.into())
}
