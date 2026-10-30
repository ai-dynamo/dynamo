// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-memory copies of the service's lists, refreshed on an interval. A failed
//! refresh keeps the last list and marks it stale, so a discovery outage does
//! not drop what a caller already knows.

use std::collections::HashSet;
use std::error::Error;
use std::future::Future;
use std::time::Duration;

use dynamo_global_common::RouterReplica;
use tokio::sync::watch;
use tokio::time::{Interval, MissedTickBehavior};
use tokio_util::sync::CancellationToken;
use tokio_util::task::AbortOnDropHandle;

use crate::{DiscoveryClientError, GlobalDiscoveryClient};

/// Whether a cache holds the service's current list.
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

    use dynamo_global_common::RouterReplicaId;
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
}
