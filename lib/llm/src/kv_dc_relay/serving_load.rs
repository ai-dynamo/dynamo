// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Serving load behind `SubscribeServingLoad`.
//!
//! Two inputs feed it:
//!
//! - router scheduler views of each pool's worker ranks, which each pool's load
//!   collector feeds into [`SchedulerLoadState`](super::load::SchedulerLoadState)
//!   from [`SCHEDULER_LOAD_SUBJECT`];
//! - frontend load frames from [`FRONTEND_LOAD_TOPIC`], which [`frontend`]
//!   folds into per-model views.
//!
//! [`ServingLoadAggregator`] watches the publishers registered for both
//! streams ([`roster`]) and publishes one complete [`ServingLoadSnapshot`] per
//! load window.

mod frontend;
mod roster;

use std::sync::Arc;
use std::time::{Duration, Instant};

use dynamo_kv_router::identity::PoolId;
use dynamo_kv_router::indexer::cuckoo::ProducerIdentity;
use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::component::Component;
use dynamo_runtime::discovery::EventChannelQuery;
use dynamo_runtime::protocols::EndpointId;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use futures::StreamExt;
use futures::stream::BoxStream;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio_stream::StreamMap;
use tokio_util::sync::CancellationToken;

use self::frontend::{
    FRONTEND_FRESHNESS, FrontendEvent, FrontendReports, frontend_events, frontend_namespaces,
    sync_subscriptions,
};
use self::roster::Rosters;
use super::load::{Coverage, PoolDeployment, SCHEDULER_FRESHNESS, SchedulerLoadSnapshot};
use super::pool_registry::PoolRegistry;
use super::topology::TopologyPublisher;
use crate::frontend_load::{FRONTEND_LOAD_TOPIC, RequestGauges, RequestTotals};
use crate::kv_router::SCHEDULER_LOAD_SUBJECT;

/// A source's report counts for this many of its publish intervals, so one or
/// two lost reports do not make an aggregate incomplete.
const FRESHNESS_INTERVALS: u32 = 3;

pub(super) const fn freshness(publish_interval: Duration) -> Duration {
    publish_interval.saturating_mul(FRESHNESS_INTERVALS)
}

/// How long the Relay remembers a silent source. Bounds the state left behind
/// by departed routers and frontends; much longer than any freshness window, so
/// a briefly silent source is resumed rather than mistaken for a new one.
pub(super) const SOURCE_RETENTION: Duration = Duration::from_secs(10 * 60);

/// How long the catalog may have pools while a stream has no publisher, before
/// the Relay warns that producers are probably not publishing.
const SILENT_STREAM_WARNING: Duration = Duration::from_secs(30);

/// One complete serving-load view: every catalog pool and every model a catalog
/// pool registers or a fresh frontend serves.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ServingLoadSnapshot {
    pub(super) pools: Vec<PoolServingLoad>,
    pub(super) models: Vec<ModelServingLoad>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PoolServingLoad {
    pub(super) producer: ProducerIdentity,
    pub(super) scheduler: SchedulerLoadSnapshot,
    /// `None` for a pool without valid serving facts.
    pub(super) deployment: Option<PoolDeployment>,
}

/// Frontend load of one canonical model in one namespace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ModelServingLoad {
    pub(super) namespace: String,
    pub(super) model: String,
    /// In-flight requests summed over the namespace's fresh frontends.
    pub(super) gauges: Coverage<RequestGauges>,
    /// Relay-lifetime cumulative counters.
    pub(super) totals: RequestTotals,
    /// Oldest publish time among the namespace's fresh frontends; 0 when none.
    pub(super) source_observed_ms: u64,
    pub(super) expected_frontends: u64,
    pub(super) observed_frontends: u64,
    pub(super) ready_frontends: u64,
    pub(super) serving_pools: Vec<PoolId>,
}

/// Builds [`ServingLoadSnapshot`]s from the pool registry and frontend frames.
pub(super) struct ServingLoadAggregator {
    drt: DistributedRuntime,
    /// Where frontends without an exact namespace publish.
    relay_namespace: String,
    pools: Arc<PoolRegistry>,
    topology: Arc<TopologyPublisher>,
    snapshots: watch::Sender<Option<Arc<ServingLoadSnapshot>>>,
}

impl ServingLoadAggregator {
    /// Starts the aggregator, which snapshots every `window`. The returned
    /// watch holds `None` until the first snapshot and closes if the
    /// aggregator stops; it only stops on `cancel`.
    pub(super) fn spawn(
        component: &Component,
        pools: Arc<PoolRegistry>,
        topology: Arc<TopologyPublisher>,
        window: Duration,
        cancel: CancellationToken,
    ) -> (watch::Receiver<Option<Arc<ServingLoadSnapshot>>>, JoinHandle<()>) {
        let (snapshots, receiver) = watch::channel(None);
        let aggregator = Self {
            drt: component.drt().clone(),
            relay_namespace: component.namespace().name(),
            pools,
            topology,
            snapshots,
        };
        (receiver, tokio::spawn(aggregator.run(window, cancel)))
    }

    async fn run(self, window: Duration, cancel: CancellationToken) {
        let mut subscriptions = StreamMap::<String, BoxStream<'static, FrontendEvent>>::new();
        let mut frontend_rosters = Rosters::<String>::default();
        let mut scheduler_rosters = Rosters::<EndpointId>::default();
        let mut reports = FrontendReports::default();
        let mut silent_frontends = SilentStream::new("frontend_load");
        let mut silent_routers = SilentStream::new("scheduler_load");
        let mut tick = tokio::time::interval(window);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            // Events first: a slow snapshot must not leave frames undrained.
            tokio::select! {
                biased;
                _ = cancel.cancelled() => return,
                Some((namespace, event)) = subscriptions.next() => {
                    reports.record(namespace, event);
                    continue;
                }
                Some((namespace, event)) = frontend_rosters.next() => {
                    frontend_rosters.apply(namespace, event, Instant::now());
                    continue;
                }
                Some((endpoint, event)) = scheduler_rosters.next() => {
                    scheduler_rosters.apply(endpoint, event, Instant::now());
                    continue;
                }
                _ = tick.tick() => {}
            }
            let catalog = self.pools.catalog();
            let topology = self.topology.snapshot();
            let namespaces = frontend_namespaces(&catalog, &topology, &self.relay_namespace);
            sync_subscriptions(&mut subscriptions, &namespaces, |namespace| {
                frontend_events(self.drt.clone(), namespace.to_string()).boxed()
            });
            frontend_rosters.sync(
                namespaces.iter().map(|&namespace| namespace.to_string()).collect(),
                |namespace| EventChannelQuery::namespace_topic(namespace, FRONTEND_LOAD_TOPIC),
                &self.drt,
            );
            scheduler_rosters.sync(
                catalog
                    .pools()
                    .iter()
                    .map(|pool| pool.serving_endpoint().clone())
                    .collect(),
                |endpoint| EventChannelQuery::endpoint_topic(endpoint.clone(), SCHEDULER_LOAD_SUBJECT),
                &self.drt,
            );
            let now = Instant::now();
            reports.prune(|namespace, publisher| frontend_rosters.contains(namespace, publisher), now);
            let snapshot = ServingLoadSnapshot {
                pools: self
                    .pools
                    .serving_pool_loads(&scheduler_rosters.recent(SCHEDULER_FRESHNESS, now), now),
                models: reports.model_loads(
                    &catalog,
                    &self.relay_namespace,
                    &frontend_rosters.recent(FRONTEND_FRESHNESS, now),
                    now,
                ),
            };
            let has_pools = !snapshot.pools.is_empty();
            silent_frontends.check(
                has_pools
                    && !frontend_rosters.any_registered()
                    && snapshot.models.iter().all(|model| model.observed_frontends == 0),
                now,
            );
            silent_routers.check(
                has_pools
                    && !scheduler_rosters.any_registered()
                    && snapshot
                        .pools
                        .iter()
                        .all(|pool| matches!(pool.scheduler.coverage, Coverage::Missing)),
                now,
            );
            self.snapshots.send_replace(Some(Arc::new(snapshot)));
        }
    }
}

/// Warns once when the catalog has pools but a stream has had no publisher,
/// registered or heard, for [`SILENT_STREAM_WARNING`]: producers publish only
/// with `DYN_SERVING_LOAD_PUBLISH` set.
struct SilentStream {
    stream: &'static str,
    since: Option<Instant>,
    warned: bool,
}

impl SilentStream {
    fn new(stream: &'static str) -> Self {
        Self {
            stream,
            since: None,
            warned: false,
        }
    }

    fn check(&mut self, silent: bool, now: Instant) {
        if !silent {
            if self.warned {
                tracing::info!(stream = self.stream, "KV DC Relay serving-load publishers appeared");
            }
            self.since = None;
            self.warned = false;
            return;
        }
        let since = *self.since.get_or_insert(now);
        if !self.warned && now.saturating_duration_since(since) >= SILENT_STREAM_WARNING {
            self.warned = true;
            tracing::warn!(
                stream = self.stream,
                silent_secs = SILENT_STREAM_WARNING.as_secs(),
                "KV DC Relay has no serving-load publishers; set DYN_SERVING_LOAD_PUBLISH=1 on frontends, routers and workers"
            );
        }
    }
}
