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
//! [`ServingLoadAggregator`] refreshes publisher discovery once per
//! [`AGGREGATE_INTERVAL`] and publishes one complete [`ServingLoadSnapshot`].

mod frontend;

use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dynamo_kv_router::identity::PoolId;
use dynamo_kv_router::indexer::cuckoo::ProducerIdentity;
use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::component::Component;
use dynamo_runtime::discovery::{DiscoveryInstance, DiscoveryQuery, EventChannelQuery};
use dynamo_runtime::traits::DistributedRuntimeProvider;
use futures::StreamExt;
use futures::future::join_all;
use futures::stream::BoxStream;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio_stream::StreamMap;
use tokio_util::sync::CancellationToken;

use self::frontend::{
    FrontendEvent, FrontendReports, frontend_events, frontend_namespaces, sync_subscriptions,
};
use super::load::{Coverage, PoolDeployment, SchedulerLoadSnapshot};
use super::pool_registry::PoolRegistry;
use super::topology::TopologyPublisher;
use crate::frontend_load::{FRONTEND_LOAD_TOPIC, RequestGauges, RequestTotals};
use crate::kv_router::SCHEDULER_LOAD_SUBJECT;
use crate::utils::retry::FailureStreak;

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

const AGGREGATE_INTERVAL: Duration = Duration::from_secs(1);
/// Bounds every discovery listing so a hung backend delays a snapshot instead of
/// stalling it.
const EVENT_CHANNEL_LIST_TIMEOUT: Duration = Duration::from_secs(1);

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
    snapshots: watch::Sender<Arc<ServingLoadSnapshot>>,
}

impl ServingLoadAggregator {
    /// Starts the aggregator. The returned watch closes if it stops; it only
    /// stops on `cancel`.
    pub(super) fn spawn(
        component: &Component,
        pools: Arc<PoolRegistry>,
        topology: Arc<TopologyPublisher>,
        cancel: CancellationToken,
    ) -> (watch::Receiver<Arc<ServingLoadSnapshot>>, JoinHandle<()>) {
        let (snapshots, receiver) = watch::channel(Arc::default());
        let aggregator = Self {
            drt: component.drt().clone(),
            relay_namespace: component.namespace().name(),
            pools,
            topology,
            snapshots,
        };
        (receiver, tokio::spawn(aggregator.run(cancel)))
    }

    async fn run(self, cancel: CancellationToken) {
        let mut subscriptions = StreamMap::<String, BoxStream<'static, FrontendEvent>>::new();
        let mut reports = FrontendReports::default();
        let mut discovery_failures = FailureStreak::default();
        let mut tick = tokio::time::interval(AGGREGATE_INTERVAL);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => return,
                _ = tick.tick() => {}
                Some((namespace, event)) = subscriptions.next() => {
                    reports.record(namespace, event);
                    continue;
                }
            }
            let catalog = self.pools.catalog();
            let topology = self.topology.snapshot();
            let namespaces = frontend_namespaces(&catalog, &topology, &self.relay_namespace);
            sync_subscriptions(&mut subscriptions, &namespaces, |namespace| {
                frontend_events(self.drt.clone(), namespace.to_string()).boxed()
            });
            let frontend_queries = namespaces
                .iter()
                .map(|&namespace| {
                    (
                        namespace.to_string(),
                        EventChannelQuery::namespace_topic(namespace, FRONTEND_LOAD_TOPIC),
                    )
                })
                .collect::<Vec<_>>();
            let scheduler_queries = catalog
                .pools()
                .iter()
                .map(|pool| {
                    let endpoint = pool.serving_endpoint().clone();
                    (
                        endpoint.clone(),
                        EventChannelQuery::endpoint_topic(endpoint, SCHEDULER_LOAD_SUBJECT),
                    )
                })
                .collect::<Vec<_>>();
            let discovery = async {
                let (frontends, schedulers) = tokio::join!(
                    list_event_publishers(&self.drt, frontend_queries),
                    list_event_publishers(&self.drt, scheduler_queries),
                );
                report_discovery(
                    &mut discovery_failures,
                    frontends.1.into_iter().chain(schedulers.1),
                );
                (frontends.0, schedulers.0)
            };
            let (frontends, schedulers) = tokio::select! {
                biased;
                _ = cancel.cancelled() => return,
                discovered = discovery => discovered,
            };
            let now = Instant::now();
            reports.prune(&frontends, now);
            self.snapshots.send_replace(Arc::new(ServingLoadSnapshot {
                pools: self.pools.serving_pool_loads(&schedulers, now),
                models: reports.model_loads(&catalog, &frontends, now),
            }));
        }
    }
}

/// Publishers registered in discovery for each keyed query, and the errors of
/// the listings that failed; a failed key is absent from the map.
///
/// Discovery is advisory: ZMQ-broker publishers skip registration, so an empty
/// listing does not mean there are no publishers.
async fn list_event_publishers<K: Eq + Hash>(
    drt: &DistributedRuntime,
    queries: impl IntoIterator<Item = (K, EventChannelQuery)>,
) -> (HashMap<K, HashSet<u64>>, Vec<anyhow::Error>) {
    let discovery = drt.discovery();
    let listings = join_all(queries.into_iter().map(|(key, query)| {
        let discovery = discovery.clone();
        async move {
            let listed = tokio::time::timeout(
                EVENT_CHANNEL_LIST_TIMEOUT,
                discovery.list(DiscoveryQuery::EventChannels(query.clone())),
            )
            .await
            .unwrap_or_else(|_| {
                Err(anyhow::anyhow!(
                    "timed out after {EVENT_CHANNEL_LIST_TIMEOUT:?}"
                ))
            })
            .map_err(|error| error.context(format!("listing {query:?}")));
            (key, listed)
        }
    }))
    .await;
    let mut publishers = HashMap::new();
    let mut errors = Vec::new();
    for (key, listed) in listings {
        match listed {
            Ok(instances) => {
                let ids = instances
                    .into_iter()
                    .filter_map(|instance| match instance {
                        DiscoveryInstance::EventChannel { instance_id, .. } => Some(instance_id),
                        _ => None,
                    })
                    .collect();
                publishers.insert(key, ids);
            }
            Err(error) => errors.push(error),
        }
    }
    (publishers, errors)
}

/// Warns once per streak of ticks with failed listings and logs recovery.
fn report_discovery(failures: &mut FailureStreak, errors: impl IntoIterator<Item = anyhow::Error>) {
    let mut errors = errors.into_iter();
    let Some(error) = errors.next() else {
        if let Some(failures) = failures.recover() {
            tracing::info!(failures, "KV DC Relay serving-load discovery recovered");
        }
        return;
    };
    let failed = 1 + errors.count();
    let error = format!("{error:#}");
    if failures.fail() {
        tracing::warn!(%error, failed, "KV DC Relay serving-load discovery failed; affected coverage cannot be proven complete");
    } else {
        tracing::debug!(%error, failed, failures = failures.failures(), "KV DC Relay serving-load discovery still failing");
    }
}
