// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend load behind `ServingLoadUpdate.models`.
//!
//! Every frontend publishes a [`FrontendLoadFrame`] every
//! [`FRONTEND_LOAD_PUBLISH_INTERVAL`] in one namespace. The Relay subscribes in
//! the namespaces of [`frontend_namespaces`], keeps the latest frame per
//! frontend, folds each frontend's cumulative counters into Relay-lifetime
//! totals, and builds one [`ModelServingLoad`] per (namespace, model).
//!
//! Frontends are keyed by `frontend_instance_id`, which is stable for a
//! frontend's lifetime. The event-plane `publisher_id` is not: it changes
//! whenever the publisher reconnects, so it is only matched against discovery.

use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::time::{Duration, Instant};

use dynamo_kv_router::identity::PoolId;
use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::transports::event_plane::{Codec, EventSubscriber};
use futures::Stream;
use tokio_stream::StreamMap;

use super::super::identity::DcPoolCatalog;
use super::super::load::Coverage;
use super::super::topology::TopologySnapshot;
use super::{ModelServingLoad, SOURCE_RETENTION, freshness};
use crate::frontend_load::{
    FRONTEND_LOAD_PUBLISH_INTERVAL, FRONTEND_LOAD_TOPIC, FrontendLoadFrame, FrontendModelLoad,
    RequestGauges, RequestTotals,
};
use crate::utils::retry::{Backoff, FailureStreak};

/// How long a frontend's frame counts as observed.
const FRONTEND_FRESHNESS: Duration = freshness(FRONTEND_LOAD_PUBLISH_INTERVAL);
const RESUBSCRIBE_INITIAL: Duration = Duration::from_millis(100);
const RESUBSCRIBE_MAX: Duration = Duration::from_secs(5);

/// One frame received in a namespace.
pub(super) struct FrontendEvent {
    /// Event-plane publisher that sent the frame; changes on publisher reconnect.
    publisher_id: u64,
    published_at_ms: u64,
    received_at: Instant,
    frame: FrontendLoadFrame,
}

/// Namespaces the Relay hears frontends in: every namespace that holds a
/// catalog pool or a served model (the namespace source's view), plus the
/// Relay's own, where frontends without an exact namespace publish.
pub(super) fn frontend_namespaces<'a>(
    catalog: &'a DcPoolCatalog,
    topology: &'a TopologySnapshot,
    relay_namespace: &'a str,
) -> BTreeSet<&'a str> {
    catalog
        .pools()
        .iter()
        .map(|pool| pool.serving_endpoint().namespace.as_str())
        .chain(
            topology
                .entries
                .iter()
                .map(|entry| entry.namespace.as_str()),
        )
        .chain([relay_namespace])
        .collect()
}

/// Makes `subscriptions` hold exactly one stream per namespace, keeping existing
/// streams. Removing a namespace drops its stream, which unsubscribes.
pub(super) fn sync_subscriptions<S: Stream + Unpin>(
    subscriptions: &mut StreamMap<String, S>,
    namespaces: &BTreeSet<&str>,
    mut subscribe: impl FnMut(&str) -> S,
) {
    let stale = subscriptions
        .keys()
        .filter(|namespace| !namespaces.contains(namespace.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    for namespace in stale {
        subscriptions.remove(&namespace);
    }
    for &namespace in namespaces {
        if !subscriptions.contains_key(namespace) {
            subscriptions.insert(namespace.to_string(), subscribe(namespace));
        }
    }
}

/// Frontend frames published in `namespace`, resubscribing with backoff after
/// event-plane failures. Undecodable frames are skipped. Never ends.
pub(super) fn frontend_events(
    drt: DistributedRuntime,
    namespace: String,
) -> impl Stream<Item = FrontendEvent> + Send + 'static {
    async_stream::stream! {
        let mut retry = Backoff::new(RESUBSCRIBE_INITIAL, RESUBSCRIBE_MAX);
        let mut failures = FailureStreak::default();
        let mut decode_failures = FailureStreak::default();
        loop {
            let subscriber = match drt.namespace(namespace.as_str()) {
                Ok(scope) => EventSubscriber::for_namespace(&scope, FRONTEND_LOAD_TOPIC).await,
                Err(error) => Err(error),
            };
            let error = match subscriber {
                Ok(mut subscriber) => loop {
                    let envelope = match subscriber.next().await {
                        Some(Ok(envelope)) => envelope,
                        Some(Err(error)) => break error,
                        None => break anyhow::anyhow!("stream closed"),
                    };
                    retry.reset();
                    if let Some(failures) = failures.recover() {
                        tracing::info!(%namespace, failures, "KV DC Relay frontend-load subscription recovered");
                    }
                    match Codec::default().decode_payload::<FrontendLoadFrame>(&envelope.payload) {
                        Ok(frame) => {
                            decode_failures.recover();
                            yield FrontendEvent {
                                publisher_id: envelope.publisher_id,
                                published_at_ms: envelope.published_at,
                                received_at: Instant::now(),
                                frame,
                            };
                        }
                        Err(error) if decode_failures.fail() => {
                            tracing::warn!(%namespace, publisher_id = envelope.publisher_id, %error, "Skipping undecodable frontend load frames");
                        }
                        Err(error) => {
                            tracing::debug!(%namespace, publisher_id = envelope.publisher_id, %error, "Skipping undecodable frontend load frame");
                        }
                    }
                },
                Err(error) => error,
            };
            let delay = retry.next_delay();
            if failures.fail() {
                tracing::warn!(%namespace, %error, retry_ms = delay.as_millis(), "KV DC Relay frontend-load subscription failed; resubscribing");
            } else {
                tracing::debug!(%namespace, %error, failures = failures.failures(), retry_ms = delay.as_millis(), "KV DC Relay frontend-load subscription still failing");
            }
            tokio::time::sleep(delay).await;
        }
    }
}

/// The Relay's memory of one frontend process.
struct FrontendSource {
    incarnation: u64,
    /// The incarnation this one replaced. Its frames can still arrive late and
    /// must not be mistaken for yet another restart.
    replaced_incarnation: Option<u64>,
    sequence: u64,
    publisher_id: u64,
    published_at_ms: u64,
    received_at: Instant,
    serving_ready: bool,
    models: Vec<FrontendModelLoad>,
    /// Last cumulative totals per model during this incarnation. A model keeps
    /// its baseline while absent from frames because the frontend's totals for
    /// it survive model removal and re-registration.
    baselines: HashMap<String, RequestTotals>,
}

impl FrontendSource {
    fn is_fresh(&self, now: Instant) -> bool {
        now.saturating_duration_since(self.received_at) <= FRONTEND_FRESHNESS
    }
}

/// Frontend reports, keyed by (namespace, `frontend_instance_id`).
#[derive(Default)]
pub(super) struct FrontendReports {
    frontends: HashMap<(String, u64), FrontendSource>,
    /// Relay-lifetime totals per (namespace, model). Bounded by the models ever
    /// served and never reset while the Relay runs.
    totals: HashMap<(String, String), RequestTotals>,
}

impl FrontendReports {
    /// Records one frame. Duplicate, reordered, and replaced-incarnation frames
    /// are ignored; a new incarnation of a known frontend is a restart and
    /// starts new baselines.
    pub(super) fn record(&mut self, namespace: String, event: FrontendEvent) {
        let FrontendEvent {
            publisher_id,
            published_at_ms,
            received_at,
            frame,
        } = event;
        let frontend_instance_id = frame.frontend_instance_id;
        let source = match self
            .frontends
            .entry((namespace.clone(), frontend_instance_id))
        {
            Entry::Occupied(entry) => {
                let source = entry.into_mut();
                if source.incarnation == frame.incarnation {
                    if frame.sequence <= source.sequence {
                        return;
                    }
                } else if source.replaced_incarnation == Some(frame.incarnation) {
                    return;
                } else {
                    tracing::info!(
                        %namespace,
                        frontend_instance_id,
                        "Frontend restarted; resetting its load baselines"
                    );
                    source.replaced_incarnation = Some(source.incarnation);
                    source.incarnation = frame.incarnation;
                    source.baselines.clear();
                }
                source
            }
            Entry::Vacant(entry) => entry.insert(FrontendSource {
                incarnation: frame.incarnation,
                replaced_incarnation: None,
                sequence: 0,
                publisher_id,
                published_at_ms,
                received_at,
                serving_ready: false,
                models: Vec::new(),
                baselines: HashMap::new(),
            }),
        };

        for model in &frame.models {
            let baseline = source
                .baselines
                .insert(model.model.clone(), model.totals)
                .unwrap_or_default();
            self.totals
                .entry((namespace.clone(), model.model.clone()))
                .or_default()
                .add(&model.totals.growth_since(&baseline));
        }

        source.sequence = frame.sequence;
        source.publisher_id = publisher_id;
        source.published_at_ms = published_at_ms;
        source.received_at = received_at;
        source.serving_ready = frame.serving_ready;
        source.models = frame.models;
    }

    /// Forgets frontends silent for [`SOURCE_RETENTION`] that are not
    /// discovered, which bounds state under frontend churn.
    ///
    /// Forgetting a frontend also forgets its baselines: if it later resumes
    /// the same incarnation, its cumulative counters are added in full again.
    /// That requires a long partition of a frontend that also left discovery.
    pub(super) fn prune(&mut self, discovered: &HashMap<String, HashSet<u64>>, now: Instant) {
        self.frontends.retain(|(namespace, _), source| {
            now.saturating_duration_since(source.received_at) <= SOURCE_RETENTION
                || discovered
                    .get(namespace)
                    .is_some_and(|publishers| publishers.contains(&source.publisher_id))
        });
    }

    /// One view per model a catalog pool registers or a fresh frontend serves.
    /// `discovered` holds the frontend publishers registered per namespace; a
    /// namespace whose listing failed is absent.
    pub(super) fn model_loads(
        &self,
        catalog: &DcPoolCatalog,
        discovered: &HashMap<String, HashSet<u64>>,
        now: Instant,
    ) -> Vec<ModelServingLoad> {
        let mut models = BTreeMap::<(&str, &str), ModelAggregate>::new();
        for pool in catalog.pools() {
            let namespace = pool.serving_endpoint().namespace.as_str();
            for registration in pool.registrations() {
                models
                    .entry((namespace, registration.model().as_str()))
                    .or_default()
                    .serving_pools
                    .insert(pool.pool_id());
            }
        }
        for ((namespace, _), source) in &self.frontends {
            if !source.is_fresh(now) {
                continue;
            }
            for model in &source.models {
                let aggregate = models
                    .entry((namespace.as_str(), model.model.as_str()))
                    .or_default();
                aggregate.ready_frontends += u64::from(source.serving_ready);
                aggregate.gauges.add(&model.gauges);
            }
        }
        let mut coverage = HashMap::<&str, NamespaceCoverage>::new();
        models
            .into_iter()
            .map(|((namespace, model), aggregate)| {
                let coverage = *coverage
                    .entry(namespace)
                    .or_insert_with(|| self.coverage(namespace, discovered.get(namespace), now));
                ModelServingLoad {
                    namespace: namespace.to_string(),
                    model: model.to_string(),
                    gauges: match coverage.status {
                        Coverage::Complete(()) => Coverage::Complete(aggregate.gauges),
                        Coverage::Partial => Coverage::Partial,
                        Coverage::Missing => Coverage::Missing,
                    },
                    totals: self
                        .totals
                        .get(&(namespace.to_string(), model.to_string()))
                        .copied()
                        .unwrap_or_default(),
                    source_observed_ms: coverage.source_observed_ms,
                    expected_frontends: coverage.expected,
                    observed_frontends: coverage.observed,
                    ready_frontends: aggregate.ready_frontends,
                    serving_pools: aggregate.serving_pools.into_iter().collect(),
                }
            })
            .collect()
    }

    /// Expected frontends are the publishers discovered in the namespace plus
    /// every frontend there with a fresh frame, so broker-mode frontends (never
    /// discovered) and frontends discovery has not caught up with still count.
    fn coverage(
        &self,
        namespace: &str,
        discovered: Option<&HashSet<u64>>,
        now: Instant,
    ) -> NamespaceCoverage {
        let mut observed = 0_u64;
        let mut expected = 0_u64;
        let mut oldest = None::<u64>;
        let mut known = HashSet::new();
        for ((source_namespace, _), source) in &self.frontends {
            if source_namespace != namespace {
                continue;
            }
            known.insert(source.publisher_id);
            let discovered = discovered.is_some_and(|ids| ids.contains(&source.publisher_id));
            if source.is_fresh(now) {
                observed += 1;
                expected += 1;
                oldest = Some(
                    oldest.map_or(source.published_at_ms, |at| at.min(source.published_at_ms)),
                );
            } else if discovered {
                expected += 1;
            }
        }
        if let Some(discovered) = discovered {
            expected += discovered.difference(&known).count() as u64;
        }
        let status = if discovered.is_some() && expected > 0 && observed == expected {
            Coverage::Complete(())
        } else if observed > 0 {
            Coverage::Partial
        } else {
            Coverage::Missing
        };
        NamespaceCoverage {
            status,
            expected,
            observed,
            source_observed_ms: oldest.unwrap_or_default(),
        }
    }
}

/// Frontend coverage shared by every model in one namespace.
#[derive(Clone, Copy)]
struct NamespaceCoverage {
    status: Coverage<()>,
    expected: u64,
    observed: u64,
    source_observed_ms: u64,
}

/// One model's sum over its namespace's fresh frontends.
#[derive(Default)]
struct ModelAggregate {
    serving_pools: BTreeSet<PoolId>,
    ready_frontends: u64,
    gauges: RequestGauges,
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::sync::Arc;

    use dynamo_kv_router::identity::{
        CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, RoutingScopeId,
    };
    use dynamo_kv_router::indexer::cuckoo::{CkfConfig, DcCkfState, ProducerIdentity};
    use dynamo_runtime::protocols::EndpointId;

    use super::*;
    use crate::kv_dc_relay::identity::{
        CanonicalModelId, CanonicalModelRegistration, DcPoolDescriptor, DcRelayIdentity,
        KvQueryHashFormat, KvQuerySemantics, WorkerRole,
    };
    use crate::kv_dc_relay::topology::{TopologyEntry, TopologyReadinessState};

    const NS: &str = "ns";
    const FRONTEND: u64 = 10;
    const PUBLISHER: u64 = 1;

    fn frame(incarnation: u64, sequence: u64, models: Vec<FrontendModelLoad>) -> FrontendLoadFrame {
        FrontendLoadFrame {
            frontend_instance_id: FRONTEND,
            incarnation,
            sequence,
            serving_ready: true,
            models,
        }
    }

    fn event(publisher_id: u64, frame: FrontendLoadFrame) -> FrontendEvent {
        FrontendEvent {
            publisher_id,
            published_at_ms: frame.sequence,
            received_at: Instant::now(),
            frame,
        }
    }

    fn record(reports: &mut FrontendReports, publisher_id: u64, frame: FrontendLoadFrame) {
        reports.record(NS.to_string(), event(publisher_id, frame));
    }

    fn model_load(model: &str, awaiting: u64, started: u64) -> FrontendModelLoad {
        FrontendModelLoad {
            model: model.to_string(),
            aliases: Vec::new(),
            gauges: RequestGauges {
                requests_awaiting_first_token: awaiting,
                awaiting_first_token_input_tokens: 17,
                inflight_input_tokens: 31,
                ..RequestGauges::default()
            },
            totals: RequestTotals {
                requests_started: started,
                output_tokens: 7,
                ..RequestTotals::default()
            },
        }
    }

    fn relay() -> DcRelayIdentity {
        DcRelayIdentity::new(1, 2)
    }

    fn catalog(pools: Vec<DcPoolDescriptor>) -> DcPoolCatalog {
        DcPoolCatalog::new(relay(), 1, pools)
    }

    fn pool(namespace: &str, model: &str) -> DcPoolDescriptor {
        let format = DcCkfState::new(CkfConfig::new(32)).unwrap().format();
        let pool_id = PoolId::new(
            IndexerDomainId::new(
                CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
                RoutingScopeId::new([2; 16], IdentitySource::Explicit),
            ),
            DcId::new(3),
        );
        DcPoolDescriptor::new(
            ProducerIdentity::new(pool_id, 5, 7, format),
            EndpointId::from(format!("{namespace}.worker.generate").as_str()),
            Arc::from([CanonicalModelRegistration::new(
                CanonicalModelId::new(model).unwrap(),
                Vec::new(),
            )]),
            KvQuerySemantics::new(16, KvQueryHashFormat::DynamoStandardV1).unwrap(),
            Arc::from([WorkerRole::Decode]),
        )
    }

    fn listed(publishers: &[u64]) -> HashMap<String, HashSet<u64>> {
        HashMap::from([(NS.to_string(), publishers.iter().copied().collect())])
    }

    fn only_model(
        reports: &FrontendReports,
        discovered: &HashMap<String, HashSet<u64>>,
    ) -> ModelServingLoad {
        let mut models = reports.model_loads(&catalog(Vec::new()), discovered, Instant::now());
        assert_eq!(models.len(), 1);
        models.pop().unwrap()
    }

    fn awaiting(model: &ModelServingLoad) -> Option<u64> {
        match model.gauges {
            Coverage::Complete(gauges) => Some(gauges.requests_awaiting_first_token),
            Coverage::Partial | Coverage::Missing => None,
        }
    }

    #[test]
    fn stale_sequences_are_ignored_within_an_incarnation() {
        let mut reports = FrontendReports::default();
        record(
            &mut reports,
            PUBLISHER,
            frame(1, 2, vec![model_load("a", 2, 5)]),
        );
        record(
            &mut reports,
            PUBLISHER,
            frame(1, 1, vec![model_load("a", 9, 9)]),
        );
        record(
            &mut reports,
            PUBLISHER,
            frame(1, 2, vec![model_load("a", 9, 9)]),
        );

        let model = only_model(&reports, &listed(&[PUBLISHER]));
        assert_eq!(awaiting(&model), Some(2));
        assert_eq!(model.totals.requests_started, 5);
    }

    #[test]
    fn totals_recover_skipped_frames_and_count_restarts_in_full() {
        let mut reports = FrontendReports::default();
        record(
            &mut reports,
            PUBLISHER,
            frame(1, 1, vec![model_load("a", 1, 2)]),
        );
        record(
            &mut reports,
            PUBLISHER,
            frame(1, 4, vec![model_load("a", 1, 5)]),
        );
        // Same frontend, new process: sequence and counters restart.
        record(&mut reports, 2, frame(2, 1, vec![model_load("a", 1, 3)]));

        let model = only_model(&reports, &listed(&[2]));
        assert_eq!(model.totals.requests_started, 8);
        assert_eq!(model.totals.output_tokens, 14);
        assert!(matches!(model.gauges, Coverage::Complete(_)));
    }

    #[test]
    fn incarnation_flip_flop_does_not_double_count_late_frames() {
        let mut reports = FrontendReports::default();
        record(
            &mut reports,
            PUBLISHER,
            frame(1, 7, vec![model_load("a", 1, 5)]),
        );
        record(&mut reports, 2, frame(2, 1, vec![model_load("a", 1, 3)]));
        // A late frame of the replaced process must not look like another restart.
        record(
            &mut reports,
            PUBLISHER,
            frame(1, 8, vec![model_load("a", 1, 6)]),
        );
        record(&mut reports, 2, frame(2, 2, vec![model_load("a", 4, 4)]));

        let model = only_model(&reports, &listed(&[2]));
        assert_eq!(model.totals.requests_started, 9);
        assert_eq!(awaiting(&model), Some(4));
    }

    #[test]
    fn publisher_reconnect_keeps_the_frontend_identity() {
        let mut reports = FrontendReports::default();
        record(
            &mut reports,
            PUBLISHER,
            frame(1, 1, vec![model_load("a", 1, 5)]),
        );
        record(&mut reports, 2, frame(1, 2, vec![model_load("a", 1, 8)]));

        let model = only_model(&reports, &listed(&[2]));
        assert_eq!(model.expected_frontends, 1);
        assert_eq!(model.totals.requests_started, 8);
        assert!(matches!(model.gauges, Coverage::Complete(_)));
    }

    #[test]
    fn undiscovered_fresh_frontends_count_as_expected() {
        // Broker mode never registers publishers; discovery may also lag.
        let mut reports = FrontendReports::default();
        record(
            &mut reports,
            PUBLISHER,
            frame(1, 1, vec![model_load("a", 2, 5)]),
        );
        let mut other = frame(1, 1, vec![model_load("a", 3, 5)]);
        other.frontend_instance_id = 20;
        record(&mut reports, 2, other);

        let model = only_model(&reports, &listed(&[]));
        assert_eq!(model.expected_frontends, 2);
        assert_eq!(model.ready_frontends, 2);
        let Coverage::Complete(gauges) = model.gauges else {
            panic!("expected complete gauges, got {:?}", model.gauges);
        };
        assert_eq!(gauges.requests_awaiting_first_token, 5);
        assert_eq!(gauges.inflight_input_tokens, 62);
        assert_eq!(model.totals.output_tokens, 14);
    }

    #[test]
    fn silent_or_unknown_discovered_frontends_degrade_the_model() {
        let mut reports = FrontendReports::default();
        let mut stale = event(PUBLISHER, frame(1, 1, vec![model_load("a", 2, 5)]));
        stale.received_at = Instant::now() - FRONTEND_FRESHNESS - Duration::from_millis(1);
        reports.record(NS.to_string(), stale);
        let mut fresh = frame(1, 1, vec![model_load("a", 3, 5)]);
        fresh.frontend_instance_id = 20;
        record(&mut reports, 2, fresh);

        let model = only_model(&reports, &listed(&[PUBLISHER, 2, 99]));
        assert_eq!(model.gauges, Coverage::Partial);
        assert_eq!(model.expected_frontends, 3);
        assert_eq!(model.observed_frontends, 1);
        // Totals are kept regardless of coverage.
        assert_eq!(model.totals.requests_started, 10);

        // Failed discovery cannot prove completeness.
        assert_eq!(
            only_model(&reports, &HashMap::new()).gauges,
            Coverage::Partial
        );
    }

    #[test]
    fn coverage_is_per_namespace() {
        let mut reports = FrontendReports::default();
        record(
            &mut reports,
            PUBLISHER,
            frame(1, 1, vec![model_load("a", 2, 5)]),
        );
        let mut other = frame(1, 1, vec![model_load("a", 4, 1)]);
        other.frontend_instance_id = 20;
        reports.record("other".to_string(), event(2, other));

        // "other" has a discovered frontend that never published.
        let mut discovered = listed(&[PUBLISHER]);
        discovered.insert("other".to_string(), HashSet::from([2, 3]));
        let models = reports.model_loads(&catalog(Vec::new()), &discovered, Instant::now());
        let keys = models
            .iter()
            .map(|model| (model.namespace.as_str(), model.model.as_str()))
            .collect::<Vec<_>>();
        assert_eq!(keys, [("ns", "a"), ("other", "a")]);
        assert_eq!(awaiting(&models[0]), Some(2));
        assert_eq!(models[0].totals.requests_started, 5);
        assert_eq!(models[1].gauges, Coverage::Partial);
        assert_eq!(models[1].expected_frontends, 2);
        assert_eq!(models[1].totals.requests_started, 1);
    }

    #[test]
    fn fresh_frontend_without_a_model_contributes_zero() {
        let mut reports = FrontendReports::default();
        record(
            &mut reports,
            PUBLISHER,
            frame(1, 1, vec![model_load("a", 2, 5)]),
        );
        let mut other = frame(1, 1, vec![model_load("b", 4, 5)]);
        other.frontend_instance_id = 20;
        other.serving_ready = false;
        record(&mut reports, 2, other);

        let models = reports.model_loads(&catalog(Vec::new()), &listed(&[]), Instant::now());
        let (a, b) = (&models[0], &models[1]);
        assert_eq!(a.observed_frontends, 2);
        assert_eq!(a.ready_frontends, 1);
        assert_eq!(awaiting(a), Some(2));
        assert_eq!(b.ready_frontends, 0);
        assert_eq!(awaiting(b), Some(4));
    }

    #[test]
    fn catalog_models_without_frontends_are_unavailable_with_zero_totals() {
        let reports = FrontendReports::default();
        let catalog = catalog(vec![pool(NS, "a")]);
        let models = reports.model_loads(&catalog, &listed(&[]), Instant::now());
        assert_eq!(models.len(), 1);
        assert_eq!(
            (models[0].namespace.as_str(), models[0].model.as_str()),
            (NS, "a")
        );
        assert_eq!(models[0].gauges, Coverage::Missing);
        assert_eq!(models[0].serving_pools, [catalog.pools()[0].pool_id()]);
        assert_eq!(models[0].totals, RequestTotals::default());
        assert_eq!(models[0].expected_frontends, 0);
    }

    #[test]
    fn prune_forgets_only_long_silent_undiscovered_frontends() {
        let mut reports = FrontendReports::default();
        record(
            &mut reports,
            PUBLISHER,
            frame(1, 1, vec![model_load("a", 1, 5)]),
        );
        let later = Instant::now() + SOURCE_RETENTION + Duration::from_secs(1);

        reports.prune(&listed(&[PUBLISHER]), later);
        assert_eq!(reports.frontends.len(), 1);
        reports.prune(&HashMap::new(), later);
        assert!(reports.frontends.is_empty());
        // Totals outlive the frontend.
        assert_eq!(
            reports.totals[&(NS.to_string(), "a".to_string())].requests_started,
            5
        );
    }

    #[test]
    fn subscriptions_follow_the_namespace_set() {
        let subscribed = Cell::new(0);
        let subscribe = |_: &str| {
            subscribed.set(subscribed.get() + 1);
            futures::stream::pending::<()>()
        };
        let mut subscriptions = StreamMap::new();

        sync_subscriptions(&mut subscriptions, &BTreeSet::from(["a", "b"]), subscribe);
        sync_subscriptions(&mut subscriptions, &BTreeSet::from(["b", "c"]), subscribe);

        let mut keys = subscriptions.keys().cloned().collect::<Vec<_>>();
        keys.sort();
        assert_eq!(keys, ["b", "c"]);
        // "b" kept its subscription.
        assert_eq!(subscribed.get(), 3);
    }

    #[test]
    fn frontends_are_heard_in_pool_model_and_relay_namespaces() {
        let catalog = catalog(vec![pool("pools", "a")]);
        let topology = TopologySnapshot {
            revision: 1,
            entries: vec![TopologyEntry {
                namespace: "models".to_string(),
                model: CanonicalModelId::new("b").unwrap(),
                state: TopologyReadinessState::Unavailable,
                present_roles: Vec::new(),
                missing_roles: Vec::new(),
                members: Vec::new(),
                duplicate_role_endpoints: Vec::new(),
                legacy_fallback_active: false,
                adapters: Vec::new(),
            }],
        };
        assert_eq!(
            frontend_namespaces(&catalog, &topology, "dynamo"),
            BTreeSet::from(["dynamo", "models", "pools"])
        );
        assert_eq!(
            frontend_namespaces(
                &DcPoolCatalog::new(relay(), 0, Vec::new()),
                &TopologySnapshot::default(),
                "dynamo"
            ),
            BTreeSet::from(["dynamo"])
        );
    }
}
