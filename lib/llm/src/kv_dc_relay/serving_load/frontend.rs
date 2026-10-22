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

use super::super::identity::{CanonicalModelId, DcPoolCatalog};
use super::super::load::Coverage;
use super::super::topology::TopologySnapshot;
use super::{ModelServingLoad, SOURCE_RETENTION, freshness};
use crate::frontend_load::{
    FRONTEND_LOAD_PUBLISH_INTERVAL, FRONTEND_LOAD_TOPIC, FrontendLoadFrame, FrontendModelLoad,
    RequestGauges, RequestTotals,
};
use crate::utils::retry::{Backoff, FailureStreak};

/// How long a frontend's frame counts as observed.
pub(super) const FRONTEND_FRESHNESS: Duration = freshness(FRONTEND_LOAD_PUBLISH_INTERVAL);
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

#[cfg(test)]
impl FrontendEvent {
    /// A frame from one ready frontend serving no models.
    pub(super) fn sample() -> Self {
        Self {
            publisher_id: 1,
            published_at_ms: 0,
            received_at: Instant::now(),
            frame: FrontendLoadFrame {
                frontend_instance_id: 1,
                incarnation: 1,
                sequence: 1,
                is_serving_ready: true,
                models: Vec::new(),
            },
        }
    }
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
    warned_invalid_model: bool,
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
    /// Relay-lifetime totals per namespace and model. Bounded by the models
    /// ever served and never reset while the Relay runs.
    totals: HashMap<String, HashMap<String, RequestTotals>>,
}

impl FrontendReports {
    /// Records one frame. Duplicate, reordered, and replaced-incarnation frames
    /// are ignored; a new incarnation of a known frontend is a restart and
    /// starts new baselines. A frontend first heard after its first frame only
    /// contributes growth from then on, so totals count what this Relay saw.
    /// Models whose names the wire would reject are dropped here, where they
    /// enter, rather than invalidating every consumer's window.
    pub(super) fn record(&mut self, namespace: String, event: FrontendEvent) {
        let FrontendEvent {
            publisher_id,
            published_at_ms,
            received_at,
            frame,
        } = event;
        let frontend_instance_id = frame.frontend_instance_id;
        let mut counts_from_zero = true;
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
                    tracing::debug!(
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
            Entry::Vacant(entry) => {
                counts_from_zero = frame.sequence == 1;
                entry.insert(FrontendSource {
                    incarnation: frame.incarnation,
                    replaced_incarnation: None,
                    sequence: 0,
                    publisher_id,
                    published_at_ms,
                    received_at,
                    serving_ready: false,
                    models: Vec::new(),
                    baselines: HashMap::new(),
                    warned_invalid_model: false,
                })
            }
        };

        let mut models = frame.models;
        models.retain(|model| {
            let valid = CanonicalModelId::new(model.model.as_str()).is_ok();
            if !valid && !std::mem::replace(&mut source.warned_invalid_model, true) {
                tracing::warn!(%namespace, frontend_instance_id, model = ?model.model, "Dropping frontend load of a model with an invalid name");
            }
            valid
        });

        let totals = self.totals.entry(namespace).or_default();
        for model in &models {
            let baseline = match source.baselines.get_mut(&model.model) {
                Some(baseline) => std::mem::replace(baseline, model.totals),
                None => {
                    source.baselines.insert(model.model.clone(), model.totals);
                    if counts_from_zero {
                        RequestTotals::default()
                    } else {
                        model.totals
                    }
                }
            };
            let growth = model.totals.growth_since(&baseline);
            match totals.get_mut(&model.model) {
                Some(total) => total.add(&growth),
                None => {
                    totals.insert(model.model.clone(), growth);
                }
            }
        }

        source.sequence = frame.sequence;
        source.publisher_id = publisher_id;
        source.published_at_ms = published_at_ms;
        source.received_at = received_at;
        source.serving_ready = frame.is_serving_ready;
        source.models = models;
    }

    /// Forgets frontends silent for [`SOURCE_RETENTION`] whose publisher is no
    /// longer `registered`, which bounds state under frontend churn.
    ///
    /// A forgotten frontend that resumes is heard mid-stream, so it counts
    /// only its growth from then on.
    pub(super) fn prune(&mut self, registered: impl Fn(&String, u64) -> bool, now: Instant) {
        self.frontends.retain(|(namespace, _), source| {
            now.saturating_duration_since(source.received_at) <= SOURCE_RETENTION
                || registered(namespace, source.publisher_id)
        });
    }

    /// One view per model a catalog pool registers or a fresh frontend serves.
    ///
    /// Frontends without an exact namespace publish in `relay_namespace` but
    /// serve every namespace's models, so their load of a model the catalog
    /// registers in exactly one other namespace counts there.
    ///
    /// `registered` holds, per namespace with a known roster, the frontend
    /// publishers expected besides those with a fresh frame.
    pub(super) fn model_loads(
        &self,
        catalog: &DcPoolCatalog,
        relay_namespace: &str,
        registered: &HashMap<String, HashSet<u64>>,
        now: Instant,
    ) -> Vec<ModelServingLoad> {
        let mut models = BTreeMap::<(&str, &str), ModelAggregate>::new();
        let mut homes = HashMap::<&str, BTreeSet<&str>>::new();
        for pool in catalog.pools() {
            let namespace = pool.serving_endpoint().namespace.as_str();
            for registration in pool.registrations() {
                let model = registration.model().as_str();
                models
                    .entry((namespace, model))
                    .or_default()
                    .serving_pools
                    .insert(pool.pool_id());
                homes.entry(model).or_default().insert(namespace);
            }
        }
        for ((namespace, _), source) in &self.frontends {
            if !source.is_fresh(now) {
                continue;
            }
            for model in &source.models {
                let namespace = Some(namespace.as_str())
                    .filter(|&namespace| namespace != relay_namespace)
                    .or_else(|| relay_home(&homes, &model.model, relay_namespace))
                    .unwrap_or(namespace);
                let aggregate = models.entry((namespace, model.model.as_str())).or_default();
                aggregate.ready_frontends += u64::from(source.serving_ready && model.is_ready);
                aggregate.gauges.add(&model.gauges);
            }
        }
        let mut coverage = HashMap::<&str, NamespaceCoverage>::new();
        models
            .into_iter()
            .map(|((namespace, model), aggregate)| {
                let relay_load = relay_home(&homes, model, relay_namespace) == Some(namespace);
                let mut model_coverage =
                    self.cached_coverage(&mut coverage, namespace, registered, now);
                let mut totals = self.totals(namespace, model);
                if relay_load {
                    model_coverage = model_coverage.and(self.cached_coverage(
                        &mut coverage,
                        relay_namespace,
                        registered,
                        now,
                    ));
                    totals.add(&self.totals(relay_namespace, model));
                }
                ModelServingLoad {
                    namespace: namespace.to_string(),
                    model: model.to_string(),
                    gauges: match model_coverage.status() {
                        Coverage::Complete(()) => Coverage::Complete(aggregate.gauges),
                        Coverage::Partial => Coverage::Partial,
                        Coverage::Missing => Coverage::Missing,
                    },
                    totals,
                    source_observed_ms: model_coverage.source_observed_ms.unwrap_or_default(),
                    expected_frontends: model_coverage.expected,
                    observed_frontends: model_coverage.observed,
                    ready_frontends: aggregate.ready_frontends,
                    serving_pools: aggregate.serving_pools.into_iter().collect(),
                }
            })
            .collect()
    }

    fn cached_coverage<'a>(
        &self,
        cache: &mut HashMap<&'a str, NamespaceCoverage>,
        namespace: &'a str,
        registered: &HashMap<String, HashSet<u64>>,
        now: Instant,
    ) -> NamespaceCoverage {
        *cache
            .entry(namespace)
            .or_insert_with(|| self.coverage(namespace, registered.get(namespace), now))
    }

    fn totals(&self, namespace: &str, model: &str) -> RequestTotals {
        self.totals
            .get(namespace)
            .and_then(|totals| totals.get(model))
            .copied()
            .unwrap_or_default()
    }

    /// Expected frontends are every frontend in the namespace with a fresh
    /// frame plus the `registered` publishers not heard yet, so broker-mode
    /// frontends (never registered) still count and a registered publisher
    /// that went silent stops counting once its frame expires.
    fn coverage(
        &self,
        namespace: &str,
        registered: Option<&HashSet<u64>>,
        now: Instant,
    ) -> NamespaceCoverage {
        let mut coverage = NamespaceCoverage {
            roster_known: registered.is_some(),
            ..NamespaceCoverage::default()
        };
        let mut known = HashSet::new();
        for ((source_namespace, _), source) in &self.frontends {
            if source_namespace != namespace {
                continue;
            }
            known.insert(source.publisher_id);
            if source.is_fresh(now) {
                coverage.observed += 1;
                coverage.expected += 1;
                coverage.source_observed_ms = Some(
                    coverage
                        .source_observed_ms
                        .map_or(source.published_at_ms, |at| at.min(source.published_at_ms)),
                );
            } else if registered.is_some_and(|ids| ids.contains(&source.publisher_id)) {
                coverage.expected += 1;
            }
        }
        if let Some(registered) = registered {
            coverage.expected += registered.difference(&known).count() as u64;
        }
        coverage
    }
}

/// The one other namespace whose catalog pools register `model`, where load
/// heard in `relay_namespace` counts.
fn relay_home<'a>(
    homes: &HashMap<&str, BTreeSet<&'a str>>,
    model: &str,
    relay_namespace: &str,
) -> Option<&'a str> {
    let homes = homes.get(model)?;
    match homes.first() {
        Some(&home) if homes.len() == 1 && home != relay_namespace => Some(home),
        _ => None,
    }
}

/// Frontend coverage of one namespace, or of several combined.
#[derive(Clone, Copy, Default)]
struct NamespaceCoverage {
    /// Whether every namespace's roster is known; without it a silent
    /// registered frontend cannot be ruled out.
    roster_known: bool,
    expected: u64,
    observed: u64,
    /// Oldest publish time among the fresh frontends.
    source_observed_ms: Option<u64>,
}

impl NamespaceCoverage {
    fn and(self, other: Self) -> Self {
        Self {
            roster_known: self.roster_known && other.roster_known,
            expected: self.expected + other.expected,
            observed: self.observed + other.observed,
            source_observed_ms: match (self.source_observed_ms, other.source_observed_ms) {
                (Some(left), Some(right)) => Some(left.min(right)),
                (left, right) => left.or(right),
            },
        }
    }

    fn status(&self) -> Coverage<()> {
        if self.roster_known && self.expected > 0 && self.observed == self.expected {
            Coverage::Complete(())
        } else if self.observed > 0 {
            Coverage::Partial
        } else {
            Coverage::Missing
        }
    }
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
    const RELAY_NS: &str = "relay";
    const FRONTEND: u64 = 10;
    const PUBLISHER: u64 = 1;

    fn frame(incarnation: u64, sequence: u64, models: Vec<FrontendModelLoad>) -> FrontendLoadFrame {
        FrontendLoadFrame {
            frontend_instance_id: FRONTEND,
            incarnation,
            sequence,
            is_serving_ready: true,
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
            is_ready: true,
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
        let mut models =
            reports.model_loads(&catalog(Vec::new()), RELAY_NS, discovered, Instant::now());
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
            frame(1, 1, vec![model_load("a", 2, 5)]),
        );
        record(
            &mut reports,
            PUBLISHER,
            frame(1, 0, vec![model_load("a", 9, 9)]),
        );
        record(
            &mut reports,
            PUBLISHER,
            frame(1, 1, vec![model_load("a", 9, 9)]),
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

    /// An alias leaving the frame lowers the frontend's total; that is not a
    /// restart, so nothing is counted again.
    #[test]
    fn totals_that_drop_within_an_incarnation_count_nothing_again() {
        let mut reports = FrontendReports::default();
        record(
            &mut reports,
            PUBLISHER,
            frame(1, 1, vec![model_load("a", 1, 1500)]),
        );
        record(
            &mut reports,
            PUBLISHER,
            frame(1, 2, vec![model_load("a", 1, 1000)]),
        );
        record(
            &mut reports,
            PUBLISHER,
            frame(1, 3, vec![model_load("a", 1, 1001)]),
        );

        let model = only_model(&reports, &listed(&[PUBLISHER]));
        assert_eq!(model.totals.requests_started, 1501);
    }

    #[test]
    fn incarnation_flip_flop_does_not_double_count_late_frames() {
        let mut reports = FrontendReports::default();
        record(
            &mut reports,
            PUBLISHER,
            frame(1, 1, vec![model_load("a", 1, 5)]),
        );
        record(&mut reports, 2, frame(2, 1, vec![model_load("a", 1, 3)]));
        // A late frame of the replaced process must not look like another restart.
        record(
            &mut reports,
            PUBLISHER,
            frame(1, 2, vec![model_load("a", 1, 6)]),
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
        let models =
            reports.model_loads(&catalog(Vec::new()), RELAY_NS, &discovered, Instant::now());
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
        other.is_serving_ready = false;
        record(&mut reports, 2, other);

        let models =
            reports.model_loads(&catalog(Vec::new()), RELAY_NS, &listed(&[]), Instant::now());
        let (a, b) = (&models[0], &models[1]);
        assert_eq!(a.observed_frontends, 2);
        assert_eq!(a.ready_frontends, 1);
        assert_eq!(awaiting(a), Some(2));
        assert_eq!(b.ready_frontends, 0);
        assert_eq!(awaiting(b), Some(4));
    }

    /// A ready frontend is not ready for a model it cannot serve yet.
    #[test]
    fn ready_frontends_count_only_models_ready_to_serve() {
        let mut reports = FrontendReports::default();
        let mut not_ready = model_load("a", 0, 0);
        not_ready.is_ready = false;
        record(&mut reports, PUBLISHER, frame(1, 1, vec![not_ready]));

        let model = only_model(&reports, &listed(&[PUBLISHER]));
        assert_eq!((model.observed_frontends, model.ready_frontends), (1, 0));
    }

    #[test]
    fn catalog_models_without_frontends_are_unavailable_with_zero_totals() {
        let reports = FrontendReports::default();
        let catalog = catalog(vec![pool(NS, "a")]);
        let models = reports.model_loads(&catalog, RELAY_NS, &listed(&[]), Instant::now());
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

    /// A frontend without an exact namespace publishes in the Relay's
    /// namespace; its load counts under the namespace that serves the model.
    #[test]
    fn relay_namespace_frontends_count_under_the_models_catalog_namespace() {
        let mut reports = FrontendReports::default();
        reports.record(
            RELAY_NS.to_string(),
            event(PUBLISHER, frame(1, 1, vec![model_load("a", 2, 5)])),
        );
        let catalog = catalog(vec![pool(NS, "a")]);
        let registered = HashMap::from([
            (NS.to_string(), HashSet::new()),
            (RELAY_NS.to_string(), HashSet::from([PUBLISHER])),
        ]);

        let models = reports.model_loads(&catalog, RELAY_NS, &registered, Instant::now());
        assert_eq!(models.len(), 1);
        let model = &models[0];
        assert_eq!((model.namespace.as_str(), model.model.as_str()), (NS, "a"));
        assert_eq!(awaiting(model), Some(2));
        assert_eq!(model.totals.requests_started, 5);
        assert_eq!((model.expected_frontends, model.observed_frontends), (1, 1));
        assert_eq!(model.serving_pools, [catalog.pools()[0].pool_id()]);
    }

    /// One frontend serving a name the wire rejects must not poison the window:
    /// the Relay drops that model and keeps the rest.
    #[test]
    fn invalid_model_names_are_dropped_at_the_trust_boundary() {
        let mut reports = FrontendReports::default();
        record(
            &mut reports,
            PUBLISHER,
            frame(
                1,
                1,
                vec![model_load("llama ", 1, 1), model_load("a", 2, 5)],
            ),
        );

        let model = only_model(&reports, &listed(&[PUBLISHER]));
        assert_eq!(model.model, "a");
        assert_eq!(awaiting(&model), Some(2));
    }

    /// A live frontend the Relay forgot, or first hears mid-stream, adds only
    /// the growth it reports from then on.
    #[test]
    fn a_frontend_first_heard_mid_stream_counts_only_later_growth() {
        let mut reports = FrontendReports::default();
        record(
            &mut reports,
            PUBLISHER,
            frame(1, 1, vec![model_load("a", 1, 100)]),
        );
        let later = Instant::now() + SOURCE_RETENTION + Duration::from_secs(1);
        reports.prune(|_, _| false, later);
        assert!(reports.frontends.is_empty());

        record(
            &mut reports,
            PUBLISHER,
            frame(1, 7, vec![model_load("a", 1, 101)]),
        );
        record(
            &mut reports,
            PUBLISHER,
            frame(1, 8, vec![model_load("a", 1, 103)]),
        );
        assert_eq!(reports.totals(NS, "a").requests_started, 102);
    }

    #[test]
    fn prune_forgets_only_long_silent_unregistered_frontends() {
        let mut reports = FrontendReports::default();
        record(
            &mut reports,
            PUBLISHER,
            frame(1, 1, vec![model_load("a", 1, 5)]),
        );
        let later = Instant::now() + SOURCE_RETENTION + Duration::from_secs(1);

        reports.prune(|_, publisher| publisher == PUBLISHER, later);
        assert_eq!(reports.frontends.len(), 1);
        reports.prune(|_, _| false, later);
        assert!(reports.frontends.is_empty());
        // Totals outlive the frontend.
        assert_eq!(reports.totals(NS, "a").requests_started, 5);
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
