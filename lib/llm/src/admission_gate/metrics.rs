// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prometheus metrics for the backend admission gate, under
//! [`METRIC_PREFIX`].
//!
//! One instance per gate rather than the process-global statics the other
//! metric modules use, so each worker's gate, and each test's, counts only its
//! own requests, and the collectors registered for scraping are exactly the
//! ones that gate updates. The gate, [`super`], owns one and drives it from its
//! authoritative state under the gate lock, so no transition can drift a gauge.
//! A worker's gate creates its family once, as the gate is created, through the
//! runtime's shared metric creation on its worker's component, which adds the
//! component's standard labels to every series and registers each collector on
//! the component's registry. A collector that registry refuses is kept,
//! unregistered, with a warning.
//!
//! Every metric carries a [`DP_RANK`] label: the global data-parallel rank the
//! gate admitted the request under, or [`DP_RANK_UNSPECIFIED`] for a request
//! without a usable rank. Each request contributes to exactly one series of
//! each metric, so summing a worker's series of a metric, including
//! `unspecified`, gives the worker's total; no aggregate series is published.
//!
//! The three gauges are the gate's occupancy. `engine_request_count` is
//! admitted requests that have not finished, including those whose wait for the
//! engine's output has already ended; `engine_request_waiting_count` is the
//! subset still waiting — for a first generated token, or a prefill's
//! completion; `request_queue_count` is requests waiting to be admitted, which
//! hold neither. The limits are
//! configuration rather than state, so they are not exported.
//!
//! Every request the gate receives is counted once by
//! [`REQUEST_RECEIVE_TOTAL`], and once more by [`REQUEST_ADMIT_TOTAL`] if it is
//! passed into the engine, labelled by whether it waited in the gate queue
//! first. A request refused or cancelled before that handoff is counted by the
//! rejection or cancellation counter instead. Cancellation is counted on its
//! own: the caller went away, which is neither a rejection nor an overload.
//!
//! The TCP request plane's own pool saturation metrics are an unrelated family
//! and stay in [`dynamo_runtime::metrics::work_handler_pool`].

use prometheus::{IntCounter, IntCounterVec, IntGauge, IntGaugeVec};

use dynamo_runtime::metrics::prometheus_names::{clamp_u64_to_i64, labels::DP_RANK};
use dynamo_runtime::metrics::{
    MetricOptions, MetricsHierarchy, PrometheusMetric, create_metric_with_options,
};

/// Metric names for this family, kept here rather than in
/// [`dynamo_runtime::metrics::prometheus_names`] because that module is the
/// source for generated Python constants and these are not needed there.
const METRIC_PREFIX: &str = "dynamo_backend_admission";
const ENGINE_REQUEST_COUNT: &str = "engine_request_count";
const ENGINE_REQUEST_WAITING_COUNT: &str = "engine_request_waiting_count";
const REQUEST_QUEUE_COUNT: &str = "request_queue_count";
const REQUEST_RECEIVE_TOTAL: &str = "request_receive_total";
const REQUEST_ADMIT_TOTAL: &str = "request_admit_total";
const REJECTION_TOTAL: &str = "rejection_total";
const CANCELLATION_TOTAL: &str = "cancellation_total";

/// How each metric of the family is created: named under [`METRIC_PREFIX`],
/// and kept with a warning, rather than failing, when its registry refuses it.
const OPTIONS: MetricOptions = MetricOptions {
    prefix: METRIC_PREFIX,
    warn_on_registration_failure: true,
};

/// The [`DP_RANK`] value of a request admitted without a usable rank.
pub(super) const DP_RANK_UNSPECIFIED: &str = "unspecified";

/// How a request passed into the engine reached it.
const SOURCE_LABEL: &str = "source";
/// Admitted without waiting in the gate queue.
const SOURCE_DIRECT: &str = "direct";
/// Admitted after waiting in the gate queue.
const SOURCE_QUEUE: &str = "queue";

/// How a request passed into the engine reached it: the `source` label of
/// [`REQUEST_ADMIT_TOTAL`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum AdmissionSource {
    /// Both engine limits had room and nothing was queued ahead of it.
    Direct,
    /// It waited in the gate queue first.
    Queue,
}

/// Gate counts are `usize`; Prometheus gauges are `i64`. Saturating rather than
/// wrapping, via the shared helper, so an absurd value cannot report as negative.
fn gauge_value(count: usize) -> i64 {
    clamp_u64_to_i64(count as u64)
}

/// One metric of the family on `hierarchy`, by `label_names`, created by the
/// runtime's shared metric creation under [`OPTIONS`].
fn create<T: PrometheusMetric>(
    hierarchy: &dyn MetricsHierarchy,
    name: &str,
    help: &str,
    label_names: &[&str],
) -> T {
    create_metric_with_options(hierarchy, OPTIONS, name, help, &[], None, Some(label_names))
        .expect("backend admission metric")
}

/// One gate's Prometheus view, under [`METRIC_PREFIX`].
pub(super) struct BackendAdmissionMetrics {
    engine_request_count: IntGaugeVec,
    engine_request_waiting_count: IntGaugeVec,
    request_queue_count: IntGaugeVec,
    receives: IntCounterVec,
    admissions: IntCounterVec,
    rejections: IntCounterVec,
    cancellations: IntCounterVec,
}

impl BackendAdmissionMetrics {
    /// The family, created on `hierarchy` — the component a worker serves —
    /// with that component's standard labels on every series, and registered
    /// on its registry.
    pub(super) fn new(hierarchy: &dyn MetricsHierarchy) -> Self {
        let gauge = |name, help: &str| -> IntGaugeVec { create(hierarchy, name, help, &[DP_RANK]) };
        let counter =
            |name, help: &str| -> IntCounterVec { create(hierarchy, name, help, &[DP_RANK]) };
        Self {
            engine_request_count: gauge(
                ENGINE_REQUEST_COUNT,
                "Requests admitted to the engine that have not finished, including those whose \
                 wait for the engine's output has ended, by admission DP rank",
            ),
            engine_request_waiting_count: gauge(
                ENGINE_REQUEST_WAITING_COUNT,
                "Requests admitted to the engine still waiting for its output, a first generated \
                 token or a completion, by admission DP rank",
            ),
            request_queue_count: gauge(
                REQUEST_QUEUE_COUNT,
                "Requests currently waiting in the backend admission queue, by admission DP rank",
            ),
            receives: counter(
                REQUEST_RECEIVE_TOTAL,
                "Requests received by the backend admission gate, by admission DP rank",
            ),
            admissions: create(
                hierarchy,
                REQUEST_ADMIT_TOTAL,
                "Requests passed into the engine by the backend admission gate, by \
                 admission DP rank and by whether they waited in its queue",
                &[DP_RANK, SOURCE_LABEL],
            ),
            rejections: counter(
                REJECTION_TOTAL,
                "Requests rejected by the backend admission gate because they could not be \
                 admitted and its queue had no room, by admission DP rank",
            ),
            cancellations: counter(
                CANCELLATION_TOTAL,
                "Requests cancelled before backend admission, by admission DP rank",
            ),
        }
    }

    /// Every series of this family for one [`DP_RANK`] value, created at zero
    /// if it did not exist yet, so a `rate()` over a rank that has seen no
    /// traffic reads as quiet rather than as a gap.
    pub(super) fn series(&self, dp_rank: &str) -> RankSeries {
        RankSeries {
            engine_requests: self.engine_request_count.with_label_values(&[dp_rank]),
            engine_requests_waiting: self
                .engine_request_waiting_count
                .with_label_values(&[dp_rank]),
            queued: self.request_queue_count.with_label_values(&[dp_rank]),
            received: self.receives.with_label_values(&[dp_rank]),
            admitted_direct: self.admissions.with_label_values(&[dp_rank, SOURCE_DIRECT]),
            admitted_queue: self.admissions.with_label_values(&[dp_rank, SOURCE_QUEUE]),
            rejected: self.rejections.with_label_values(&[dp_rank]),
            cancelled: self.cancellations.with_label_values(&[dp_rank]),
        }
    }
}

/// Every series of the family for one admission DP rank, resolved once so the
/// gate never formats a label on its hot path.
pub(super) struct RankSeries {
    engine_requests: IntGauge,
    engine_requests_waiting: IntGauge,
    queued: IntGauge,
    received: IntCounter,
    admitted_direct: IntCounter,
    admitted_queue: IntCounter,
    rejected: IntCounter,
    cancelled: IntCounter,
}

impl RankSeries {
    /// Publish this rank's occupancy. The caller passes its authoritative
    /// counts rather than stepping these per transition, so no transition can
    /// drift them.
    pub(super) fn set_occupancy(&self, engine_requests: usize, waiting: usize, queued: usize) {
        self.engine_requests.set(gauge_value(engine_requests));
        self.engine_requests_waiting.set(gauge_value(waiting));
        self.queued.set(gauge_value(queued));
    }

    /// Count one request entering the gate, whatever then becomes of it.
    pub(super) fn received(&self) {
        self.received.inc();
    }

    /// Count one request at its actual engine handoff, as the gate is about to
    /// poll `generate` for it. Being offered engine capacity alone is not an
    /// admission, so a candidate that never reaches the engine — cancelled or
    /// departed — is never counted here, and nothing the engine does afterwards
    /// uncounts one that did.
    pub(super) fn admitted(&self, source: AdmissionSource) {
        match source {
            AdmissionSource::Direct => self.admitted_direct.inc(),
            AdmissionSource::Queue => self.admitted_queue.inc(),
        }
    }

    /// Count one request rejected because it could not be admitted and the
    /// queue had no room for it.
    pub(super) fn rejected(&self) {
        self.rejected.inc();
    }

    /// Count one request cancelled before backend admission, whether it was
    /// already cancelled when the gate looked, went away while queued, was found
    /// cancelled at its dequeue, or refused capacity it had been offered.
    pub(super) fn cancelled(&self) {
        self.cancelled.inc();
    }
}

/// Read-back for the gate's own tests, so the collectors stay private to this
/// module.
#[cfg(test)]
impl RankSeries {
    /// Published as (engine requests, engine-waiting requests, queued).
    pub(super) fn published(&self) -> (i64, i64, i64) {
        (
            self.engine_requests.get(),
            self.engine_requests_waiting.get(),
            self.queued.get(),
        )
    }

    /// Outcome counters as (received, admitted direct, admitted from the
    /// queue, rejected, cancelled).
    pub(super) fn outcomes(&self) -> (u64, u64, u64, u64, u64) {
        (
            self.received.get(),
            self.admitted_direct.get(),
            self.admitted_queue.get(),
            self.rejected.get(),
            self.cancelled.get(),
        )
    }
}

/// A metrics hierarchy for tests: a level named `name` under `parents`, from
/// the root, with a registry of its own.
#[cfg(test)]
pub(super) struct TestHierarchy {
    name: &'static str,
    parents: Vec<TestHierarchy>,
    registry: dynamo_runtime::MetricsRegistry,
    connection_id: Option<u64>,
}

#[cfg(test)]
impl TestHierarchy {
    /// No parents and no identity, so a family created here carries only its
    /// own labels, and nothing else scrapes it.
    pub(super) fn standalone() -> Self {
        Self::level("", Vec::new(), None)
    }

    fn level(name: &'static str, parents: Vec<TestHierarchy>, connection_id: Option<u64>) -> Self {
        Self {
            name,
            parents,
            registry: Default::default(),
            connection_id,
        }
    }
}

#[cfg(test)]
impl MetricsHierarchy for TestHierarchy {
    fn basename(&self) -> String {
        self.name.to_string()
    }

    fn parent_hierarchies(&self) -> Vec<&dyn MetricsHierarchy> {
        self.parents
            .iter()
            .map(|parent| parent as &dyn MetricsHierarchy)
            .collect()
    }

    fn get_metrics_registry(&self) -> &dynamo_runtime::MetricsRegistry {
        &self.registry
    }

    fn connection_id(&self) -> Option<u64> {
        self.connection_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each series of a gathered family, rendered as its `name=value` labels.
    fn series_labels(families: &[prometheus::proto::MetricFamily], name: &str) -> Vec<String> {
        let family = families
            .iter()
            .find(|family| family.name() == name)
            .expect("family is registered");
        let mut labels: Vec<String> = family
            .get_metric()
            .iter()
            .map(|metric| {
                metric
                    .get_label()
                    .iter()
                    .map(|label| format!("{}={}", label.name(), label.value()))
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .collect();
        labels.sort();
        labels
    }

    /// A registry receives exactly these seven collectors — no limit gauges —
    /// and a rank's series exist at zero as soon as it is resolved, labelled by
    /// `dp_rank` alone, plus `source` on admissions.
    #[test]
    fn a_registry_receives_exactly_the_whole_family() {
        let hierarchy = TestHierarchy::standalone();
        let metrics = BackendAdmissionMetrics::new(&hierarchy);
        let unspecified = metrics.series(DP_RANK_UNSPECIFIED);
        let rank = metrics.series("4");
        assert_eq!(unspecified.published(), (0, 0, 0));
        assert_eq!(rank.outcomes(), (0, 0, 0, 0, 0));

        let families = hierarchy.registry.get_prometheus_registry().gather();
        let mut names: Vec<&str> = families.iter().map(|family| family.name()).collect();
        names.sort();
        assert_eq!(
            names,
            [
                "dynamo_backend_admission_cancellation_total",
                "dynamo_backend_admission_engine_request_count",
                "dynamo_backend_admission_engine_request_waiting_count",
                "dynamo_backend_admission_rejection_total",
                "dynamo_backend_admission_request_admit_total",
                "dynamo_backend_admission_request_queue_count",
                "dynamo_backend_admission_request_receive_total",
            ]
        );
        for name in [
            "dynamo_backend_admission_cancellation_total",
            "dynamo_backend_admission_engine_request_count",
            "dynamo_backend_admission_engine_request_waiting_count",
            "dynamo_backend_admission_rejection_total",
            "dynamo_backend_admission_request_queue_count",
            "dynamo_backend_admission_request_receive_total",
        ] {
            assert_eq!(
                series_labels(&families, name),
                ["dp_rank=4", "dp_rank=unspecified"],
                "{name}"
            );
        }
        assert_eq!(
            series_labels(&families, "dynamo_backend_admission_request_admit_total"),
            [
                "dp_rank=4,source=direct",
                "dp_rank=4,source=queue",
                "dp_rank=unspecified,source=direct",
                "dp_rank=unspecified,source=queue",
            ]
        );
    }

    /// A worker's family, created on the component it serves, keeps its names
    /// and carries that component's standard labels on every series — no
    /// endpoint label, as the family is the worker's — beside `dp_rank`, plus
    /// `source` on admissions.
    #[test]
    fn a_worker_family_carries_its_component_labels_on_every_series() {
        let namespace = TestHierarchy::level("ns", Vec::new(), None);
        let component = TestHierarchy::level(
            "backend",
            vec![TestHierarchy::standalone(), namespace],
            Some(0x1a),
        );
        let metrics = BackendAdmissionMetrics::new(&component);
        metrics.series("4");

        let families = component.registry.get_prometheus_registry().gather();
        assert_eq!(families.len(), 7);
        let owner = "dynamo_component=backend,dynamo_namespace=ns";
        for family in &families {
            let name = family.name();
            assert!(name.starts_with("dynamo_backend_admission_"), "{name}");
            let expected = if name == "dynamo_backend_admission_request_admit_total" {
                vec![
                    format!("dp_rank=4,{owner},source=direct,worker_id=1a"),
                    format!("dp_rank=4,{owner},source=queue,worker_id=1a"),
                ]
            } else {
                vec![format!("dp_rank=4,{owner},worker_id=1a")]
            };
            let series = series_labels(&families, name);
            assert!(
                series
                    .iter()
                    .all(|pairs| !pairs.contains("dynamo_endpoint=")),
                "{name}"
            );
            assert_eq!(series, expected, "{name}");
        }
    }

    /// Each counter and gauge reaches its own series, and one rank's events
    /// never reach another rank's series.
    #[test]
    fn every_metric_lands_on_its_own_series() {
        let metrics = BackendAdmissionMetrics::new(&TestHierarchy::standalone());
        let four = metrics.series("4");
        let five = metrics.series("5");

        four.set_occupancy(3, 2, 1);
        assert_eq!(four.published(), (3, 2, 1));
        assert_eq!(five.published(), (0, 0, 0));

        four.received();
        four.received();
        four.admitted(AdmissionSource::Direct);
        four.admitted(AdmissionSource::Queue);
        four.admitted(AdmissionSource::Queue);
        four.rejected();
        four.cancelled();
        assert_eq!(four.outcomes(), (2, 1, 2, 1, 1));
        assert_eq!(five.outcomes(), (0, 0, 0, 0, 0));

        // Resolving the same rank again reaches the same series.
        assert_eq!(metrics.series("4").outcomes(), (2, 1, 2, 1, 1));
    }
}
