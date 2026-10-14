// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prometheus metrics for the backend admission gate, under
//! [`METRIC_PREFIX`].
//!
//! One instance per gate rather than the process-global statics the other
//! metric modules use, so a test that builds its own gate never observes
//! another's counts, and so the collectors registered for scraping are exactly
//! the ones that gate updates. The gate, [`super`], owns one and drives it from
//! its authoritative state under the gate lock, so no transition can drift a
//! gauge.
//!
//! Every metric carries a [`DP_RANK`] label: the global data-parallel rank the
//! gate admitted the request under, or [`DP_RANK_UNSPECIFIED`] for a request
//! without a usable rank. Each request contributes to exactly one series of
//! each metric, so summing a metric's series, including `unspecified`, gives
//! the process total; no aggregate series is published.
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

use prometheus::{IntCounter, IntCounterVec, IntGauge, IntGaugeVec, Opts};

use dynamo_runtime::MetricsRegistry;
use dynamo_runtime::metrics::prometheus_names::{clamp_u64_to_i64, labels::DP_RANK};

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

fn metric_name(suffix: &str) -> String {
    format!("{METRIC_PREFIX}_{suffix}")
}

/// Gate counts are `usize`; Prometheus gauges are `i64`. Saturating rather than
/// wrapping, via the shared helper, so an absurd value cannot report as negative.
fn gauge_value(count: usize) -> i64 {
    clamp_u64_to_i64(count as u64)
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
    pub(super) fn new() -> Self {
        let gauge = |suffix, help: &str| {
            IntGaugeVec::new(Opts::new(metric_name(suffix), help), &[DP_RANK])
                .expect("backend admission gauge")
        };
        let counter = |suffix, help: &str| {
            IntCounterVec::new(Opts::new(metric_name(suffix), help), &[DP_RANK])
                .expect("backend admission counter")
        };
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
            admissions: IntCounterVec::new(
                Opts::new(
                    metric_name(REQUEST_ADMIT_TOTAL),
                    "Requests passed into the engine by the backend admission gate, by \
                     admission DP rank and by whether they waited in its queue",
                ),
                &[DP_RANK, SOURCE_LABEL],
            )
            .expect("backend admission counter"),
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

    /// Expose this instance's collectors for scraping.
    pub(super) fn register(&self, registry: &MetricsRegistry) {
        let collectors: [(Box<dyn prometheus::core::Collector>, &str); 7] = [
            (
                Box::new(self.engine_request_count.clone()),
                ENGINE_REQUEST_COUNT,
            ),
            (
                Box::new(self.engine_request_waiting_count.clone()),
                ENGINE_REQUEST_WAITING_COUNT,
            ),
            (
                Box::new(self.request_queue_count.clone()),
                REQUEST_QUEUE_COUNT,
            ),
            (Box::new(self.receives.clone()), REQUEST_RECEIVE_TOTAL),
            (Box::new(self.admissions.clone()), REQUEST_ADMIT_TOTAL),
            (Box::new(self.rejections.clone()), REJECTION_TOTAL),
            (Box::new(self.cancellations.clone()), CANCELLATION_TOTAL),
        ];
        for (collector, name) in collectors {
            registry.add_metric_or_warn(collector, name);
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
        let metrics = BackendAdmissionMetrics::new();
        let registry = MetricsRegistry::default();
        metrics.register(&registry);
        let unspecified = metrics.series(DP_RANK_UNSPECIFIED);
        let rank = metrics.series("4");
        assert_eq!(unspecified.published(), (0, 0, 0));
        assert_eq!(rank.outcomes(), (0, 0, 0, 0, 0));

        let families = registry.get_prometheus_registry().gather();
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

    /// Each counter and gauge reaches its own series, and one rank's events
    /// never reach another rank's series.
    #[test]
    fn every_metric_lands_on_its_own_series() {
        let metrics = BackendAdmissionMetrics::new();
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
