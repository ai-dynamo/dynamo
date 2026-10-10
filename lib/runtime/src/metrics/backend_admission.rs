// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prometheus metrics for the backend admission gate, under
//! [`METRIC_PREFIX`].
//!
//! One instance per gate rather than the process-global statics the other
//! metric modules use, so a test that builds its own gate never observes
//! another's counts, and so the collectors registered for scraping are exactly
//! the ones that gate updates. [`crate::admission_gate`] owns one and drives it
//! from its authoritative state under the gate lock, so no transition can drift
//! a gauge.
//!
//! The three gauges are the gate's occupancy. `engine_request_count` is
//! admitted requests that have not finished, including those that have already
//! produced their first response; `engine_wait_count` is the subset still
//! awaiting their first engine response; `request_queue_count` is requests
//! waiting to be admitted, which hold neither. The limits are configuration
//! rather than state, so they are not exported.
//!
//! Every request the gate receives is counted once by
//! [`REQUEST_RECEIVE_TOTAL`], and once more by [`REQUEST_ADMIT_TOTAL`] if it is
//! passed into the engine, labelled by whether it waited in the gate queue
//! first. A request refused or cancelled before that handoff is counted by the
//! rejection or cancellation counter instead. Cancellation is counted on its
//! own: the caller went away, which is neither a rejection nor an overload.
//!
//! The TCP request plane's own pool saturation metrics are an unrelated family
//! and stay in [`super::work_handler_pool`].

use prometheus::{IntCounter, IntCounterVec, IntGauge, Opts};

use super::prometheus_names::clamp_u64_to_i64;
use crate::MetricsRegistry;

/// Metric names for this family, kept here rather than in
/// [`super::prometheus_names`] because that module is the source for generated
/// Python constants and these are not needed there.
const METRIC_PREFIX: &str = "dynamo_backend_admission";
const ENGINE_REQUEST_COUNT: &str = "engine_request_count";
const ENGINE_WAIT_COUNT: &str = "engine_wait_count";
const REQUEST_QUEUE_COUNT: &str = "request_queue_count";
const REQUEST_RECEIVE_TOTAL: &str = "request_receive_total";
const REQUEST_ADMIT_TOTAL: &str = "request_admit_total";
const REJECTION_TOTAL: &str = "rejection_total";
const CANCELLATION_TOTAL: &str = "cancellation_total";

/// How a request passed into the engine reached it.
const SOURCE_LABEL: &str = "source";
/// Admitted without waiting in the gate queue.
const SOURCE_DIRECT: &str = "direct";
/// Admitted after waiting in the gate queue.
const SOURCE_QUEUE: &str = "queue";

/// How a request passed into the engine reached it: the `source` label of
/// [`REQUEST_ADMIT_TOTAL`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum AdmissionSource {
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
pub(crate) struct BackendAdmissionMetrics {
    engine_request_count: IntGauge,
    engine_wait_count: IntGauge,
    request_queue_count: IntGauge,
    receives: IntCounter,
    admissions: IntCounterVec,
    rejections: IntCounter,
    cancellations: IntCounter,
}

impl BackendAdmissionMetrics {
    pub(crate) fn new() -> Self {
        let gauge = |suffix, help: &str| {
            IntGauge::new(metric_name(suffix), help).expect("backend admission gauge")
        };
        let counter = |suffix, help: &str| {
            IntCounter::new(metric_name(suffix), help).expect("backend admission counter")
        };
        let admissions = IntCounterVec::new(
            Opts::new(
                metric_name(REQUEST_ADMIT_TOTAL),
                "Requests passed into the engine by the backend admission gate, by whether \
                 they waited in its queue",
            ),
            &[SOURCE_LABEL],
        )
        .expect("backend admission counter");
        // A counter vec has no child until it is labelled, so a gate that has
        // admitted nothing yet would expose no series at all. Create both at
        // zero, so a rate() over either reads as quiet rather than as a gap.
        for source in [SOURCE_DIRECT, SOURCE_QUEUE] {
            admissions.with_label_values(&[source]);
        }
        Self {
            engine_request_count: gauge(
                ENGINE_REQUEST_COUNT,
                "Requests admitted to the engine that have not finished, including those that \
                 have already produced their first response",
            ),
            engine_wait_count: gauge(
                ENGINE_WAIT_COUNT,
                "Requests admitted to the engine that are still awaiting their first engine \
                 response",
            ),
            request_queue_count: gauge(
                REQUEST_QUEUE_COUNT,
                "Requests currently waiting in the backend admission queue",
            ),
            receives: counter(
                REQUEST_RECEIVE_TOTAL,
                "Requests received by the backend admission gate",
            ),
            admissions,
            rejections: counter(
                REJECTION_TOTAL,
                "Requests rejected by the backend admission gate because they could not be \
                 admitted and its queue was full",
            ),
            cancellations: counter(
                CANCELLATION_TOTAL,
                "Requests cancelled before backend admission",
            ),
        }
    }

    /// Publish the occupancy gauges. The caller passes its authoritative counts
    /// rather than stepping these per transition, so no transition can drift
    /// them.
    pub(crate) fn set_occupancy(&self, engine_requests: usize, engine_waits: usize, queued: usize) {
        self.engine_request_count.set(gauge_value(engine_requests));
        self.engine_wait_count.set(gauge_value(engine_waits));
        self.request_queue_count.set(gauge_value(queued));
    }

    /// Count one request entering the gate, whatever then becomes of it.
    pub(crate) fn received(&self) {
        self.receives.inc();
    }

    /// Count one request at its actual engine handoff, as the gate is about to
    /// poll `generate` for it. Being offered engine capacity alone is not an
    /// admission, so a candidate that never reaches the engine — cancelled or
    /// departed — is never counted here, and nothing the engine does afterwards
    /// uncounts one that did.
    pub(crate) fn admitted(&self, source: AdmissionSource) {
        let source = match source {
            AdmissionSource::Direct => SOURCE_DIRECT,
            AdmissionSource::Queue => SOURCE_QUEUE,
        };
        self.admissions.with_label_values(&[source]).inc();
    }

    /// Count one request rejected because it could not be admitted and the
    /// queue had no room for it.
    pub(crate) fn rejected(&self) {
        self.rejections.inc();
    }

    /// Count one request cancelled before backend admission, whether it was
    /// already cancelled when the gate looked or went away while queued.
    pub(crate) fn cancelled(&self) {
        self.cancellations.inc();
    }

    /// Expose this instance's collectors for scraping.
    pub(crate) fn register(&self, registry: &MetricsRegistry) {
        let collectors: [(Box<dyn prometheus::core::Collector>, &str); 7] = [
            (
                Box::new(self.engine_request_count.clone()),
                ENGINE_REQUEST_COUNT,
            ),
            (Box::new(self.engine_wait_count.clone()), ENGINE_WAIT_COUNT),
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

/// Read-back for the gate's own tests, so the collectors stay private to this
/// module.
#[cfg(test)]
impl BackendAdmissionMetrics {
    /// Published as (engine requests, engine waits, queued).
    pub(crate) fn published(&self) -> (i64, i64, i64) {
        (
            self.engine_request_count.get(),
            self.engine_wait_count.get(),
            self.request_queue_count.get(),
        )
    }

    /// Requests received.
    pub(crate) fn receives(&self) -> u64 {
        self.receives.get()
    }

    /// Requests passed into the engine as (direct, queue).
    pub(crate) fn admissions(&self) -> (u64, u64) {
        let count = |source| self.admissions.with_label_values(&[source]).get();
        (count(SOURCE_DIRECT), count(SOURCE_QUEUE))
    }

    /// Requests rejected.
    pub(crate) fn rejections(&self) -> u64 {
        self.rejections.get()
    }

    /// Requests cancelled before admission.
    pub(crate) fn cancellations(&self) -> u64 {
        self.cancellations.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh instance that has counted nothing: the whole family, including
    /// both admission-source series, must already be scrapeable, and a registry
    /// must receive exactly these seven collectors — no limit gauges, and no
    /// label on the rejection counter.
    #[test]
    fn a_registry_receives_exactly_the_whole_family() {
        let metrics = BackendAdmissionMetrics::new();
        assert_eq!(metrics.published(), (0, 0, 0));

        let registry = MetricsRegistry::default();
        metrics.register(&registry);

        let families = registry.get_prometheus_registry().gather();
        let mut names: Vec<&str> = families.iter().map(|family| family.name()).collect();
        names.sort();
        assert_eq!(
            names,
            [
                "dynamo_backend_admission_cancellation_total",
                "dynamo_backend_admission_engine_request_count",
                "dynamo_backend_admission_engine_wait_count",
                "dynamo_backend_admission_rejection_total",
                "dynamo_backend_admission_request_admit_total",
                "dynamo_backend_admission_request_queue_count",
                "dynamo_backend_admission_request_receive_total",
            ]
        );

        // Every counter series exists at zero before its first event, and only
        // admission carries a label.
        assert_eq!(metrics.receives(), 0);
        assert_eq!(metrics.admissions(), (0, 0));
        assert_eq!(metrics.rejections(), 0);
        assert_eq!(metrics.cancellations(), 0);
        // Each series of a family, rendered as its `name=value` labels.
        let series = |name: &str| -> Vec<String> {
            let family = families
                .iter()
                .find(|family| family.name() == name)
                .expect("family is registered");
            family
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
                .collect()
        };
        for unlabelled in [
            "dynamo_backend_admission_cancellation_total",
            "dynamo_backend_admission_rejection_total",
            "dynamo_backend_admission_request_receive_total",
        ] {
            assert_eq!(series(unlabelled), [""], "{unlabelled}");
        }
        assert_eq!(
            series("dynamo_backend_admission_request_admit_total"),
            ["source=direct", "source=queue"]
        );
    }

    /// Each counter and gauge reaches its own series, so no transition can be
    /// attributed to the wrong one.
    #[test]
    fn every_metric_lands_on_its_own_series() {
        let metrics = BackendAdmissionMetrics::new();

        metrics.set_occupancy(3, 2, 1);
        assert_eq!(metrics.published(), (3, 2, 1));

        metrics.received();
        metrics.received();
        assert_eq!(metrics.receives(), 2);

        metrics.admitted(AdmissionSource::Direct);
        metrics.admitted(AdmissionSource::Queue);
        metrics.admitted(AdmissionSource::Queue);
        assert_eq!(metrics.admissions(), (1, 2));

        metrics.rejected();
        assert_eq!(metrics.rejections(), 1);

        metrics.cancelled();
        assert_eq!(metrics.cancellations(), 1);
    }
}
