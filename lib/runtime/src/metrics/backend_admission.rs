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
//! The two live counts each pair with the bound that governs them —
//! `engine_request_count` with `engine_request_limit`, `request_queue_count`
//! with `request_queue_limit` — so saturation reads off the family without
//! knowing how the gate is configured.
//!
//! Every request the gate receives is classified exactly once by
//! [`REQUEST_TOTAL`], and the counters that follow it describe what became of
//! the ones that queued. Cancellation is counted on its own: the caller went
//! away, which is neither a rejection nor an overload.
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
const ENGINE_REQUEST_LIMIT: &str = "engine_request_limit";
const REQUEST_QUEUE_COUNT: &str = "request_queue_count";
const REQUEST_QUEUE_LIMIT: &str = "request_queue_limit";
const REQUEST_TOTAL: &str = "request_total";
const DEQUEUE_TOTAL: &str = "dequeue_total";
const REJECTION_TOTAL: &str = "rejection_total";
const CANCELLATION_TOTAL: &str = "cancellation_total";

/// Which admission path a request took. Every request the gate receives lands
/// on exactly one of these.
const PATH_LABEL: &str = "path";
/// Took a backend concurrency slot immediately.
const PATH_DIRECT: &str = "direct";
/// No slot was free, so the request took the queue-admission path. A request
/// shed because the queue was also full counts here too: it is the outcome of
/// the queue path, not a fourth path of its own.
const PATH_QUEUE: &str = "queue";
/// Already cancelled when the gate looked, so neither live path was selected.
const PATH_CANCELLED: &str = "cancelled";

/// Which end of the FIFO a queued request was admitted from.
const SOURCE_LABEL: &str = "source";
const SOURCE_FIFO: &str = "fifo";
const SOURCE_ADAPTIVE_LIFO: &str = "adaptive_lifo";

/// Why the gate refused a request. A cancellation has no value here.
const REASON_LABEL: &str = "reason";
const REASON_QUEUE_FULL: &str = "queue_full";
/// The request outlived its permitted queue residence. The queue did not time
/// out; this one request did.
const REASON_REQUEST_EXPIRED: &str = "request_expired";

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
    engine_request_limit: IntGauge,
    request_queue_count: IntGauge,
    request_queue_limit: IntGauge,
    requests: IntCounterVec,
    dequeues: IntCounterVec,
    rejections: IntCounterVec,
    cancellations: IntCounter,
}

impl BackendAdmissionMetrics {
    /// The sizing gauges are published here rather than by the caller, so an
    /// instance is never scrapeable with an unset limit or queue bound. The
    /// queue bound is fixed for its life; the engine request limit can still
    /// move when a late capacity hint lands.
    pub(crate) fn new(engine_request_limit: usize, request_queue_limit: usize) -> Self {
        let gauge = |suffix, help: &str| {
            IntGauge::new(metric_name(suffix), help).expect("backend admission gauge")
        };
        // A counter vec has no child until it is labelled, so a gate that has
        // counted nothing yet would expose no series at all. Create every series
        // at zero, so a rate() over one reads as quiet rather than as a gap.
        let counter_vec = |suffix, help: &str, label: &str, values: &[&str]| {
            let counter = IntCounterVec::new(Opts::new(metric_name(suffix), help), &[label])
                .expect("backend admission counter");
            for value in values {
                counter.with_label_values(&[value]);
            }
            counter
        };
        let metrics = Self {
            engine_request_count: gauge(
                ENGINE_REQUEST_COUNT,
                "Requests currently holding a backend admission concurrency slot",
            ),
            engine_request_limit: gauge(
                ENGINE_REQUEST_LIMIT,
                "Effective concurrent-request limit of the backend admission gate",
            ),
            request_queue_count: gauge(
                REQUEST_QUEUE_COUNT,
                "Requests currently waiting in the backend admission queue",
            ),
            request_queue_limit: gauge(
                REQUEST_QUEUE_LIMIT,
                "Effective length bound of the backend admission queue",
            ),
            requests: counter_vec(
                REQUEST_TOTAL,
                "Requests received by the backend admission gate, by admission path",
                PATH_LABEL,
                &[PATH_DIRECT, PATH_QUEUE, PATH_CANCELLED],
            ),
            dequeues: counter_vec(
                DEQUEUE_TOTAL,
                "Queued requests given a backend admission concurrency slot, by queue end",
                SOURCE_LABEL,
                &[SOURCE_FIFO, SOURCE_ADAPTIVE_LIFO],
            ),
            rejections: counter_vec(
                REJECTION_TOTAL,
                "Requests refused by the backend admission gate",
                REASON_LABEL,
                &[REASON_QUEUE_FULL, REASON_REQUEST_EXPIRED],
            ),
            cancellations: IntCounter::new(
                metric_name(CANCELLATION_TOTAL),
                "Requests cancelled before backend admission",
            )
            .expect("backend admission cancellations counter"),
        };
        metrics.set_engine_request_limit(engine_request_limit);
        metrics
            .request_queue_limit
            .set(gauge_value(request_queue_limit));
        metrics
    }

    /// Publish the occupancy gauges. The caller passes its authoritative counts
    /// rather than stepping these per transition, so no transition can drift
    /// them.
    pub(crate) fn set_occupancy(&self, engine_requests: usize, queued: usize) {
        self.engine_request_count.set(gauge_value(engine_requests));
        self.request_queue_count.set(gauge_value(queued));
    }

    /// Publish the effective concurrent-request limit, which a late capacity
    /// hint can still change.
    pub(crate) fn set_engine_request_limit(&self, limit: usize) {
        self.engine_request_limit.set(gauge_value(limit));
    }

    /// Count one request admitted straight into a free slot.
    pub(crate) fn received_direct(&self) {
        self.received(PATH_DIRECT);
    }

    /// Count one request that found no free slot and took the queue path,
    /// whether it went on to wait or was shed for a full queue.
    pub(crate) fn received_queue(&self) {
        self.received(PATH_QUEUE);
    }

    /// Count one request that was already cancelled when the gate looked.
    pub(crate) fn received_cancelled(&self) {
        self.received(PATH_CANCELLED);
    }

    fn received(&self, path: &str) {
        self.requests.with_label_values(&[path]).inc();
    }

    /// Count one queued request that took a slot, from the adaptive-LIFO tail
    /// when `from_tail` and from the FIFO front otherwise. Counted where the
    /// request consumes the offer, so a candidate that never becomes an
    /// admission — expired, cancelled or departed — never reaches here.
    pub(crate) fn dequeued(&self, from_tail: bool) {
        let source = if from_tail {
            SOURCE_ADAPTIVE_LIFO
        } else {
            SOURCE_FIFO
        };
        self.dequeues.with_label_values(&[source]).inc();
    }

    /// Count one request shed with the limit and the queue both full.
    pub(crate) fn rejected_queue_full(&self) {
        self.rejected(REASON_QUEUE_FULL);
    }

    /// Count one request given up on for outliving its permitted queue
    /// residence.
    pub(crate) fn rejected_request_expired(&self) {
        self.rejected(REASON_REQUEST_EXPIRED);
    }

    /// Count one refusal. Cancellation has no reason value here: the caller
    /// went away, which is neither a rejection nor an overload.
    fn rejected(&self, reason: &str) {
        self.rejections.with_label_values(&[reason]).inc();
    }

    /// Count one request cancelled before backend admission, whether it was
    /// already cancelled when the gate looked or went away while queued. It
    /// keeps whichever path it was classified under.
    pub(crate) fn cancelled(&self) {
        self.cancellations.inc();
    }

    /// Expose this instance's collectors for scraping.
    pub(crate) fn register(&self, registry: &MetricsRegistry) {
        let collectors: [(Box<dyn prometheus::core::Collector>, &str); 8] = [
            (
                Box::new(self.engine_request_count.clone()),
                ENGINE_REQUEST_COUNT,
            ),
            (
                Box::new(self.engine_request_limit.clone()),
                ENGINE_REQUEST_LIMIT,
            ),
            (
                Box::new(self.request_queue_count.clone()),
                REQUEST_QUEUE_COUNT,
            ),
            (
                Box::new(self.request_queue_limit.clone()),
                REQUEST_QUEUE_LIMIT,
            ),
            (Box::new(self.requests.clone()), REQUEST_TOTAL),
            (Box::new(self.dequeues.clone()), DEQUEUE_TOTAL),
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
    /// Published as (engine requests, queued, engine limit, queue limit).
    pub(crate) fn published(&self) -> (i64, i64, i64, i64) {
        (
            self.engine_request_count.get(),
            self.request_queue_count.get(),
            self.engine_request_limit.get(),
            self.request_queue_limit.get(),
        )
    }

    /// Requests received as (direct, queue, cancelled).
    pub(crate) fn received_paths(&self) -> (u64, u64, u64) {
        let count = |path| self.requests.with_label_values(&[path]).get();
        (count(PATH_DIRECT), count(PATH_QUEUE), count(PATH_CANCELLED))
    }

    /// Queued requests admitted as (fifo, adaptive_lifo).
    pub(crate) fn dequeues(&self) -> (u64, u64) {
        let count = |source| self.dequeues.with_label_values(&[source]).get();
        (count(SOURCE_FIFO), count(SOURCE_ADAPTIVE_LIFO))
    }

    /// Refusals as (queue_full, request_expired).
    pub(crate) fn refusals(&self) -> (u64, u64) {
        let count = |reason| self.rejections.with_label_values(&[reason]).get();
        (count(REASON_QUEUE_FULL), count(REASON_REQUEST_EXPIRED))
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
    /// every label series, must already be scrapeable, and a registry must
    /// receive exactly these eight collectors.
    #[test]
    fn a_registry_receives_exactly_the_whole_family() {
        let metrics = BackendAdmissionMetrics::new(7, 9);
        // Construction publishes the sizing it was given, so nothing is
        // scraped before the gate has said how it is sized.
        assert_eq!(metrics.published(), (0, 0, 7, 9));

        let registry = MetricsRegistry::default();
        metrics.register(&registry);

        let families = registry.get_prometheus_registry().gather();
        let mut names: Vec<&str> = families.iter().map(|family| family.name()).collect();
        names.sort();
        assert_eq!(
            names,
            [
                "dynamo_backend_admission_cancellation_total",
                "dynamo_backend_admission_dequeue_total",
                "dynamo_backend_admission_engine_request_count",
                "dynamo_backend_admission_engine_request_limit",
                "dynamo_backend_admission_rejection_total",
                "dynamo_backend_admission_request_queue_count",
                "dynamo_backend_admission_request_queue_limit",
                "dynamo_backend_admission_request_total",
            ]
        );

        // Every counter series exists at zero before its first event.
        assert_eq!(metrics.received_paths(), (0, 0, 0));
        assert_eq!(metrics.dequeues(), (0, 0));
        assert_eq!(metrics.refusals(), (0, 0));
        assert_eq!(metrics.cancellations(), 0);
        let series: Vec<(&str, usize)> = families
            .iter()
            .filter(|family| family.name().ends_with("_total"))
            .map(|family| (family.name(), family.get_metric().len()))
            .collect();
        assert_eq!(
            series,
            [
                ("dynamo_backend_admission_cancellation_total", 1),
                ("dynamo_backend_admission_dequeue_total", 2),
                ("dynamo_backend_admission_rejection_total", 2),
                ("dynamo_backend_admission_request_total", 3),
            ]
        );
    }

    /// Each counter reaches its own series, so no transition can be attributed
    /// to the wrong one.
    #[test]
    fn every_counter_lands_on_its_own_series() {
        let metrics = BackendAdmissionMetrics::new(1, 1);

        metrics.received_direct();
        metrics.received_queue();
        metrics.received_queue();
        metrics.received_cancelled();
        assert_eq!(metrics.received_paths(), (1, 2, 1));

        metrics.dequeued(false);
        metrics.dequeued(true);
        metrics.dequeued(true);
        assert_eq!(metrics.dequeues(), (1, 2));

        metrics.rejected_queue_full();
        metrics.rejected_request_expired();
        assert_eq!(metrics.refusals(), (1, 1));

        metrics.cancelled();
        assert_eq!(metrics.cancellations(), 1);
    }
}
