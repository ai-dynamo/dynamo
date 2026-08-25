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
//! The TCP request plane's own pool saturation metrics are an unrelated family
//! and stay in [`super::work_handler_pool`].

use prometheus::{IntCounterVec, IntGauge, Opts};

use super::prometheus_names::clamp_u64_to_i64;
use crate::MetricsRegistry;

/// Metric names for this family, kept here rather than in
/// [`super::prometheus_names`] because that module is the source for generated
/// Python constants and these are not needed there.
const METRIC_PREFIX: &str = "dynamo_backend_admission";
const ACTIVE_REQUESTS: &str = "active_requests";
const QUEUE_DEPTH: &str = "queue_depth";
const CONCURRENCY_LIMIT: &str = "concurrency_limit";
const QUEUE_CAPACITY: &str = "queue_capacity";
const REJECTIONS_TOTAL: &str = "rejections_total";
const REASON_LABEL: &str = "reason";
/// Values for [`REASON_LABEL`]. A cancellation is neither a rejection nor an
/// overload — the caller went away — so it has no value here and is never counted.
const REASON_QUEUE_FULL: &str = "queue_full";
const REASON_QUEUE_TIMEOUT: &str = "queue_timeout";

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
    active_requests: IntGauge,
    queue_depth: IntGauge,
    concurrency_limit: IntGauge,
    queue_capacity: IntGauge,
    rejections: IntCounterVec,
}

impl BackendAdmissionMetrics {
    /// The sizing gauges are published here rather than by the caller, so an
    /// instance is never scrapeable with an unset limit or capacity. The queue
    /// length is fixed for its life; the limit can still move when a late
    /// capacity hint lands.
    pub(crate) fn new(concurrency_limit: usize, queue_capacity: usize) -> Self {
        let gauge = |suffix, help: &str| {
            IntGauge::new(metric_name(suffix), help).expect("backend admission gauge")
        };
        let rejections = IntCounterVec::new(
            Opts::new(
                metric_name(REJECTIONS_TOTAL),
                "Requests refused by the backend admission gate",
            ),
            &[REASON_LABEL],
        )
        .expect("backend admission rejections counter");
        // A counter vec has no child until it is labelled, so a gate that has
        // not refused anything yet would expose no series at all. Create both at
        // zero, so a rate() over them reads as quiet rather than as a gap.
        for reason in [REASON_QUEUE_FULL, REASON_QUEUE_TIMEOUT] {
            rejections.with_label_values(&[reason]);
        }
        let metrics = Self {
            active_requests: gauge(
                ACTIVE_REQUESTS,
                "Requests currently holding an admitted backend slot",
            ),
            queue_depth: gauge(
                QUEUE_DEPTH,
                "Requests currently waiting in the backend admission queue",
            ),
            concurrency_limit: gauge(
                CONCURRENCY_LIMIT,
                "Effective concurrent-request limit of the backend admission gate",
            ),
            queue_capacity: gauge(
                QUEUE_CAPACITY,
                "Configured length of the backend admission queue",
            ),
            rejections,
        };
        metrics.set_concurrency_limit(concurrency_limit);
        metrics.queue_capacity.set(gauge_value(queue_capacity));
        metrics
    }

    /// Publish the occupancy gauges. The caller passes its authoritative counts
    /// rather than stepping these per transition, so no transition can drift
    /// them.
    pub(crate) fn set_occupancy(&self, active: usize, queued: usize) {
        self.active_requests.set(gauge_value(active));
        self.queue_depth.set(gauge_value(queued));
    }

    /// Publish the effective concurrent-request limit, which a late capacity
    /// hint can still change.
    pub(crate) fn set_concurrency_limit(&self, limit: usize) {
        self.concurrency_limit.set(gauge_value(limit));
    }

    /// Count one request shed with the limit and the queue both full.
    pub(crate) fn rejected_queue_full(&self) {
        self.rejected(REASON_QUEUE_FULL);
    }

    /// Count one request given up on for outliving the queue delay.
    pub(crate) fn rejected_queue_timeout(&self) {
        self.rejected(REASON_QUEUE_TIMEOUT);
    }

    /// Count one refusal. Cancellation has no reason value here: the caller
    /// went away, which is neither a rejection nor an overload.
    fn rejected(&self, reason: &str) {
        self.rejections.with_label_values(&[reason]).inc();
    }

    /// Expose this instance's collectors for scraping.
    pub(crate) fn register(&self, registry: &MetricsRegistry) {
        let collectors: [(Box<dyn prometheus::core::Collector>, &str); 5] = [
            (Box::new(self.active_requests.clone()), ACTIVE_REQUESTS),
            (Box::new(self.queue_depth.clone()), QUEUE_DEPTH),
            (Box::new(self.concurrency_limit.clone()), CONCURRENCY_LIMIT),
            (Box::new(self.queue_capacity.clone()), QUEUE_CAPACITY),
            (Box::new(self.rejections.clone()), REJECTIONS_TOTAL),
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
    /// Published as (active, queued, limit, capacity).
    pub(crate) fn published(&self) -> (i64, i64, i64, i64) {
        (
            self.active_requests.get(),
            self.queue_depth.get(),
            self.concurrency_limit.get(),
            self.queue_capacity.get(),
        )
    }

    /// Refusals as (queue_full, queue_timeout).
    pub(crate) fn refusals(&self) -> (u64, u64) {
        let count = |reason| self.rejections.with_label_values(&[reason]).get();
        (count(REASON_QUEUE_FULL), count(REASON_QUEUE_TIMEOUT))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh instance that has counted nothing: the whole family, including
    /// both reason series, must already be scrapeable, and a registry must
    /// receive exactly these five collectors.
    #[test]
    fn a_registry_receives_exactly_the_five_collectors() {
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
                "dynamo_backend_admission_active_requests",
                "dynamo_backend_admission_concurrency_limit",
                "dynamo_backend_admission_queue_capacity",
                "dynamo_backend_admission_queue_depth",
                "dynamo_backend_admission_rejections_total",
            ]
        );
        let mut reasons: Vec<(&str, u64)> = families
            .iter()
            .filter(|family| family.name().ends_with(REJECTIONS_TOTAL))
            .flat_map(|family| family.get_metric())
            .map(|metric| {
                (
                    metric.get_label()[0].value(),
                    metric.get_counter().value() as u64,
                )
            })
            .collect();
        reasons.sort();
        assert_eq!(reasons, [(REASON_QUEUE_FULL, 0), (REASON_QUEUE_TIMEOUT, 0)]);
    }
}
