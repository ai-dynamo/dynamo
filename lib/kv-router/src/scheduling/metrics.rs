// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Full-distribution histograms for the scheduler admission path.
//!
//! These record every occurrence, not just ones that cross a DYN_STALL_OP_WARN_MS log
//! threshold -- the WARN-based `dynamo_stall_op` tracing only samples the tail (whatever
//! happened to be slow enough to log), which is not a distribution. These histograms give
//! percentiles over the full population, at negligible per-call cost (a lock-free atomic
//! bucket increment, no formatting or I/O).

#[cfg(feature = "metrics")]
use std::sync::OnceLock;

#[cfg(feature = "metrics")]
use prometheus::{HistogramOpts, HistogramVec};

/// Buckets for CPU-bound admission compute (project_worker_loads + select_worker).
/// Mirrors `compute_overhead_buckets` in dynamo-llm's router metrics.
#[cfg(feature = "metrics")]
fn admission_compute_buckets() -> Vec<f64> {
    prometheus::exponential_buckets(0.001, 2.0, 15).unwrap()
}

/// Buckets for queue-wait (an async wait, can run much longer under real backpressure).
/// Mirrors `async_overhead_buckets` in dynamo-llm's router metrics.
#[cfg(feature = "metrics")]
fn queue_wait_buckets() -> Vec<f64> {
    prometheus::exponential_buckets(0.01, 3.0, 17).unwrap()
}

/// Buckets for actor mailbox/poll timing (can be sub-millisecond when healthy, seconds
/// under real backlog/starvation).
#[cfg(feature = "metrics")]
fn actor_timing_buckets() -> Vec<f64> {
    prometheus::exponential_buckets(0.01, 3.0, 17).unwrap()
}

#[cfg(feature = "metrics")]
pub struct SchedulingMetrics {
    /// project_worker_loads + select_worker, labeled by worker_type ("prefill"/"decode").
    /// Busy-time: this runs synchronously on the scheduler actor task, no `.await` inside.
    pub admission_compute_ms: HistogramVec,
    /// Time a request sits parked in the pending heap before admission is attempted,
    /// labeled by worker_type. Wait-time: nothing computing, just queued.
    pub queue_wait_ms: HistogramVec,
    /// Time from `AdmissionCommand` creation (right before `admission_tx.send`) to the
    /// actor's `run()` loop receiving it via `rx.recv().await`. Direct evidence of actor
    /// mailbox backlog / poll starvation, independent of pending-heap capacity waits.
    pub actor_mailbox_wait_ms: HistogramVec,
    /// Time between the actor's `run()` loop receiving consecutive commands. Large gaps
    /// with a non-empty mailbox mean the actor task itself is not getting runtime time;
    /// with an empty mailbox this is legitimate idle time, so read alongside mailbox depth.
    pub actor_poll_gap_ms: HistogramVec,
    /// Full `admit_one` body (project_worker_loads + select_worker + booking), covering
    /// every return path. Superset of `admission_compute_ms` (which stops after
    /// select_worker); the gap between the two is booking cost.
    pub schedule_compute_ms: HistogramVec,
    /// Age of a pending-heap entry sampled each time `handle_update` confirms it is still
    /// blocked by `all_workers_prefill_busy` (i.e. genuinely capacity-blocked, not just
    /// waiting for the actor to be scheduled). Complements `queue_wait_ms`, which only
    /// observes the terminal wait once a request is finally admitted.
    pub pending_queue_age_ms: HistogramVec,
    /// Time for `queue.update()` to complete after a worker state change
    /// (prefill-complete / free), labeled by event ("prefill_completed"/"free").
    /// Measures the async scheduler actor round-trip only; the synchronous slots
    /// call that precedes it is not included.
    pub worker_state_update_to_scheduler_ms: HistogramVec,
}

#[cfg(feature = "metrics")]
static SCHEDULING_METRICS: OnceLock<SchedulingMetrics> = OnceLock::new();

#[cfg(feature = "metrics")]
impl SchedulingMetrics {
    fn new() -> Result<Self, prometheus::Error> {
        Ok(Self {
            admission_compute_ms: HistogramVec::new(
                HistogramOpts::new(
                    "dynamo_router_overhead_admission_compute_ms",
                    "Time in project_worker_loads + select_worker on the scheduler actor task, in milliseconds",
                )
                .buckets(admission_compute_buckets()),
                &["worker_type"],
            )?,
            queue_wait_ms: HistogramVec::new(
                HistogramOpts::new(
                    "dynamo_router_overhead_queue_wait_ms",
                    "Time a request sits parked in the pending heap before admission, in milliseconds",
                )
                .buckets(queue_wait_buckets()),
                &["worker_type"],
            )?,
            actor_mailbox_wait_ms: HistogramVec::new(
                HistogramOpts::new(
                    "dynamo_router_actor_mailbox_wait_ms",
                    "Time from enqueueing router work to the router actor starting it, in milliseconds",
                )
                .buckets(actor_timing_buckets()),
                &["worker_type", "command"],
            )?,
            actor_poll_gap_ms: HistogramVec::new(
                HistogramOpts::new(
                    "dynamo_router_actor_poll_gap_ms",
                    "Time between consecutive router actor mailbox receives, in milliseconds",
                )
                .buckets(actor_timing_buckets()),
                &["worker_type"],
            )?,
            schedule_compute_ms: HistogramVec::new(
                HistogramOpts::new(
                    "dynamo_router_schedule_compute_ms",
                    "Full admit_one body (project_worker_loads + select_worker + booking) CPU time, in milliseconds",
                )
                .buckets(admission_compute_buckets()),
                &["worker_type"],
            )?,
            pending_queue_age_ms: HistogramVec::new(
                HistogramOpts::new(
                    "dynamo_router_pending_queue_age_ms",
                    "Age of a pending request sampled while still capacity-blocked, in milliseconds",
                )
                .buckets(queue_wait_buckets()),
                &["worker_type"],
            )?,
            worker_state_update_to_scheduler_ms: HistogramVec::new(
                HistogramOpts::new(
                    "dynamo_worker_state_update_to_scheduler_ms",
                    "Time for queue.update() to complete after a worker state change (prefill-complete/free), in milliseconds. Measures async actor round-trip only.",
                )
                .buckets(actor_timing_buckets()),
                &["event"],
            )?,
        })
    }

    /// Get or create the metrics, memoized. Safe to call before registration; observations
    /// before `register()` runs are just not exported until a registry is wired up.
    pub fn get_or_init() -> &'static SchedulingMetrics {
        SCHEDULING_METRICS.get_or_init(|| Self::new().expect("scheduling metrics"))
    }

    pub fn observe_admission_compute(&self, worker_type: &str, ms: u128) {
        self.admission_compute_ms
            .with_label_values(&[worker_type])
            .observe(ms as f64);
    }

    pub fn observe_queue_wait(&self, worker_type: &str, ms: u64) {
        self.queue_wait_ms
            .with_label_values(&[worker_type])
            .observe(ms as f64);
    }

    pub fn observe_actor_mailbox_wait(&self, worker_type: &str, command: &str, ms: f64) {
        self.actor_mailbox_wait_ms
            .with_label_values(&[worker_type, command])
            .observe(ms);
    }

    pub fn observe_actor_poll_gap(&self, worker_type: &str, ms: f64) {
        self.actor_poll_gap_ms
            .with_label_values(&[worker_type])
            .observe(ms);
    }

    pub fn observe_schedule_compute(&self, worker_type: &str, ms: f64) {
        self.schedule_compute_ms
            .with_label_values(&[worker_type])
            .observe(ms);
    }

    pub fn observe_pending_queue_age(&self, worker_type: &str, ms: u64) {
        self.pending_queue_age_ms
            .with_label_values(&[worker_type])
            .observe(ms as f64);
    }

    pub fn observe_worker_state_update_to_scheduler(&self, event: &str, ms: f64) {
        self.worker_state_update_to_scheduler_ms
            .with_label_values(&[event])
            .observe(ms);
    }
}

/// Register scheduling metrics with the given registry. Idempotent via the underlying
/// `OnceLock`-memoized histograms; safe to call once per process from frontend HTTP setup.
#[cfg(feature = "metrics")]
pub fn register_scheduling_metrics(registry: &prometheus::Registry) -> Result<(), prometheus::Error> {
    let m = SchedulingMetrics::get_or_init();
    registry.register(Box::new(m.admission_compute_ms.clone()))?;
    registry.register(Box::new(m.queue_wait_ms.clone()))?;
    registry.register(Box::new(m.actor_mailbox_wait_ms.clone()))?;
    registry.register(Box::new(m.actor_poll_gap_ms.clone()))?;
    registry.register(Box::new(m.schedule_compute_ms.clone()))?;
    registry.register(Box::new(m.pending_queue_age_ms.clone()))?;
    registry.register(Box::new(m.worker_state_update_to_scheduler_ms.clone()))?;
    Ok(())
}

