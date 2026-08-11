// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tokio runtime metrics and event-loop canary

use once_cell::sync::{Lazy, OnceCell};
use prometheus::{Counter, Gauge, Histogram, HistogramOpts, IntCounterVec, IntGaugeVec, Opts};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::runtime::Handle;
use tokio_util::sync::CancellationToken;

use super::prometheus_names::{frontend_perf, name_prefix, tokio_perf as names};
use crate::MetricsRegistry;

const QUEUE_DEPTH_THRESHOLD_PER_WORKER: usize = 4;
const QUEUE_OVERLOAD_DURATION: Duration = Duration::from_secs(15);
const QUEUE_OVERLOAD_LOG_INTERVAL: Duration = Duration::from_secs(5 * 60);

fn tokio_metric_name(suffix: &str) -> String {
    format!("{}_{}", name_prefix::TOKIO, suffix)
}

// --- Tokio runtime gauges/counters (updated every 1s by collector) ---

pub static TOKIO_GLOBAL_QUEUE_DEPTH: Lazy<Gauge> = Lazy::new(|| {
    Gauge::new(
        tokio_metric_name(names::GLOBAL_QUEUE_DEPTH),
        "Number of tasks in the runtime global queue",
    )
    .expect("tokio global_queue_depth gauge")
});

pub static TOKIO_BUDGET_FORCED_YIELD_TOTAL: Lazy<Counter> = Lazy::new(|| {
    Counter::new(
        tokio_metric_name(names::BUDGET_FORCED_YIELD_TOTAL),
        "Number of times tasks were forced to yield after exhausting budget",
    )
    .expect("tokio budget_forced_yield_total counter")
});

pub static TOKIO_BLOCKING_THREADS: Lazy<Gauge> = Lazy::new(|| {
    Gauge::new(
        tokio_metric_name(names::BLOCKING_THREADS),
        "Number of blocking threads",
    )
    .expect("tokio blocking_threads gauge")
});

pub static TOKIO_BLOCKING_IDLE_THREADS: Lazy<Gauge> = Lazy::new(|| {
    Gauge::new(
        tokio_metric_name(names::BLOCKING_IDLE_THREADS),
        "Number of idle blocking threads",
    )
    .expect("tokio blocking_idle_threads gauge")
});

pub static TOKIO_BLOCKING_QUEUE_DEPTH: Lazy<Gauge> = Lazy::new(|| {
    Gauge::new(
        tokio_metric_name(names::BLOCKING_QUEUE_DEPTH),
        "Number of tasks in the blocking thread pool queue",
    )
    .expect("tokio blocking_queue_depth gauge")
});

pub static TOKIO_ALIVE_TASKS: Lazy<Gauge> = Lazy::new(|| {
    Gauge::new(
        tokio_metric_name(names::ALIVE_TASKS),
        "Number of alive tasks in the runtime",
    )
    .expect("tokio alive_tasks gauge")
});

// Per-worker metrics (GaugeVec/IntCounterVec with label "worker")
pub static TOKIO_WORKER_MEAN_POLL_TIME_NS: Lazy<IntGaugeVec> = Lazy::new(|| {
    IntGaugeVec::new(
        Opts::new(
            tokio_metric_name(names::WORKER_MEAN_POLL_TIME_NS),
            "Worker mean task poll time (nanoseconds)",
        ),
        &["worker"],
    )
    .expect("tokio worker_mean_poll_time_ns gauge vec")
});

pub static TOKIO_WORKER_BUSY_RATIO_VEC: Lazy<IntGaugeVec> = Lazy::new(|| {
    IntGaugeVec::new(
        Opts::new(
            tokio_metric_name(names::WORKER_BUSY_RATIO),
            "Worker busy ratio (0-1) as integer mill ratio; >950 = saturated",
        ),
        &["worker"],
    )
    .expect("tokio worker_busy_ratio vec")
});

pub static TOKIO_WORKER_PARK_COUNT_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    IntCounterVec::new(
        Opts::new(
            tokio_metric_name(names::WORKER_PARK_COUNT_TOTAL),
            "Total number of times worker has parked",
        ),
        &["worker"],
    )
    .expect("tokio worker_park_count_total")
});

pub static TOKIO_WORKER_LOCAL_QUEUE_DEPTH: Lazy<IntGaugeVec> = Lazy::new(|| {
    IntGaugeVec::new(
        Opts::new(
            tokio_metric_name(names::WORKER_LOCAL_QUEUE_DEPTH),
            "Number of tasks in worker local queue",
        ),
        &["worker"],
    )
    .expect("tokio worker_local_queue_depth")
});

pub static TOKIO_WORKER_STEAL_COUNT_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    IntCounterVec::new(
        Opts::new(
            tokio_metric_name(names::WORKER_STEAL_COUNT_TOTAL),
            "Total number of tasks stolen by worker",
        ),
        &["worker"],
    )
    .expect("tokio worker_steal_count_total")
});

pub static TOKIO_WORKER_OVERFLOW_COUNT_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    IntCounterVec::new(
        Opts::new(
            tokio_metric_name(names::WORKER_OVERFLOW_COUNT_TOTAL),
            "Total number of times worker local queue overflowed",
        ),
        &["worker"],
    )
    .expect("tokio worker_overflow_count_total")
});

// --- Event loop canary ---
pub static EVENT_LOOP_DELAY_SECONDS: Lazy<Histogram> = Lazy::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            format!(
                "{}_{}",
                name_prefix::FRONTEND,
                frontend_perf::EVENT_LOOP_DELAY_SECONDS
            ),
            "Event loop delay canary: drift from 10ms sleep (seconds)",
        )
        .buckets(vec![
            0.0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0,
        ]),
    )
    .expect("event_loop_delay_seconds histogram")
});

pub static EVENT_LOOP_STALL_TOTAL: Lazy<Counter> = Lazy::new(|| {
    Counter::new(
        format!(
            "{}_{}",
            name_prefix::FRONTEND,
            frontend_perf::EVENT_LOOP_STALL_TOTAL
        ),
        "Number of event loop stalls (delay > 5ms)",
    )
    .expect("event_loop_stall_total counter")
});

/// Guards idempotency for the `MetricsRegistry` registration path.
static REGISTERED: OnceCell<()> = OnceCell::new();

/// Guards idempotency for the raw `prometheus::Registry` registration path.
/// Kept separate from `REGISTERED` so that calling `ensure_tokio_perf_metrics_registered`
/// first does not silently prevent the metrics from being registered in the prometheus registry.
static PROMETHEUS_REGISTERED: OnceCell<()> = OnceCell::new();

/// Register tokio perf and canary metrics with the given registry. Idempotent.
pub fn ensure_tokio_perf_metrics_registered(registry: &MetricsRegistry) {
    let _ = REGISTERED.get_or_init(|| {
        registry
            .add_metric(Box::new(TOKIO_GLOBAL_QUEUE_DEPTH.clone()))
            .ok();
        registry
            .add_metric(Box::new(TOKIO_BUDGET_FORCED_YIELD_TOTAL.clone()))
            .ok();
        registry
            .add_metric(Box::new(TOKIO_BLOCKING_THREADS.clone()))
            .ok();
        registry
            .add_metric(Box::new(TOKIO_BLOCKING_IDLE_THREADS.clone()))
            .ok();
        registry
            .add_metric(Box::new(TOKIO_BLOCKING_QUEUE_DEPTH.clone()))
            .ok();
        registry
            .add_metric(Box::new(TOKIO_ALIVE_TASKS.clone()))
            .ok();
        registry
            .add_metric(Box::new(TOKIO_WORKER_MEAN_POLL_TIME_NS.clone()))
            .ok();
        registry
            .add_metric(Box::new(TOKIO_WORKER_BUSY_RATIO_VEC.clone()))
            .ok();
        registry
            .add_metric(Box::new(TOKIO_WORKER_PARK_COUNT_TOTAL.clone()))
            .ok();
        registry
            .add_metric(Box::new(TOKIO_WORKER_LOCAL_QUEUE_DEPTH.clone()))
            .ok();
        registry
            .add_metric(Box::new(TOKIO_WORKER_STEAL_COUNT_TOTAL.clone()))
            .ok();
        registry
            .add_metric(Box::new(TOKIO_WORKER_OVERFLOW_COUNT_TOTAL.clone()))
            .ok();
        registry
            .add_metric(Box::new(EVENT_LOOP_DELAY_SECONDS.clone()))
            .ok();
        registry
            .add_metric(Box::new(EVENT_LOOP_STALL_TOTAL.clone()))
            .ok();
    });
}

/// Register tokio perf and canary metrics with a raw Prometheus registry.
pub fn ensure_tokio_perf_metrics_registered_prometheus(
    registry: &prometheus::Registry,
) -> Result<(), prometheus::Error> {
    if PROMETHEUS_REGISTERED.get().is_some() {
        return Ok(());
    }
    registry.register(Box::new(TOKIO_GLOBAL_QUEUE_DEPTH.clone()))?;
    registry.register(Box::new(TOKIO_BUDGET_FORCED_YIELD_TOTAL.clone()))?;
    registry.register(Box::new(TOKIO_BLOCKING_THREADS.clone()))?;
    registry.register(Box::new(TOKIO_BLOCKING_IDLE_THREADS.clone()))?;
    registry.register(Box::new(TOKIO_BLOCKING_QUEUE_DEPTH.clone()))?;
    registry.register(Box::new(TOKIO_ALIVE_TASKS.clone()))?;
    registry.register(Box::new(TOKIO_WORKER_MEAN_POLL_TIME_NS.clone()))?;
    registry.register(Box::new(TOKIO_WORKER_BUSY_RATIO_VEC.clone()))?;
    registry.register(Box::new(TOKIO_WORKER_PARK_COUNT_TOTAL.clone()))?;
    registry.register(Box::new(TOKIO_WORKER_LOCAL_QUEUE_DEPTH.clone()))?;
    registry.register(Box::new(TOKIO_WORKER_STEAL_COUNT_TOTAL.clone()))?;
    registry.register(Box::new(TOKIO_WORKER_OVERFLOW_COUNT_TOTAL.clone()))?;
    registry.register(Box::new(EVENT_LOOP_DELAY_SECONDS.clone()))?;
    registry.register(Box::new(EVENT_LOOP_STALL_TOTAL.clone()))?;
    let _ = PROMETHEUS_REGISTERED.set(());
    Ok(())
}

/// Run the tokio metrics collector (1s interval) and event-loop canary.
/// Spawn this on the runtime you want to monitor (e.g. primary handle).
/// The loop exits cleanly when `cancel` is triggered.
pub async fn tokio_metrics_and_canary_loop(cancel: CancellationToken) {
    let canary_interval = Duration::from_millis(10);
    let stall_threshold = Duration::from_millis(5);
    let collect_interval = Duration::from_secs(1);
    let mut next_collect = Instant::now() + collect_interval;
    let mut prev_counters = PrevWorkerCounters::new();
    let mut queue_overload = QueueOverloadLogState::default();
    loop {
        let start = Instant::now();
        tokio::select! {
            _ = tokio::time::sleep(canary_interval) => {}
            _ = cancel.cancelled() => {
                tracing::debug!("tokio metrics and canary loop shutting down");
                return;
            }
        }
        let delay = start.elapsed().saturating_sub(canary_interval);
        EVENT_LOOP_DELAY_SECONDS.observe(delay.as_secs_f64());
        if delay > stall_threshold {
            EVENT_LOOP_STALL_TOTAL.inc();
        }
        if Instant::now() >= next_collect {
            let now = Instant::now();
            next_collect = now + collect_interval;
            let queue_depths = sample_tokio_metrics(&mut prev_counters);
            if let Some(overload_duration) = queue_overload.observe(now, &queue_depths) {
                tracing::warn!(
                    worker_count = queue_depths.worker_count,
                    queue_depth_threshold = queue_depths.threshold,
                    global_queue_depth = queue_depths.global,
                    worker_local_queue_depth_total = queue_depths.local,
                    total_queue_depth = queue_depths.total,
                    overload_duration_seconds = overload_duration.as_secs(),
                    "Frontend Tokio runtime may be overloaded: runnable task queue has remained high"
                );
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct QueueDepthSnapshot {
    worker_count: usize,
    threshold: usize,
    global: usize,
    local: usize,
    total: usize,
}

impl QueueDepthSnapshot {
    fn new(worker_count: usize, global: usize, local: usize) -> Self {
        Self {
            worker_count,
            threshold: worker_count.saturating_mul(QUEUE_DEPTH_THRESHOLD_PER_WORKER),
            global,
            local,
            total: global.saturating_add(local),
        }
    }
}

#[derive(Default)]
struct QueueOverloadLogState {
    overload_started_at: Option<Instant>,
    last_warning_at: Option<Instant>,
}

impl QueueOverloadLogState {
    fn observe(&mut self, now: Instant, queue_depths: &QueueDepthSnapshot) -> Option<Duration> {
        if queue_depths.total < queue_depths.threshold {
            self.overload_started_at = None;
            return None;
        }

        let overload_started_at = *self.overload_started_at.get_or_insert(now);
        let overload_duration = now.saturating_duration_since(overload_started_at);
        if overload_duration < QUEUE_OVERLOAD_DURATION {
            return None;
        }

        if self.last_warning_at.is_some_and(|last_warning_at| {
            now.saturating_duration_since(last_warning_at) < QUEUE_OVERLOAD_LOG_INTERVAL
        }) {
            return None;
        }

        self.last_warning_at = Some(now);
        Some(overload_duration)
    }
}

static PREV_BUDGET_FORCED_YIELD: AtomicU64 = AtomicU64::new(0);

/// Per-worker previous samples for the monotonic _TOTAL counters.
/// Owned by the single `tokio_metrics_and_canary_loop` task — no locks needed.
struct PrevWorkerCounters {
    park: Vec<u64>,
    steal: Vec<u64>,
    overflow: Vec<u64>,
}

impl PrevWorkerCounters {
    fn new() -> Self {
        Self {
            park: Vec::new(),
            steal: Vec::new(),
            overflow: Vec::new(),
        }
    }

    fn ensure_capacity(&mut self, num_workers: usize) {
        if self.park.len() < num_workers {
            self.park.resize(num_workers, 0);
            self.steal.resize(num_workers, 0);
            self.overflow.resize(num_workers, 0);
        }
    }
}

fn sample_tokio_metrics(prev: &mut PrevWorkerCounters) -> QueueDepthSnapshot {
    let metrics = Handle::current().metrics();

    let global_queue_depth = metrics.global_queue_depth();
    TOKIO_GLOBAL_QUEUE_DEPTH.set(global_queue_depth as f64);
    let budget = metrics.budget_forced_yield_count();
    let prev_budget = PREV_BUDGET_FORCED_YIELD.swap(budget, Ordering::Relaxed);
    TOKIO_BUDGET_FORCED_YIELD_TOTAL.inc_by((budget.saturating_sub(prev_budget)) as f64);
    TOKIO_BLOCKING_THREADS.set(metrics.num_blocking_threads() as f64);
    TOKIO_BLOCKING_IDLE_THREADS.set(metrics.num_idle_blocking_threads() as f64);
    TOKIO_BLOCKING_QUEUE_DEPTH.set(metrics.blocking_queue_depth() as f64);
    TOKIO_ALIVE_TASKS.set(metrics.num_alive_tasks() as f64);

    let num_workers = metrics.num_workers();
    prev.ensure_capacity(num_workers);
    let mut local_queue_depth: usize = 0;

    for w in 0..num_workers {
        let worker_label = w.to_string();
        let mean_poll = metrics.worker_mean_poll_time(w);
        let worker_local_queue_depth = metrics.worker_local_queue_depth(w);
        local_queue_depth = local_queue_depth.saturating_add(worker_local_queue_depth);

        TOKIO_WORKER_MEAN_POLL_TIME_NS
            .with_label_values(&[&worker_label])
            .set(mean_poll.as_nanos() as i64);

        TOKIO_WORKER_LOCAL_QUEUE_DEPTH
            .with_label_values(&[&worker_label])
            .set(worker_local_queue_depth as i64);

        // Monotonically increasing totals: track deltas so we use inc_by on a Counter.
        let park = metrics.worker_park_count(w);
        TOKIO_WORKER_PARK_COUNT_TOTAL
            .with_label_values(&[&worker_label])
            .inc_by(park.saturating_sub(prev.park[w]));
        prev.park[w] = park;

        let steal = metrics.worker_steal_count(w);
        TOKIO_WORKER_STEAL_COUNT_TOTAL
            .with_label_values(&[&worker_label])
            .inc_by(steal.saturating_sub(prev.steal[w]));
        prev.steal[w] = steal;

        let overflow = metrics.worker_overflow_count(w);
        TOKIO_WORKER_OVERFLOW_COUNT_TOTAL
            .with_label_values(&[&worker_label])
            .inc_by(overflow.saturating_sub(prev.overflow[w]));
        prev.overflow[w] = overflow;

        // Busy ratio: total_busy_duration over 1s interval -> ratio. We don't have delta here;
        // use mean_poll_time as proxy: if high, worker is busy. Store as 0-1000 (per mille).
        let busy_proxy = (mean_poll.as_secs_f64() / 0.001).min(1.0); // 1ms = saturated
        TOKIO_WORKER_BUSY_RATIO_VEC
            .with_label_values(&[&worker_label])
            .set((busy_proxy * 1000.0) as i64);
    }

    QueueDepthSnapshot::new(num_workers, global_queue_depth, local_queue_depth)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_depth_threshold_scales_with_worker_count() {
        assert_eq!(QueueDepthSnapshot::new(1, 0, 0).threshold, 4);
        assert_eq!(QueueDepthSnapshot::new(4, 0, 0).threshold, 16);
        assert_eq!(QueueDepthSnapshot::new(16, 0, 0).threshold, 64);
    }

    #[test]
    fn queue_depth_snapshot_adds_global_and_local_queues() {
        assert_eq!(
            QueueDepthSnapshot::new(4, 7, 9),
            QueueDepthSnapshot {
                worker_count: 4,
                threshold: 16,
                global: 7,
                local: 9,
                total: 16,
            }
        );
    }

    #[test]
    fn queue_overload_requires_sustained_pressure_and_resets() {
        let start = Instant::now();
        let overloaded = QueueDepthSnapshot::new(4, 8, 8);
        let below_threshold = QueueDepthSnapshot::new(4, 7, 8);
        let mut state = QueueOverloadLogState::default();

        assert_eq!(state.observe(start, &overloaded), None);
        assert_eq!(
            state.observe(
                start + QUEUE_OVERLOAD_DURATION - Duration::from_nanos(1),
                &overloaded
            ),
            None
        );
        assert_eq!(
            state.observe(start + QUEUE_OVERLOAD_DURATION, &below_threshold),
            None
        );

        let restarted_at = start + QUEUE_OVERLOAD_DURATION + Duration::from_secs(1);
        assert_eq!(state.observe(restarted_at, &overloaded), None);
        assert_eq!(
            state.observe(restarted_at + QUEUE_OVERLOAD_DURATION, &overloaded),
            Some(QUEUE_OVERLOAD_DURATION)
        );
    }

    #[test]
    fn queue_overload_warnings_are_rate_limited_across_episodes() {
        let start = Instant::now();
        let overloaded = QueueDepthSnapshot::new(1, 2, 2);
        let below_threshold = QueueDepthSnapshot::new(1, 1, 2);
        let mut state = QueueOverloadLogState::default();

        assert_eq!(state.observe(start, &overloaded), None);
        let first_warning_at = start + QUEUE_OVERLOAD_DURATION;
        assert_eq!(
            state.observe(first_warning_at, &overloaded),
            Some(QUEUE_OVERLOAD_DURATION)
        );

        assert_eq!(
            state.observe(first_warning_at + Duration::from_secs(1), &below_threshold),
            None
        );
        let second_episode_at = first_warning_at + Duration::from_secs(2);
        assert_eq!(state.observe(second_episode_at, &overloaded), None);
        assert_eq!(
            state.observe(second_episode_at + QUEUE_OVERLOAD_DURATION, &overloaded),
            None
        );
        assert_eq!(
            state.observe(first_warning_at + QUEUE_OVERLOAD_LOG_INTERVAL, &overloaded),
            Some(QUEUE_OVERLOAD_LOG_INTERVAL - Duration::from_secs(2))
        );
    }
}
