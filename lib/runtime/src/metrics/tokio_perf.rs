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
const QUEUE_DEPTH_RECOVERY_THRESHOLD_PER_WORKER: usize = 2;
const QUEUE_OVERLOAD_DURATION: Duration = Duration::from_secs(15);
const QUEUE_OVERLOAD_LOG_INTERVAL: Duration = Duration::from_secs(5 * 60);
const QUEUE_SAMPLE_INTERVAL: Duration = Duration::from_secs(1);
const QUEUE_SAMPLE_MAX_GAP: Duration = Duration::from_secs(2);

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

pub static TOKIO_QUEUE_OVERLOAD_WARNINGS_TOTAL: Lazy<Counter> = Lazy::new(|| {
    Counter::new(
        tokio_metric_name(names::QUEUE_OVERLOAD_WARNINGS_TOTAL),
        "Number of warnings for sustained Tokio runnable queue pressure",
    )
    .expect("tokio queue_overload_warnings_total counter")
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
            .add_metric(Box::new(TOKIO_QUEUE_OVERLOAD_WARNINGS_TOTAL.clone()))
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
    registry.register(Box::new(TOKIO_QUEUE_OVERLOAD_WARNINGS_TOTAL.clone()))?;
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
    let mut next_collect = Instant::now() + QUEUE_SAMPLE_INTERVAL;
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
        let now = Instant::now();
        if now >= next_collect {
            next_collect = now + QUEUE_SAMPLE_INTERVAL;
            let queue_depths = sample_tokio_metrics(&mut prev_counters);
            if let Some(overload_duration) = queue_overload.observe(now, &queue_depths) {
                warn_queue_overload(&queue_depths, overload_duration);
            }
        }
    }
}

fn warn_queue_overload(queue_depths: &QueueDepthSnapshot, overload_duration: Duration) {
    TOKIO_QUEUE_OVERLOAD_WARNINGS_TOTAL.inc();
    tracing::warn!(
        worker_count = queue_depths.worker_count,
        total_queue_depth_threshold = queue_depths.total_threshold(),
        worker_local_queue_depth_threshold = queue_depths.worker_local_threshold(),
        global_queue_depth = queue_depths.global,
        worker_local_queue_depth_total = queue_depths.local_total,
        worker_local_queue_depth_max = queue_depths.local_max,
        total_queue_depth = queue_depths.total(),
        overload_duration_seconds = overload_duration.as_secs(),
        "Tokio runtime may be overloaded: runnable task queues have remained high. Possible causes include CPU contention, blocking work on async workers, long-running task polls, or too few runtime workers."
    );
}

#[derive(Debug)]
struct QueueDepthSnapshot {
    worker_count: usize,
    global: usize,
    local_total: usize,
    local_max: usize,
}

impl QueueDepthSnapshot {
    fn effective_worker_count(&self) -> usize {
        self.worker_count.max(1)
    }

    fn total(&self) -> usize {
        self.global.saturating_add(self.local_total)
    }

    fn total_threshold(&self) -> usize {
        self.effective_worker_count()
            .saturating_mul(QUEUE_DEPTH_THRESHOLD_PER_WORKER)
    }

    fn worker_local_threshold(&self) -> usize {
        QUEUE_DEPTH_THRESHOLD_PER_WORKER
    }

    fn total_recovery_threshold(&self) -> usize {
        self.effective_worker_count()
            .saturating_mul(QUEUE_DEPTH_RECOVERY_THRESHOLD_PER_WORKER)
    }

    fn is_high_pressure(&self) -> bool {
        self.total() >= self.total_threshold() || self.local_max >= self.worker_local_threshold()
    }

    fn has_recovered(&self) -> bool {
        self.total() < self.total_recovery_threshold()
            && self.local_max < QUEUE_DEPTH_RECOVERY_THRESHOLD_PER_WORKER
    }
}

#[derive(Default)]
struct QueueOverloadLogState {
    overload_started_at: Option<Instant>,
    last_sample_at: Option<Instant>,
    last_warning_at: Option<Instant>,
}

impl QueueOverloadLogState {
    fn observe(&mut self, now: Instant, queue_depths: &QueueDepthSnapshot) -> Option<Duration> {
        if self.overload_started_at.is_none() {
            if queue_depths.is_high_pressure() {
                self.start_episode(now);
            }
            return None;
        }

        if self.last_sample_at.is_some_and(|last_sample_at| {
            now.saturating_duration_since(last_sample_at) > QUEUE_SAMPLE_MAX_GAP
        }) {
            self.reset_episode();
            if queue_depths.is_high_pressure() {
                self.start_episode(now);
            }
            return None;
        }

        self.last_sample_at = Some(now);
        if queue_depths.has_recovered() {
            self.reset_episode();
            return None;
        }

        let overload_started_at = self.overload_started_at.expect("active overload episode");
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

    fn start_episode(&mut self, now: Instant) {
        self.overload_started_at = Some(now);
        self.last_sample_at = Some(now);
        self.last_warning_at = None;
    }

    fn reset_episode(&mut self) {
        self.overload_started_at = None;
        self.last_sample_at = None;
        self.last_warning_at = None;
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
    let mut max_local_queue_depth: usize = 0;

    for w in 0..num_workers {
        let worker_label = w.to_string();
        let mean_poll = metrics.worker_mean_poll_time(w);
        let worker_local_queue_depth = metrics.worker_local_queue_depth(w);
        local_queue_depth = local_queue_depth.saturating_add(worker_local_queue_depth);
        max_local_queue_depth = max_local_queue_depth.max(worker_local_queue_depth);

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

    QueueDepthSnapshot {
        worker_count: num_workers,
        global: global_queue_depth,
        local_total: local_queue_depth,
        local_max: max_local_queue_depth,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(
        worker_count: usize,
        global: usize,
        local_total: usize,
        local_max: usize,
    ) -> QueueDepthSnapshot {
        QueueDepthSnapshot {
            worker_count,
            global,
            local_total,
            local_max,
        }
    }

    fn observe_high_pressure_for(
        state: &mut QueueOverloadLogState,
        start: Instant,
        duration: Duration,
        queue_depths: &QueueDepthSnapshot,
    ) -> Option<Duration> {
        let seconds = duration.as_secs();
        let mut result = None;
        for second in 0..=seconds {
            result = state.observe(start + Duration::from_secs(second), queue_depths);
        }
        result
    }

    #[test]
    fn queue_depth_thresholds_scale_with_worker_count() {
        for (workers, entry, recovery) in [(0, 4, 2), (1, 4, 2), (4, 16, 8), (16, 64, 32)] {
            let queue_depths = snapshot(workers, 0, 0, 0);
            assert_eq!(queue_depths.total_threshold(), entry);
            assert_eq!(queue_depths.total_recovery_threshold(), recovery);
        }
    }

    #[test]
    fn queue_pressure_uses_total_depth_and_hottest_worker() {
        assert!(!snapshot(4, 7, 8, 3).is_high_pressure());
        assert!(snapshot(4, 7, 9, 3).is_high_pressure());
        assert!(snapshot(4, 0, 4, 4).is_high_pressure());

        assert!(!snapshot(4, 0, 8, 1).has_recovered());
        assert!(!snapshot(4, 0, 7, 2).has_recovered());
        assert!(snapshot(4, 0, 7, 1).has_recovered());
    }

    #[test]
    fn queue_overload_requires_sustained_pressure_with_hysteresis() {
        let start = Instant::now();
        let overloaded = snapshot(4, 8, 8, 3);
        let hysteresis_band = snapshot(4, 0, 8, 1);
        let mut state = QueueOverloadLogState::default();

        assert_eq!(
            observe_high_pressure_for(
                &mut state,
                start,
                QUEUE_OVERLOAD_DURATION - Duration::from_secs(1),
                &overloaded,
            ),
            None
        );
        assert_eq!(
            state.observe(
                start + QUEUE_OVERLOAD_DURATION - Duration::from_millis(1),
                &hysteresis_band,
            ),
            None
        );
        assert_eq!(
            state.observe(start + QUEUE_OVERLOAD_DURATION, &hysteresis_band),
            Some(QUEUE_OVERLOAD_DURATION)
        );
    }

    #[test]
    fn sampling_gap_restarts_overload_episode() {
        let start = Instant::now();
        let overloaded = snapshot(1, 2, 2, 2);
        let mut state = QueueOverloadLogState::default();

        assert_eq!(state.observe(start, &overloaded), None);
        assert_eq!(
            state.observe(start + QUEUE_SAMPLE_MAX_GAP, &overloaded),
            None
        );
        let restarted_at =
            start + QUEUE_SAMPLE_MAX_GAP + QUEUE_SAMPLE_MAX_GAP + Duration::from_millis(1);
        assert_eq!(state.observe(restarted_at, &overloaded), None);
        for second in 1..QUEUE_OVERLOAD_DURATION.as_secs() {
            assert_eq!(
                state.observe(restarted_at + Duration::from_secs(second), &overloaded),
                None
            );
        }
        assert_eq!(
            state.observe(restarted_at + QUEUE_OVERLOAD_DURATION, &overloaded),
            Some(QUEUE_OVERLOAD_DURATION)
        );
    }

    #[test]
    fn warnings_repeat_per_episode() {
        let start = Instant::now();
        let overloaded = snapshot(1, 2, 2, 2);
        let recovered = snapshot(1, 0, 1, 1);
        let mut state = QueueOverloadLogState::default();

        assert_eq!(
            observe_high_pressure_for(&mut state, start, QUEUE_OVERLOAD_DURATION, &overloaded),
            Some(QUEUE_OVERLOAD_DURATION)
        );
        for second in (QUEUE_OVERLOAD_DURATION.as_secs() + 1)
            ..(QUEUE_OVERLOAD_DURATION + QUEUE_OVERLOAD_LOG_INTERVAL).as_secs()
        {
            assert_eq!(
                state.observe(start + Duration::from_secs(second), &overloaded),
                None
            );
        }
        assert_eq!(
            state.observe(
                start + QUEUE_OVERLOAD_DURATION + QUEUE_OVERLOAD_LOG_INTERVAL,
                &overloaded,
            ),
            Some(QUEUE_OVERLOAD_DURATION + QUEUE_OVERLOAD_LOG_INTERVAL)
        );

        let recovered_at =
            start + QUEUE_OVERLOAD_DURATION + QUEUE_OVERLOAD_LOG_INTERVAL + Duration::from_secs(1);
        assert_eq!(state.observe(recovered_at, &recovered), None);
        let second_episode_at = recovered_at + Duration::from_secs(1);
        assert_eq!(
            observe_high_pressure_for(
                &mut state,
                second_episode_at,
                QUEUE_OVERLOAD_DURATION,
                &overloaded,
            ),
            Some(QUEUE_OVERLOAD_DURATION)
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sampler_reports_runtime_worker_and_queue_depths() {
        let queue_depths = sample_tokio_metrics(&mut PrevWorkerCounters::new());

        assert_eq!(queue_depths.worker_count, 2);
        assert_eq!(queue_depths.total_threshold(), 8);
        assert!(queue_depths.local_max <= queue_depths.local_total);
        assert_eq!(
            queue_depths.total(),
            queue_depths.global.saturating_add(queue_depths.local_total)
        );
    }

    #[test]
    fn warning_counter_increments_and_is_registered() {
        let before = TOKIO_QUEUE_OVERLOAD_WARNINGS_TOTAL.get();
        warn_queue_overload(&snapshot(1, 2, 2, 2), QUEUE_OVERLOAD_DURATION);
        assert_eq!(TOKIO_QUEUE_OVERLOAD_WARNINGS_TOTAL.get(), before + 1.0);

        let metrics_registry = MetricsRegistry::new();
        ensure_tokio_perf_metrics_registered(&metrics_registry);
        assert!(
            metrics_registry
                .get_prometheus_registry()
                .gather()
                .iter()
                .any(|family| family.name() == "dynamo_tokio_queue_overload_warnings_total")
        );

        let prometheus_registry = prometheus::Registry::new();
        ensure_tokio_perf_metrics_registered_prometheus(&prometheus_registry).unwrap();
        assert!(
            prometheus_registry
                .gather()
                .iter()
                .any(|family| family.name() == "dynamo_tokio_queue_overload_warnings_total")
        );
    }
}
