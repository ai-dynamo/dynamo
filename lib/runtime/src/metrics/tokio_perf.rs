// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tokio runtime metrics and event-loop canary

use once_cell::sync::{Lazy, OnceCell};
use prometheus::{Counter, Gauge, Histogram, HistogramOpts, IntGaugeVec, Opts};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::runtime::Handle;

use super::prometheus_names::{frontend_perf, name_prefix, tokio_perf as names};
use crate::MetricsRegistry;

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

pub static TOKIO_WORKER_PARK_COUNT_TOTAL: Lazy<IntGaugeVec> = Lazy::new(|| {
    IntGaugeVec::new(
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

pub static TOKIO_WORKER_STEAL_COUNT_TOTAL: Lazy<IntGaugeVec> = Lazy::new(|| {
    IntGaugeVec::new(
        Opts::new(
            tokio_metric_name(names::WORKER_STEAL_COUNT_TOTAL),
            "Total number of tasks stolen by worker",
        ),
        &["worker"],
    )
    .expect("tokio worker_steal_count_total")
});

pub static TOKIO_WORKER_OVERFLOW_COUNT_TOTAL: Lazy<IntGaugeVec> = Lazy::new(|| {
    IntGaugeVec::new(
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
            format!("{}_{}", name_prefix::FRONTEND, frontend_perf::EVENT_LOOP_DELAY_SECONDS),
            "Event loop delay canary: drift from 10ms sleep (seconds)",
        )
        .buckets(vec![0.0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]),
    )
    .expect("event_loop_delay_seconds histogram")
});

pub static EVENT_LOOP_STALL_TOTAL: Lazy<Counter> = Lazy::new(|| {
    Counter::new(
        format!("{}_{}", name_prefix::FRONTEND, frontend_perf::EVENT_LOOP_STALL_TOTAL),
        "Number of event loop stalls (delay > 5ms)",
    )
    .expect("event_loop_stall_total counter")
});

static REGISTERED: OnceCell<()> = OnceCell::new();

/// Register tokio perf and canary metrics with the given registry. Idempotent.
pub fn ensure_tokio_perf_metrics_registered(registry: &MetricsRegistry) {
    let _ = REGISTERED.get_or_init(|| {
        registry.add_metric(Box::new(TOKIO_GLOBAL_QUEUE_DEPTH.clone())).ok();
        registry.add_metric(Box::new(TOKIO_BUDGET_FORCED_YIELD_TOTAL.clone())).ok();
        registry.add_metric(Box::new(TOKIO_BLOCKING_THREADS.clone())).ok();
        registry.add_metric(Box::new(TOKIO_BLOCKING_IDLE_THREADS.clone())).ok();
        registry.add_metric(Box::new(TOKIO_BLOCKING_QUEUE_DEPTH.clone())).ok();
        registry.add_metric(Box::new(TOKIO_ALIVE_TASKS.clone())).ok();
        registry.add_metric(Box::new(TOKIO_WORKER_MEAN_POLL_TIME_NS.clone())).ok();
        registry.add_metric(Box::new(TOKIO_WORKER_BUSY_RATIO_VEC.clone())).ok();
        registry.add_metric(Box::new(TOKIO_WORKER_PARK_COUNT_TOTAL.clone())).ok();
        registry.add_metric(Box::new(TOKIO_WORKER_LOCAL_QUEUE_DEPTH.clone())).ok();
        registry.add_metric(Box::new(TOKIO_WORKER_STEAL_COUNT_TOTAL.clone())).ok();
        registry.add_metric(Box::new(TOKIO_WORKER_OVERFLOW_COUNT_TOTAL.clone())).ok();
        registry.add_metric(Box::new(EVENT_LOOP_DELAY_SECONDS.clone())).ok();
        registry.add_metric(Box::new(EVENT_LOOP_STALL_TOTAL.clone())).ok();
    });
}

/// Register tokio perf and canary metrics with a raw Prometheus registry.
pub fn ensure_tokio_perf_metrics_registered_prometheus(
    registry: &prometheus::Registry,
) -> Result<(), prometheus::Error> {
    if REGISTERED.get().is_some() {
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
    let _ = REGISTERED.set(());
    Ok(())
}

/// Run the tokio metrics collector (1s interval) and event-loop canary.
/// Spawn this on the runtime you want to monitor (e.g. primary handle).
pub async fn tokio_metrics_and_canary_loop() {
    let canary_interval = Duration::from_millis(10);
    let stall_threshold = Duration::from_millis(5);
    let collect_interval = Duration::from_secs(1);
    let mut next_collect = Instant::now() + collect_interval;

    loop {
        let start = Instant::now();
        tokio::time::sleep(canary_interval).await;
        let delay = start.elapsed().saturating_sub(canary_interval);
        EVENT_LOOP_DELAY_SECONDS.observe(delay.as_secs_f64());
        if delay > stall_threshold {
            EVENT_LOOP_STALL_TOTAL.inc();
            tracing::warn!(delay_ms = delay.as_millis(), "event loop congestion");
        }
        if Instant::now() >= next_collect {
            next_collect += collect_interval;
            if let Err(e) = sample_tokio_metrics() {
                tracing::debug!("tokio metrics sample skipped: {}", e);
            }
        }
    }
}

static PREV_BUDGET_FORCED_YIELD: AtomicU64 = AtomicU64::new(0);

fn sample_tokio_metrics() -> Result<(), String> {
    let metrics = Handle::current().metrics();

    TOKIO_GLOBAL_QUEUE_DEPTH.set(metrics.global_queue_depth() as f64);
    let budget = metrics.budget_forced_yield_count();
    let prev = PREV_BUDGET_FORCED_YIELD.swap(budget, Ordering::Relaxed);
    TOKIO_BUDGET_FORCED_YIELD_TOTAL.inc_by((budget.saturating_sub(prev)) as f64);
    TOKIO_BLOCKING_THREADS.set(metrics.num_blocking_threads() as f64);
    TOKIO_BLOCKING_IDLE_THREADS.set(metrics.num_idle_blocking_threads() as f64);
    TOKIO_BLOCKING_QUEUE_DEPTH.set(metrics.blocking_queue_depth() as f64);
    TOKIO_ALIVE_TASKS.set(metrics.num_alive_tasks() as f64);

    let num_workers = metrics.num_workers();
    for w in 0..num_workers {
        let worker_label = w.to_string();
        let mean_poll = metrics.worker_mean_poll_time(w);
        TOKIO_WORKER_MEAN_POLL_TIME_NS
            .with_label_values(&[&worker_label])
            .set(mean_poll.as_nanos() as i64);
        TOKIO_WORKER_PARK_COUNT_TOTAL
            .with_label_values(&[&worker_label])
            .set(metrics.worker_park_count(w) as i64);
        TOKIO_WORKER_LOCAL_QUEUE_DEPTH
            .with_label_values(&[&worker_label])
            .set(metrics.worker_local_queue_depth(w) as i64);
        TOKIO_WORKER_STEAL_COUNT_TOTAL
            .with_label_values(&[&worker_label])
            .set(metrics.worker_steal_count(w) as i64);
        TOKIO_WORKER_OVERFLOW_COUNT_TOTAL
            .with_label_values(&[&worker_label])
            .set(metrics.worker_overflow_count(w) as i64);
        // Busy ratio: total_busy_duration over 1s interval -> ratio. We don't have delta here;
        // use mean_poll_time as proxy: if high, worker is busy. Store as 0-1000 (per mille).
        let busy_proxy = (mean_poll.as_secs_f64() / 0.001).min(1.0); // 1ms = saturated
        TOKIO_WORKER_BUSY_RATIO_VEC
            .with_label_values(&[&worker_label])
            .set((busy_proxy * 1000.0) as i64);
    }

    Ok(())
}
