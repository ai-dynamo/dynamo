// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-pool saturation metrics for the shared TCP server (backend side).
//!
//! These metrics expose queue buildup between `work_tx.send()` and dispatcher
//! pickup, and permit starvation in the bounded worker pool.

use once_cell::sync::{Lazy, OnceCell};
use prometheus::{Histogram, HistogramOpts, IntCounter, IntGauge};

use super::prometheus_names::{name_prefix, work_handler};
use crate::MetricsRegistry;

fn work_handler_metric_name(suffix: &str) -> String {
    format!("{}_{}", name_prefix::WORK_HANDLER, suffix)
}

/// Current items sitting in the bounded mpsc work queue awaiting dispatcher
/// pickup. Incremented on successful `work_tx.send()` and decremented immediately
/// after `work_rx.recv()`. Permit-acquire wait is NOT counted here — see
/// `WORK_HANDLER_PERMIT_WAIT_SECONDS`.
pub static WORK_HANDLER_QUEUE_DEPTH: Lazy<IntGauge> = Lazy::new(|| {
    IntGauge::new(
        work_handler_metric_name(work_handler::QUEUE_DEPTH),
        "Current items in the bounded work queue awaiting dispatcher pickup",
    )
    .expect("work_handler_queue_depth gauge")
});

/// Configured capacity of the bounded work queue. Static; set once at server init.
pub static WORK_HANDLER_QUEUE_CAPACITY: Lazy<IntGauge> = Lazy::new(|| {
    IntGauge::new(
        work_handler_metric_name(work_handler::QUEUE_CAPACITY),
        "Configured capacity of the bounded work queue",
    )
    .expect("work_handler_queue_capacity gauge")
});

/// Total times `work_tx.send().await` returned an error, which for tokio's
/// bounded mpsc only happens when the receiver (dispatcher task) is gone — the
/// channel applies backpressure on "full" rather than returning an error.
pub static WORK_HANDLER_ENQUEUE_REJECTED_TOTAL: Lazy<IntCounter> = Lazy::new(|| {
    IntCounter::new(
        work_handler_metric_name(work_handler::ENQUEUE_REJECTED_TOTAL),
        "Times enqueuing work failed because the dispatcher channel was closed",
    )
    .expect("work_handler_enqueue_rejected_total counter")
});

/// Time spent waiting to acquire a worker-pool permit. Normal operation is
/// sub-millisecond; saturation pushes p99 into seconds.
pub static WORK_HANDLER_PERMIT_WAIT_SECONDS: Lazy<Histogram> = Lazy::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            work_handler_metric_name(work_handler::PERMIT_WAIT_SECONDS),
            "Time spent waiting for a worker-pool permit (seconds)",
        )
        .buckets(vec![
            0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0,
        ]),
    )
    .expect("work_handler_permit_wait_seconds histogram")
});

/// Current number of active worker-pool tasks (permits in use).
pub static WORK_HANDLER_POOL_ACTIVE_TASKS: Lazy<IntGauge> = Lazy::new(|| {
    IntGauge::new(
        work_handler_metric_name(work_handler::POOL_ACTIVE_TASKS),
        "Current number of active worker-pool tasks (permits in use)",
    )
    .expect("work_handler_pool_active_tasks gauge")
});

/// Configured worker-pool capacity (total permits). Static; set once at server init.
pub static WORK_HANDLER_POOL_CAPACITY: Lazy<IntGauge> = Lazy::new(|| {
    IntGauge::new(
        work_handler_metric_name(work_handler::POOL_CAPACITY),
        "Configured worker-pool capacity (total permits)",
    )
    .expect("work_handler_pool_capacity gauge")
});

/// Guards idempotency for the `MetricsRegistry` registration path.
static METRICS_REGISTERED: OnceCell<()> = OnceCell::new();

/// Register worker-pool saturation metrics with the given registry. Idempotent.
pub fn ensure_work_handler_pool_metrics_registered(registry: &MetricsRegistry) {
    let _ = METRICS_REGISTERED.get_or_init(|| {
        registry.add_metric_or_warn(
            Box::new(WORK_HANDLER_QUEUE_DEPTH.clone()),
            "work_handler_queue_depth",
        );
        registry.add_metric_or_warn(
            Box::new(WORK_HANDLER_QUEUE_CAPACITY.clone()),
            "work_handler_queue_capacity",
        );
        registry.add_metric_or_warn(
            Box::new(WORK_HANDLER_ENQUEUE_REJECTED_TOTAL.clone()),
            "work_handler_enqueue_rejected_total",
        );
        registry.add_metric_or_warn(
            Box::new(WORK_HANDLER_PERMIT_WAIT_SECONDS.clone()),
            "work_handler_permit_wait_seconds",
        );
        registry.add_metric_or_warn(
            Box::new(WORK_HANDLER_POOL_ACTIVE_TASKS.clone()),
            "work_handler_pool_active_tasks",
        );
        registry.add_metric_or_warn(
            Box::new(WORK_HANDLER_POOL_CAPACITY.clone()),
            "work_handler_pool_capacity",
        );
    });
}
