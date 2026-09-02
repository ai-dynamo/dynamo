// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use dynamo_backend_common::MetricsCtx;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct SpecdecMetricsSnapshot {
    pub starts_accepted: u64,
    pub starts_rejected: u64,
    pub proposals: u64,
    pub completions: u64,
    pub cleanup_acknowledgements: u64,
    pub cleanup_timeouts: u64,
    pub active_sessions: u64,
    pub orphaned_sessions: u64,
    pub orphaned_sessions_reaped: u64,
    pub queue_depth: u64,
    pub queue_rejections: u64,
}

#[derive(Default)]
pub(crate) struct SpecdecMetrics {
    starts_accepted: AtomicU64,
    starts_rejected: AtomicU64,
    proposals: AtomicU64,
    completions: AtomicU64,
    cleanup_acknowledgements: AtomicU64,
    cleanup_timeouts: AtomicU64,
    active_sessions: AtomicU64,
    orphaned_sessions: AtomicU64,
    orphaned_sessions_reaped: AtomicU64,
    queue_depth: AtomicU64,
    queue_rejections: AtomicU64,
}

impl SpecdecMetrics {
    pub(crate) fn register(self: &Arc<Self>, ctx: MetricsCtx<'_>) {
        let metrics = self.clone();
        let labels = ctx.metrics.auto_labels().as_ref().clone();
        ctx.metrics
            .add_expfmt_callback(Arc::new(move || Ok(metrics.render(&labels))));
    }

    pub(crate) fn snapshot(&self) -> SpecdecMetricsSnapshot {
        SpecdecMetricsSnapshot {
            starts_accepted: self.starts_accepted.load(Ordering::Relaxed),
            starts_rejected: self.starts_rejected.load(Ordering::Relaxed),
            proposals: self.proposals.load(Ordering::Relaxed),
            completions: self.completions.load(Ordering::Relaxed),
            cleanup_acknowledgements: self.cleanup_acknowledgements.load(Ordering::Relaxed),
            cleanup_timeouts: self.cleanup_timeouts.load(Ordering::Relaxed),
            active_sessions: self.active_sessions.load(Ordering::Relaxed),
            orphaned_sessions: self.orphaned_sessions.load(Ordering::Relaxed),
            orphaned_sessions_reaped: self.orphaned_sessions_reaped.load(Ordering::Relaxed),
            queue_depth: self.queue_depth.load(Ordering::Relaxed),
            queue_rejections: self.queue_rejections.load(Ordering::Relaxed),
        }
    }

    pub(crate) fn start_accepted(&self) {
        self.starts_accepted.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn start_rejected(&self) {
        self.starts_rejected.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn proposal(&self) {
        self.proposals.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn completion(&self) {
        self.completions.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn cleanup_acknowledged(&self) {
        self.cleanup_acknowledgements
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn cleanup_timed_out(&self) {
        self.cleanup_timeouts.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn set_active_sessions(&self, active: usize) {
        self.active_sessions.store(active as u64, Ordering::Relaxed);
    }

    pub(crate) fn orphan_reap_started(&self) {
        self.orphaned_sessions.fetch_add(1, Ordering::Relaxed);
        self.orphaned_sessions_reaped
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn orphan_reap_finished(&self) {
        self.orphaned_sessions.fetch_sub(1, Ordering::Relaxed);
    }

    pub(crate) fn enter_queue(self: &Arc<Self>) -> QueueDepthGuard {
        self.queue_depth.fetch_add(1, Ordering::Relaxed);
        QueueDepthGuard {
            metrics: self.clone(),
            queued: true,
        }
    }

    pub(crate) fn queue_rejected(&self) {
        self.queue_rejections.fetch_add(1, Ordering::Relaxed);
    }

    fn render(&self, labels: &HashMap<String, String>) -> String {
        let snapshot = self.snapshot();
        let mut output = String::new();
        for (name, help, kind, value) in [
            (
                "starts_accepted_total",
                "Accepted mock speculative decoding START messages.",
                "counter",
                snapshot.starts_accepted,
            ),
            (
                "starts_rejected_total",
                "Rejected mock speculative decoding START messages.",
                "counter",
                snapshot.starts_rejected,
            ),
            (
                "proposals_total",
                "Mock speculative decoding proposal frames emitted.",
                "counter",
                snapshot.proposals,
            ),
            (
                "completions_total",
                "Mock speculative decoding proposal streams completed.",
                "counter",
                snapshot.completions,
            ),
            (
                "cleanup_acknowledgements_total",
                "Mock speculative decoding cleanup acknowledgements emitted or received.",
                "counter",
                snapshot.cleanup_acknowledgements,
            ),
            (
                "cleanup_timeouts_total",
                "Mock speculative decoding cleanup acknowledgement timeouts.",
                "counter",
                snapshot.cleanup_timeouts,
            ),
            (
                "active_sessions",
                "Current active mock speculative decoding sessions.",
                "gauge",
                snapshot.active_sessions,
            ),
            (
                "orphaned_sessions",
                "Current mock speculative decoding sessions being reaped as orphans.",
                "gauge",
                snapshot.orphaned_sessions,
            ),
            (
                "orphaned_sessions_reaped_total",
                "Mock speculative decoding orphaned sessions reaped.",
                "counter",
                snapshot.orphaned_sessions_reaped,
            ),
            (
                "queue_depth",
                "Current mock speculative decoding jobs waiting to begin prefill.",
                "gauge",
                snapshot.queue_depth,
            ),
            (
                "queue_rejections_total",
                "Mock speculative decoding jobs rejected by bounded admission.",
                "counter",
                snapshot.queue_rejections,
            ),
        ] {
            write_metric(&mut output, name, help, kind, value, labels);
        }
        output
    }
}

pub(crate) struct QueueDepthGuard {
    metrics: Arc<SpecdecMetrics>,
    queued: bool,
}

impl QueueDepthGuard {
    pub(crate) fn started(&mut self) {
        if self.queued {
            self.queued = false;
            self.metrics.queue_depth.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

impl Drop for QueueDepthGuard {
    fn drop(&mut self) {
        self.started();
    }
}

fn write_metric(
    output: &mut String,
    suffix: &str,
    help: &str,
    kind: &str,
    value: u64,
    labels: &HashMap<String, String>,
) {
    let name = format!("dynamo_mock_specdec_{suffix}");
    let _ = writeln!(output, "# HELP {name} {help}");
    let _ = writeln!(output, "# TYPE {name} {kind}");
    let _ = write!(output, "{name}");
    if !labels.is_empty() {
        let mut labels = labels.iter().collect::<Vec<_>>();
        labels.sort_unstable_by(|left, right| left.0.cmp(right.0));
        output.push('{');
        for (index, (key, value)) in labels.into_iter().enumerate() {
            if index != 0 {
                output.push(',');
            }
            let _ = write!(output, "{key}=\"{}\"", escape_label(value));
        }
        output.push('}');
    }
    let _ = writeln!(output, " {value}");
}

fn escape_label(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\n', "\\n")
        .replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposition_contains_every_required_surface_and_escapes_labels() {
        let metrics = SpecdecMetrics::default();
        metrics.start_accepted();
        metrics.start_rejected();
        metrics.proposal();
        metrics.completion();
        metrics.cleanup_acknowledged();
        metrics.cleanup_timed_out();
        metrics.set_active_sessions(2);
        metrics.orphan_reap_started();
        metrics.queue_rejected();
        let labels = HashMap::from([("worker_id".to_string(), "7\n\"x".to_string())]);

        let text = metrics.render(&labels);

        for name in [
            "starts_accepted_total",
            "starts_rejected_total",
            "proposals_total",
            "completions_total",
            "cleanup_acknowledgements_total",
            "cleanup_timeouts_total",
            "active_sessions",
            "orphaned_sessions",
            "orphaned_sessions_reaped_total",
            "queue_depth",
            "queue_rejections_total",
        ] {
            assert!(text.contains(&format!("dynamo_mock_specdec_{name}")));
        }
        assert!(text.contains("worker_id=\"7\\n\\\"x\""));
    }

    #[test]
    fn queue_depth_guard_balances_on_start_and_drop() {
        let metrics = Arc::new(SpecdecMetrics::default());
        let mut started = metrics.enter_queue();
        let dropped = metrics.enter_queue();
        assert_eq!(metrics.snapshot().queue_depth, 2);
        started.started();
        assert_eq!(metrics.snapshot().queue_depth, 1);
        drop(dropped);
        assert_eq!(metrics.snapshot().queue_depth, 0);
    }
}
