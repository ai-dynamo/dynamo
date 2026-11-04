// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Adapters for observations that share a lifecycle boundary.
//!
//! The existing inflight gauge and inclusive operation histogram retain their
//! own populations and boundaries. Optional TTFT intervals reuse those events;
//! only the TTFT partition subtracts nested work.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use dynamo_runtime::metrics::frontend_perf::StageGuard;
use prometheus::Histogram;

use super::ttft::{TtftSpan, TtftStage, TtftTimings};

/// One shared entry/exit for a legacy gauge and an optional TTFT interval.
/// Diagnostic-only stages have no legacy gauge in the canonical registry.
pub(crate) struct StageObservation {
    _legacy: Option<StageGuard>,
    _ttft: Option<TtftSpan>,
}

impl StageObservation {
    pub(crate) fn new(stage: TtftStage, timings: Option<&Arc<TtftTimings>>) -> Self {
        let descriptor = stage.descriptor();
        Self {
            _legacy: descriptor
                .legacy_stage
                .map(|legacy| StageGuard::new(legacy, descriptor.labels[1])),
            _ttft: timings.map(|timings| timings.start_span(stage)),
        }
    }

    /// Failed/empty attempts end diagnostic timing before cleanup completes.
    /// Keep their legacy gauge alive until its original response/drop boundary.
    pub(crate) fn end_timing(&mut self) {
        self._ttft.take();
    }
}

/// Record a completed legacy operation and reuse its measured interval.
/// Call only at the original success boundary, never from a Drop guard: errors
/// must not gain new legacy histogram samples.
pub(crate) fn observe_operation(
    histogram: &Histogram,
    stage: TtftStage,
    start: Instant,
    elapsed: Duration,
    timings: Option<&Arc<TtftTimings>>,
) {
    histogram.observe(elapsed.as_secs_f64());
    if let Some(timings) = timings {
        timings.record_duration(stage, start, elapsed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::metrics::frontend_perf::STAGE_REQUESTS;
    use prometheus::HistogramOpts;

    #[test]
    fn ttft_operation_reuses_inclusive_samples_without_changing_buckets() {
        let histogram = Histogram::with_opts(
            HistogramOpts::new("operation_adapter_test", "test").buckets(vec![0.01, 0.1, 1.0]),
        )
        .unwrap();
        let start = Instant::now();
        let timeline = Arc::new(TtftTimings::new(start));
        let elapsed = Duration::from_millis(100);
        // The same legacy observation is made with detail off and on.
        observe_operation(&histogram, TtftStage::TemplateRender, start, elapsed, None);
        observe_operation(
            &histogram,
            TtftStage::TemplateRender,
            start,
            elapsed,
            Some(&timeline),
        );
        // Nested work changes only exclusive attribution, not legacy samples.
        timeline.record_duration(
            TtftStage::StructuralTagBuild,
            start + Duration::from_millis(40),
            Duration::from_millis(20),
        );
        let stages = timeline.finish(start + elapsed);
        assert_eq!(histogram.get_sample_count(), 2);
        assert_eq!(histogram.get_sample_sum(), 0.2);
        assert_eq!(
            stages[TtftStage::TemplateRender as usize],
            Duration::from_millis(80)
        );
        assert_eq!(
            stages[TtftStage::StructuralTagBuild as usize],
            Duration::from_millis(20)
        );
        use prometheus::core::Collector;
        let families = histogram.collect();
        let buckets = families[0].get_metric()[0].get_histogram().get_bucket();
        assert_eq!(
            buckets
                .iter()
                .map(|bucket| bucket.upper_bound())
                .collect::<Vec<_>>(),
            vec![0.01, 0.1, 1.0]
        );
        assert_eq!(
            buckets
                .iter()
                .map(|bucket| bucket.cumulative_count())
                .collect::<Vec<_>>(),
            vec![0, 2, 2]
        );
    }

    #[test]
    fn ttft_abort_ends_timing_but_preserves_legacy_gauge_until_drop() {
        // A unique label isolates the global gauge from concurrent router tests.
        let gauge = STAGE_REQUESTS.with_label_values(&["dispatch", "ttft_adapter_abort_test"]);
        let start = Instant::now();
        let timeline = Arc::new(TtftTimings::new(start));
        let mut observation = StageObservation {
            _legacy: Some(StageGuard::new("dispatch", "ttft_adapter_abort_test")),
            _ttft: Some(timeline.start_span(TtftStage::BackendDecode)),
        };
        assert_eq!(gauge.get(), 1);
        observation.end_timing();
        assert_eq!(gauge.get(), 1);
        let end = Instant::now() + Duration::from_secs(1);
        let stages = timeline.finish(end);
        assert!(stages[TtftStage::FrontendOverhead as usize] >= Duration::from_secs(1));
        drop(observation);
        assert_eq!(gauge.get(), 0);
    }
}
