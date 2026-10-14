// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exclusive wall-clock attribution for the frontend's existing HTTP TTFT interval.
//!
//! Record only stage boundaries. At the first HTTP token, a single event sweep
//! partitions the interval; nested and concurrent spans never double-count time.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

/// Local pipeline context entry, shared with the Python processor and routed engine.
pub const TTFT_CONTEXT_KEY: &str = "frontend_ttft_timings";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum TtftStage {
    FrontendOverhead,
    PythonPreprocess,
    PreprocessQueue,
    TemplateTokenize,
    MultimodalPrepare,
    MmRoutingBuild,
    RoutePrefill,
    RouteDecode,
    RouteAggregated,
    BackendPrefill,
    BackendDecode,
    BackendAggregated,
    FirstTokenPrefill,
    FirstTokenDecode,
    FirstTokenAggregated,
    ResponsePostprocess,
    BackendOverlap,
}

impl TtftStage {
    pub const ALL: [Self; 17] = [
        Self::FrontendOverhead,
        Self::PythonPreprocess,
        Self::PreprocessQueue,
        Self::TemplateTokenize,
        Self::MultimodalPrepare,
        Self::MmRoutingBuild,
        Self::RoutePrefill,
        Self::RouteDecode,
        Self::RouteAggregated,
        Self::BackendPrefill,
        Self::BackendDecode,
        Self::BackendAggregated,
        Self::FirstTokenPrefill,
        Self::FirstTokenDecode,
        Self::FirstTokenAggregated,
        Self::ResponsePostprocess,
        Self::BackendOverlap,
    ];

    pub const fn labels(self) -> [&'static str; 2] {
        match self {
            Self::FrontendOverhead => ["frontend_overhead", ""],
            Self::PythonPreprocess => ["python_preprocess", ""],
            Self::PreprocessQueue => ["preprocess_queue", ""],
            Self::TemplateTokenize => ["template_tokenize", ""],
            Self::MultimodalPrepare => ["multimodal_prepare", ""],
            Self::MmRoutingBuild => ["mm_routing_build", ""],
            Self::RoutePrefill => ["kv_route", "prefill"],
            Self::RouteDecode => ["kv_route", "decode"],
            Self::RouteAggregated => ["kv_route", "aggregated"],
            Self::BackendPrefill => ["backend_first_response", "prefill"],
            Self::BackendDecode => ["backend_first_response", "decode"],
            Self::BackendAggregated => ["backend_first_response", "aggregated"],
            Self::FirstTokenPrefill => ["backend_first_token", "prefill"],
            Self::FirstTokenDecode => ["backend_first_token", "decode"],
            Self::FirstTokenAggregated => ["backend_first_token", "aggregated"],
            Self::ResponsePostprocess => ["first_token_postprocess", ""],
            Self::BackendOverlap => ["backend_overlap", "prefill_decode"],
        }
    }

    /// Restrict Python instrumentation to the fixed frontend stage vocabulary.
    pub fn from_python(stage: &str) -> Option<Self> {
        match stage {
            "python_preprocess" => Some(Self::PythonPreprocess),
            "preprocess_queue" => Some(Self::PreprocessQueue),
            "template_tokenize" => Some(Self::TemplateTokenize),
            "multimodal_prepare" => Some(Self::MultimodalPrepare),
            "mm_routing_build" => Some(Self::MmRoutingBuild),
            _ => None,
        }
    }
}

#[derive(Debug)]
struct Interval {
    stage: TtftStage,
    start: Duration,
    end: Option<Duration>,
}

/// Timing context for one HTTP request. No clocks or labels are updated per token.
#[derive(Debug)]
pub struct TtftTimings {
    start: Instant,
    intervals: Mutex<Vec<Interval>>,
    finished: AtomicBool,
    result: OnceLock<[Duration; 17]>,
}

impl TtftTimings {
    pub fn new(start: Instant) -> Self {
        Self {
            start,
            intervals: Mutex::new(Vec::with_capacity(16)),
            finished: AtomicBool::new(false),
            result: OnceLock::new(),
        }
    }

    pub fn start_span(self: &Arc<Self>, stage: TtftStage) -> TtftSpan {
        if self.finished.load(Ordering::Acquire) {
            return TtftSpan::inactive();
        }
        let start = self.start.elapsed();
        let mut intervals = self.intervals.lock();
        if self.finished.load(Ordering::Acquire) {
            return TtftSpan::inactive();
        }
        let index = intervals.len();
        intervals.push(Interval {
            stage,
            start,
            end: None,
        });
        TtftSpan {
            timings: Some(Arc::clone(self)),
            index,
        }
    }

    /// Import a completed Python interval using ages relative to the call time.
    ///
    /// This avoids comparing Python and Rust monotonic clock epochs. Binding call
    /// overhead shifts both boundaries equally; durations retain their precision.
    pub fn record_ago(&self, stage: TtftStage, start_ago: Duration, end_ago: Duration) {
        if start_ago < end_ago || self.finished.load(Ordering::Acquire) {
            return;
        }
        let now = self.start.elapsed();
        let mut intervals = self.intervals.lock();
        if self.finished.load(Ordering::Acquire) {
            return;
        }
        intervals.push(Interval {
            stage,
            start: now.saturating_sub(start_ago),
            end: Some(now.saturating_sub(end_ago)),
        });
    }

    /// Freeze at the HTTP first-token timestamp, including spans still in flight.
    /// Repeated calls return the original result, regardless of their timestamp.
    pub fn finish(&self, end: Instant) -> [Duration; 17] {
        if let Some(result) = self.result.get() {
            return *result;
        }
        let intervals = self.intervals.lock();
        if let Some(result) = self.result.get() {
            return *result;
        }
        self.finished.store(true, Ordering::Release);
        let result = partition(&intervals, end.saturating_duration_since(self.start));
        // The interval lock serializes finish calls and all interval mutations.
        let _ = self.result.set(result);
        result
    }
}

/// Close a stage on scope exit, or leave it active until HTTP observes a token.
#[derive(Debug)]
pub struct TtftSpan {
    timings: Option<Arc<TtftTimings>>,
    index: usize,
}

impl TtftSpan {
    fn inactive() -> Self {
        Self {
            timings: None,
            index: 0,
        }
    }

    pub fn leave_open(mut self) {
        self.timings.take();
    }
}

impl Drop for TtftSpan {
    fn drop(&mut self) {
        let Some(timings) = &self.timings else {
            return;
        };
        if timings.finished.load(Ordering::Acquire) {
            return;
        }
        let end = timings.start.elapsed();
        let mut intervals = timings.intervals.lock();
        if !timings.finished.load(Ordering::Acquire) {
            intervals[self.index].end = Some(end);
        }
    }
}

fn partition(intervals: &[Interval], total: Duration) -> [Duration; 17] {
    let mut events = Vec::with_capacity(intervals.len() * 2);
    for interval in intervals {
        let start = interval.start.min(total);
        let end = interval.end.unwrap_or(total).min(total);
        if start < end {
            events.push((start, interval.stage, true));
            events.push((end, interval.stage, false));
        }
    }
    events.sort_unstable_by_key(|event| event.0);

    let mut durations = [Duration::ZERO; 17];
    let mut active = [0usize; 17];
    let mut previous = Duration::ZERO;
    for (time, stage, enters) in events {
        durations[attribute(&active) as usize] += time - previous;
        if enters {
            active[stage as usize] += 1;
        } else {
            active[stage as usize] -= 1;
        }
        previous = time;
    }
    durations[attribute(&active) as usize] += total - previous;
    durations
}

fn attribute(active: &[usize; 17]) -> TtftStage {
    use TtftStage::*;
    // Explicit precedence keeps nested work exclusive. Decode wins rare routing
    // ties; concurrent backend prefill/decode is attributed separately below.
    for stage in [
        ResponsePostprocess,
        RouteDecode,
        RoutePrefill,
        RouteAggregated,
        TemplateTokenize,
        MmRoutingBuild,
        MultimodalPrepare,
        PreprocessQueue,
        PythonPreprocess,
    ] {
        if active[stage as usize] > 0 {
            return stage;
        }
    }
    let prefill = active[BackendPrefill as usize] > 0 || active[FirstTokenPrefill as usize] > 0;
    let decode = active[BackendDecode as usize] > 0 || active[FirstTokenDecode as usize] > 0;
    if (prefill && decode) || active[BackendOverlap as usize] > 0 {
        return BackendOverlap;
    }
    for stage in [
        FirstTokenDecode,
        BackendDecode,
        FirstTokenPrefill,
        BackendPrefill,
        FirstTokenAggregated,
        BackendAggregated,
    ] {
        if active[stage as usize] > 0 {
            return stage;
        }
    }
    FrontendOverhead
}

#[cfg(test)]
mod tests {
    use super::*;
    use TtftStage::*;

    fn ms(value: u64) -> Duration {
        Duration::from_millis(value)
    }

    fn interval(stage: TtftStage, start: u64, end: u64) -> Interval {
        Interval {
            stage,
            start: ms(start),
            end: Some(ms(end)),
        }
    }

    fn assert_total(durations: &[Duration; 17], expected: u64) {
        assert_eq!(durations.iter().sum::<Duration>(), ms(expected));
    }

    #[test]
    fn nested_preprocessing_is_exclusive_and_preserves_gaps() {
        let result = partition(
            &[
                interval(PythonPreprocess, 2, 18),
                interval(PreprocessQueue, 3, 7),
                interval(TemplateTokenize, 9, 15),
            ],
            ms(20),
        );
        assert_eq!(result[FrontendOverhead as usize], ms(4));
        assert_eq!(result[PythonPreprocess as usize], ms(6));
        assert_eq!(result[PreprocessQueue as usize], ms(4));
        assert_eq!(result[TemplateTokenize as usize], ms(6));
        assert_eq!(result[MultimodalPrepare as usize], Duration::ZERO);
        assert_total(&result, 20);
    }

    #[test]
    fn routing_metadata_is_exclusive_within_multimodal_preparation() {
        let result = partition(
            &[
                interval(PythonPreprocess, 0, 20),
                interval(MultimodalPrepare, 2, 18),
                interval(MmRoutingBuild, 6, 12),
            ],
            ms(20),
        );
        assert_eq!(result[PythonPreprocess as usize], ms(4));
        assert_eq!(result[MultimodalPrepare as usize], ms(10));
        assert_eq!(result[MmRoutingBuild as usize], ms(6));
        assert_total(&result, 20);
    }

    #[test]
    fn overlapping_backend_phases_count_once_and_yield_to_routing() {
        let result = partition(
            &[
                interval(BackendPrefill, 2, 10),
                interval(FirstTokenPrefill, 10, 20),
                interval(BackendDecode, 6, 14),
                interval(FirstTokenDecode, 14, 22),
                interval(RouteDecode, 8, 12),
                interval(ResponsePostprocess, 21, 24),
            ],
            ms(24),
        );
        assert_eq!(result[FrontendOverhead as usize], ms(2));
        assert_eq!(result[BackendPrefill as usize], ms(4));
        assert_eq!(result[BackendDecode as usize], Duration::ZERO);
        assert_eq!(result[FirstTokenPrefill as usize], Duration::ZERO);
        assert_eq!(result[FirstTokenDecode as usize], ms(1));
        assert_eq!(result[BackendOverlap as usize], ms(10));
        assert_eq!(result[RouteDecode as usize], ms(4));
        assert_eq!(result[ResponsePostprocess as usize], ms(3));
        assert_total(&result, 24);
    }

    #[test]
    fn repeated_and_overlapping_attempts_of_one_stage_count_their_union() {
        let result = partition(
            &[
                interval(RouteAggregated, 1, 4),
                interval(RouteAggregated, 3, 6),
                interval(RouteAggregated, 7, 9),
                interval(BackendAggregated, 6, 7),
            ],
            ms(10),
        );
        assert_eq!(result[RouteAggregated as usize], ms(7));
        assert_eq!(result[BackendAggregated as usize], ms(1));
        assert_eq!(result[FrontendOverhead as usize], ms(2));
        assert_total(&result, 10);
    }

    #[test]
    fn intervals_are_clipped_to_http_ttft_and_unused_stages_are_zero() {
        let result = partition(
            &[
                interval(MultimodalPrepare, 0, 4),
                interval(MmRoutingBuild, 4, 30),
                interval(RoutePrefill, 20, 30),
            ],
            ms(10),
        );
        assert_eq!(result[MultimodalPrepare as usize], ms(4));
        assert_eq!(result[MmRoutingBuild as usize], ms(6));
        for stage in TtftStage::ALL {
            if stage != MultimodalPrepare && stage != MmRoutingBuild {
                assert_eq!(result[stage as usize], Duration::ZERO);
            }
        }
        assert_total(&result, 10);
        assert_eq!(partition(&[], Duration::ZERO), [Duration::ZERO; 17]);
        assert_eq!(partition(&[], ms(10))[FrontendOverhead as usize], ms(10));
    }

    #[test]
    fn finish_freezes_open_spans_and_ignores_late_writes() {
        let start = Instant::now();
        let timings = Arc::new(TtftTimings::new(start));
        timings.intervals.lock().push(Interval {
            stage: BackendDecode,
            start: ms(2),
            end: None,
        });
        let span = TtftSpan {
            timings: Some(Arc::clone(&timings)),
            index: 0,
        };
        let result = timings.finish(start + ms(10));
        drop(span);
        drop(timings.start_span(RouteDecode));
        timings.record_ago(MultimodalPrepare, ms(3), ms(1));
        assert_eq!(timings.intervals.lock().len(), 1);
        assert!(timings.intervals.lock()[0].end.is_none());
        assert_eq!(timings.finish(start + ms(100)), result);
        assert_eq!(result[BackendDecode as usize], ms(8));
        assert_total(&result, 10);
    }

    #[test]
    fn leaving_a_span_open_does_not_retain_the_context() {
        let timings = Arc::new(TtftTimings::new(Instant::now()));
        let span = timings.start_span(ResponsePostprocess);
        assert_eq!(Arc::strong_count(&timings), 2);
        span.leave_open();
        assert_eq!(Arc::strong_count(&timings), 1);
        assert!(timings.intervals.lock()[0].end.is_none());
    }

    #[test]
    fn python_stage_names_are_fixed_and_invalid_ages_are_rejected() {
        for stage in [
            PythonPreprocess,
            PreprocessQueue,
            TemplateTokenize,
            MultimodalPrepare,
            MmRoutingBuild,
        ] {
            assert_eq!(TtftStage::from_python(stage.labels()[0]), Some(stage));
        }
        assert_eq!(TtftStage::from_python("request_id"), None);
        assert_eq!(TtftStage::from_python("backend_first_response"), None);
        let timings = TtftTimings::new(Instant::now());
        timings.record_ago(PythonPreprocess, ms(1), ms(2));
        assert!(timings.intervals.lock().is_empty());
    }

    #[test]
    fn imported_ages_preserve_duration_and_clip_before_request_start() {
        let timings = TtftTimings::new(Instant::now() - ms(100));
        timings.record_ago(TemplateTokenize, ms(7), ms(2));
        timings.record_ago(MultimodalPrepare, Duration::MAX, Duration::MAX);
        let intervals = timings.intervals.lock();
        assert_eq!(intervals[0].end.unwrap() - intervals[0].start, ms(5));
        assert_eq!(intervals[1].start, Duration::ZERO);
        assert_eq!(intervals[1].end, Some(Duration::ZERO));
    }
}
