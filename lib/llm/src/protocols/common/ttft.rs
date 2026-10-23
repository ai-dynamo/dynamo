// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exclusive wall-clock attribution for the frontend's existing HTTP TTFT interval.
//!
//! Record only stage boundaries. At the first HTTP token, a single event sweep
//! partitions the interval; nested and concurrent spans never double-count time.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use dynamo_runtime::metrics::frontend_perf::{STAGE_DISPATCH, STAGE_PREPROCESS, STAGE_ROUTE};
use parking_lot::Mutex;

/// Local pipeline context entry, shared with the Python processor and routed engine.
pub const TTFT_CONTEXT_KEY: &str = "frontend_ttft_timings";

/// Hard bounds apply before allocation and are independent of retry count.
pub const MAX_TTFT_INTERVALS: usize = 256;
pub const MAX_TTFT_BATCH_INTERVALS: usize = 64;

/// Metadata shared by the legacy lifecycle adapter and exclusive attribution.
/// A diagnostic child has no legacy gauge of its own; its parent names the
/// operation whose residual it subtracts from in the TTFT view only.
#[derive(Debug, Clone, Copy)]
pub struct TtftStageDescriptor {
    pub labels: [&'static str; 2],
    pub parent: Option<TtftStage>,
    pub priority: u8,
    pub python_enabled: bool,
    pub legacy_stage: Option<&'static str>,
}

macro_rules! stages {
    ($( $variant:ident => ($label:expr, $phase:literal, $parent:expr, $priority:literal, $python:literal, $legacy:expr) ),+ $(,)?) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        #[repr(usize)]
        pub enum TtftStage { $( $variant, )+ }

        impl TtftStage {
            pub const ALL: [Self; <[()]>::len(&[$(stages!(@unit $variant)),+])] = [$(Self::$variant),+];

            pub const fn descriptor(self) -> TtftStageDescriptor {
                match self {
                    $(Self::$variant => TtftStageDescriptor {
                        labels: [$label, $phase], parent: $parent,
                        priority: $priority, python_enabled: $python, legacy_stage: $legacy,
                    }),+
                }
            }
        }
    };
    (@unit $variant:ident) => { () };
}

// This is the sole stage/phase registry. Numeric IDs are the fixed order below;
// Python receives this table from the binding, never a separate whitelist.
// Keep existing IDs stable and append new entries. Greater priority wins.
stages! {
    FrontendOverhead => ("frontend_overhead", "", None, 0, false, None),
    Preprocess => (STAGE_PREPROCESS, "", None, 30, true, Some(STAGE_PREPROCESS)),
    PreprocessQueue => ("preprocess_queue", "", Some(TtftStage::Preprocess), 31, true, None),
    TemplateTokenize => ("template_tokenize", "", Some(TtftStage::Preprocess), 34, true, None),
    MultimodalPrepare => ("multimodal_prepare", "", Some(TtftStage::Preprocess), 32, true, None),
    MmRoutingBuild => ("mm_routing_build", "", Some(TtftStage::MultimodalPrepare), 33, true, None),
    RoutePrefill => (STAGE_ROUTE, "prefill", None, 41, false, Some(STAGE_ROUTE)),
    RouteDecode => (STAGE_ROUTE, "decode", None, 42, false, Some(STAGE_ROUTE)),
    RouteAggregated => (STAGE_ROUTE, "aggregated", None, 40, false, Some(STAGE_ROUTE)),
    BackendPrefill => (STAGE_DISPATCH, "prefill", None, 13, false, Some(STAGE_DISPATCH)),
    BackendDecode => (STAGE_DISPATCH, "decode", None, 15, false, Some(STAGE_DISPATCH)),
    BackendAggregated => (STAGE_DISPATCH, "aggregated", None, 11, false, Some(STAGE_DISPATCH)),
    FirstTokenPrefill => ("first_token_wait", "prefill", None, 14, false, None),
    FirstTokenDecode => ("first_token_wait", "decode", None, 16, false, None),
    FirstTokenAggregated => ("first_token_wait", "aggregated", None, 12, false, None),
    ResponsePostprocess => ("first_token_postprocess", "", None, 60, false, None),
    BackendOverlap => ("backend_overlap", "prefill_decode", None, 20, false, None),
    TemplateRender => ("template_render", "", Some(TtftStage::Preprocess), 36, true, None),
    Tokenize => ("tokenize", "", Some(TtftStage::Preprocess), 35, true, None),
    StructuralTagBuild => ("structural_tag_build", "", Some(TtftStage::Preprocess), 37, true, None),
    DispatchPreparePrefill => ("dispatch_prepare", "prefill", Some(TtftStage::BackendPrefill), 51, false, None),
    DispatchPrepareDecode => ("dispatch_prepare", "decode", Some(TtftStage::BackendDecode), 52, false, None),
    DispatchPrepareAggregated => ("dispatch_prepare", "aggregated", Some(TtftStage::BackendAggregated), 50, false, None),
    Unclassified => ("unclassified", "", None, 0, false, None),
}

impl TtftStage {
    pub const fn labels(self) -> [&'static str; 2] {
        self.descriptor().labels
    }

    pub const fn python_enabled(self) -> bool {
        self.descriptor().python_enabled
    }

    pub fn from_python(stage: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|item| item.python_enabled() && item.labels()[0] == stage)
    }

    pub fn from_python_id(id: usize) -> Option<Self> {
        Self::ALL
            .get(id)
            .copied()
            .filter(|stage| stage.python_enabled())
    }

    /// Preserve the phase captured at dispatch creation, including retries.
    pub const fn dispatch_prepare(self) -> Option<Self> {
        match self {
            Self::RoutePrefill | Self::BackendPrefill | Self::FirstTokenPrefill => {
                Some(Self::DispatchPreparePrefill)
            }
            Self::RouteDecode | Self::BackendDecode | Self::FirstTokenDecode => {
                Some(Self::DispatchPrepareDecode)
            }
            Self::RouteAggregated | Self::BackendAggregated | Self::FirstTokenAggregated => {
                Some(Self::DispatchPrepareAggregated)
            }
            _ => None,
        }
    }
}

macro_rules! reasons {
    ($name:ident { $($variant:ident => $label:literal),+ $(,)? }) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum $name { $($variant),+ }
        impl $name {
            pub const ALL: [Self; <[()]>::len(&[$(stages!(@unit $variant)),+])] = [$(Self::$variant),+];
            pub const fn as_str(self) -> &'static str {
                match self { $(Self::$variant => $label),+ }
            }
        }
    };
}

reasons! { TtftDegradedReason {
    IntervalLimit => "interval_limit",
    InvalidInterval => "invalid_interval",
    InvalidBatch => "invalid_batch",
    InvalidStage => "invalid_stage",
    InvalidVersion => "invalid_version",
    ClockMismatch => "clock_mismatch",
    TimestampOverflow => "timestamp_overflow",
    MissingTimeline => "missing_timeline",
} }
reasons! { TtftCoarseReason {
    OpaqueTemplate => "opaque_template",
    UnsupportedClock => "unsupported_clock",
    MissingCoverage => "missing_coverage",
} }
reasons! { TtftQualityStatus {
    Complete => "complete",
    Coarse => "coarse",
    Degraded => "degraded",
} }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TtftQuality {
    pub status: TtftQualityStatus,
    pub degraded_reason: Option<TtftDegradedReason>,
    pub coarse_reason: Option<TtftCoarseReason>,
    pub unexpected_overlap: bool,
}

impl Default for TtftQuality {
    fn default() -> Self {
        Self {
            status: TtftQualityStatus::Complete,
            degraded_reason: None,
            coarse_reason: None,
            unexpected_overlap: false,
        }
    }
}

#[derive(Debug)]
struct Interval {
    stage: TtftStage,
    start: Duration,
    end: Option<Duration>,
}

#[derive(Debug)]
struct TimelineState {
    intervals: Vec<Interval>,
    quality: TtftQuality,
}

impl TimelineState {
    fn degrade(&mut self, reason: TtftDegradedReason) {
        self.quality.status = TtftQualityStatus::Degraded;
        self.quality.degraded_reason.get_or_insert(reason);
    }

    fn push(&mut self, interval: Interval) -> Option<usize> {
        if self.quality.degraded_reason.is_some() {
            return None;
        }
        if self.intervals.len() >= MAX_TTFT_INTERVALS {
            self.degrade(TtftDegradedReason::IntervalLimit);
            return None;
        }
        if interval.stage == TtftStage::TemplateTokenize {
            self.coarsen(TtftCoarseReason::OpaqueTemplate);
        }
        let index = self.intervals.len();
        self.intervals.push(interval);
        Some(index)
    }

    fn coarsen(&mut self, reason: TtftCoarseReason) {
        self.quality.coarse_reason.get_or_insert(reason);
        if self.quality.status == TtftQualityStatus::Complete {
            self.quality.status = TtftQualityStatus::Coarse;
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct FrozenTimeline {
    durations: [Duration; TtftStage::ALL.len()],
    quality: TtftQuality,
}

/// Timing context for one HTTP request. No clocks or labels are updated per token.
#[derive(Debug)]
pub struct TtftTimings {
    start: Instant,
    state: Mutex<TimelineState>,
    finished: AtomicBool,
    result: OnceLock<FrozenTimeline>,
}

impl TtftTimings {
    pub fn new(start: Instant) -> Self {
        Self {
            start,
            state: Mutex::new(TimelineState {
                intervals: Vec::with_capacity(16),
                quality: TtftQuality::default(),
            }),
            finished: AtomicBool::new(false),
            result: OnceLock::new(),
        }
    }

    fn elapsed(&self) -> Duration {
        #[cfg(test)]
        CLOCK_READS.with(|reads| reads.set(reads.get() + 1));
        self.start.elapsed()
    }

    pub fn is_frozen(&self) -> bool {
        self.finished.load(Ordering::Acquire)
    }

    pub fn quality(&self) -> TtftQuality {
        if let Some(result) = self.result.get() {
            return result.quality;
        }
        self.state.lock().quality
    }

    pub fn mark_degraded(&self, reason: TtftDegradedReason) {
        if self.is_frozen() {
            return;
        }
        let mut state = self.state.lock();
        if !self.is_frozen() {
            state.degrade(reason);
        }
    }

    pub fn mark_coarse(&self, reason: TtftCoarseReason) {
        if self.is_frozen() {
            return;
        }
        let mut state = self.state.lock();
        if !self.is_frozen() {
            state.coarsen(reason);
        }
    }

    pub fn start_span(self: &Arc<Self>, stage: TtftStage) -> TtftSpan {
        if self.is_frozen() {
            return TtftSpan::inactive();
        }
        let start = self.elapsed();
        let mut state = self.state.lock();
        if self.is_frozen() {
            return TtftSpan::inactive();
        }
        let Some(index) = state.push(Interval {
            stage,
            start,
            end: None,
        }) else {
            return TtftSpan::inactive();
        };
        TtftSpan {
            timings: Some(Arc::clone(self)),
            index,
        }
    }

    /// Reuse an existing operation's start and elapsed result. The inclusive
    /// histogram keeps its own population; only this view subtracts nested work.
    pub fn record_duration(&self, stage: TtftStage, start: Instant, elapsed: Duration) {
        if self.is_frozen() {
            return;
        }
        let Some(end) = start.checked_add(elapsed) else {
            self.mark_degraded(TtftDegradedReason::TimestampOverflow);
            return;
        };
        self.record_interval(stage, start, end);
    }

    pub fn record_interval(&self, stage: TtftStage, start: Instant, end: Instant) {
        if self.is_frozen() {
            return;
        }
        let mut state = self.state.lock();
        if self.is_frozen() {
            return;
        }
        if start > end {
            state.degrade(TtftDegradedReason::InvalidInterval);
            return;
        }
        state.push(Interval {
            stage,
            start: start.saturating_duration_since(self.start),
            end: Some(end.saturating_duration_since(self.start)),
        });
    }

    /// Import a bounded batch with one Rust clock read. The caller verifies the
    /// source clock domain and submission window before converting integer ages.
    /// This never compares Python's monotonic epoch with Rust's Instant epoch.
    pub fn record_batch_ago(&self, batch: &[(TtftStage, Duration, Duration)]) -> bool {
        if self.is_frozen() {
            return false;
        }
        if batch.len() > MAX_TTFT_BATCH_INTERVALS {
            self.mark_degraded(TtftDegradedReason::InvalidBatch);
            return false;
        }
        if batch.is_empty() {
            return true;
        }
        let now = self.elapsed();
        let mut state = self.state.lock();
        if self.is_frozen() || state.quality.degraded_reason.is_some() {
            return false;
        }
        if batch.iter().any(|(stage, _, _)| !stage.python_enabled()) {
            state.degrade(TtftDegradedReason::InvalidStage);
            return false;
        }
        // Validate the whole batch before accepting any of it. Clipping an
        // untrusted, pre-request timestamp would hide a bad clock domain.
        if batch
            .iter()
            .any(|(_, start, end)| start < end || *start > now)
        {
            state.degrade(TtftDegradedReason::InvalidInterval);
            return false;
        }
        if state.intervals.len() + batch.len() > MAX_TTFT_INTERVALS {
            state.degrade(TtftDegradedReason::IntervalLimit);
            return false;
        }
        for &(stage, start_ago, end_ago) in batch {
            state.push(Interval {
                stage,
                start: now - start_ago,
                end: Some(now - end_ago),
            });
        }
        true
    }

    pub fn record_ago(&self, stage: TtftStage, start_ago: Duration, end_ago: Duration) {
        self.record_batch_ago(&[(stage, start_ago, end_ago)]);
    }

    /// Freeze at the HTTP first-token timestamp, including spans still in flight.
    /// Repeated calls return the original result, regardless of their timestamp.
    pub fn finish(&self, end: Instant) -> [Duration; TtftStage::ALL.len()] {
        if let Some(result) = self.result.get() {
            return result.durations;
        }
        let mut state = self.state.lock();
        if let Some(result) = self.result.get() {
            return result.durations;
        }
        self.finished.store(true, Ordering::Release);
        let total = end.saturating_duration_since(self.start);
        let durations = if state.quality.degraded_reason.is_some() {
            let mut durations = [Duration::ZERO; TtftStage::ALL.len()];
            durations[TtftStage::Unclassified as usize] = total;
            durations
        } else {
            let (durations, overlap) = partition_with_quality(&state.intervals, total);
            state.quality.unexpected_overlap = overlap;
            durations
        };
        let _ = self.result.set(FrozenTimeline {
            durations,
            quality: state.quality,
        });
        durations
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
        if timings.is_frozen() {
            return;
        }
        let end = timings.elapsed();
        let mut state = timings.state.lock();
        if !timings.is_frozen() {
            // Slots are never reused, even after a limit or validation failure.
            state.intervals[self.index].end = Some(end);
        }
    }
}

#[cfg(test)]
fn partition(intervals: &[Interval], total: Duration) -> [Duration; TtftStage::ALL.len()] {
    partition_with_quality(intervals, total).0
}

fn partition_with_quality(
    intervals: &[Interval],
    total: Duration,
) -> ([Duration; TtftStage::ALL.len()], bool) {
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
    let mut durations = [Duration::ZERO; TtftStage::ALL.len()];
    let mut active = [0usize; TtftStage::ALL.len()];
    let mut previous = Duration::ZERO;
    let mut unexpected_overlap = false;
    for (time, stage, enters) in events {
        if time > previous {
            durations[attribute(&active) as usize] += time - previous;
            unexpected_overlap |= has_unexpected_overlap(&active);
        }
        if enters {
            active[stage as usize] += 1;
        } else {
            active[stage as usize] -= 1;
        }
        previous = time;
    }
    if total > previous {
        durations[attribute(&active) as usize] += total - previous;
        unexpected_overlap |= has_unexpected_overlap(&active);
    }
    (durations, unexpected_overlap)
}

fn attribute(active: &[usize; TtftStage::ALL.len()]) -> TtftStage {
    use TtftStage::*;
    let prefill = active[BackendPrefill as usize] > 0 || active[FirstTokenPrefill as usize] > 0;
    let decode = active[BackendDecode as usize] > 0 || active[FirstTokenDecode as usize] > 0;
    let overlap = prefill && decode;
    TtftStage::ALL
        .into_iter()
        .filter(|stage| {
            *stage != Unclassified
                && (active[*stage as usize] > 0 || (*stage == BackendOverlap && overlap))
        })
        .max_by_key(|stage| stage.descriptor().priority)
        .unwrap_or(FrontendOverhead)
}

fn is_ancestor(parent: TtftStage, mut child: TtftStage) -> bool {
    while let Some(next) = child.descriptor().parent {
        if parent == next {
            return true;
        }
        child = next;
    }
    false
}

fn has_unexpected_overlap(active: &[usize; TtftStage::ALL.len()]) -> bool {
    // Frontend work should be sequential or nested. Concurrent backend waits
    // have lower precedence and are intentionally permitted (including P/D).
    for (index, stage) in TtftStage::ALL.into_iter().enumerate() {
        if stage.descriptor().priority < TtftStage::Preprocess.descriptor().priority
            || active[index] == 0
        {
            continue;
        }
        for other in TtftStage::ALL.into_iter().skip(index + 1) {
            if other.descriptor().priority >= TtftStage::Preprocess.descriptor().priority
                && active[other as usize] > 0
                && !is_ancestor(stage, other)
                && !is_ancestor(other, stage)
            {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
thread_local! {
    static CLOCK_READS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
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

    fn assert_total(durations: &[Duration; TtftStage::ALL.len()], expected: u64) {
        assert_eq!(durations.iter().sum::<Duration>(), ms(expected));
    }

    #[test]
    fn reused_operation_timing_is_clipped_nested_and_frozen() {
        let start = Instant::now();
        let timings = TtftTimings::new(start);
        timings.record_duration(Preprocess, start, ms(20));
        timings.record_duration(TemplateTokenize, start + ms(2), ms(7));
        timings.record_duration(MultimodalPrepare, start + ms(7), ms(20));
        let result = timings.finish(start + ms(15));
        assert_total(&result, 15);
        assert_eq!(result[TemplateTokenize as usize], ms(7));
        assert_eq!(result[MultimodalPrepare as usize], ms(6));
        timings.record_duration(RouteDecode, start, ms(100));
        assert_eq!(timings.finish(start + ms(100)), result);
    }

    #[test]
    fn nested_preprocessing_is_exclusive_and_preserves_gaps() {
        let result = partition(
            &[
                interval(Preprocess, 2, 18),
                interval(PreprocessQueue, 3, 7),
                interval(TemplateTokenize, 9, 15),
            ],
            ms(20),
        );
        assert_eq!(result[FrontendOverhead as usize], ms(4));
        assert_eq!(result[Preprocess as usize], ms(6));
        assert_eq!(result[PreprocessQueue as usize], ms(4));
        assert_eq!(result[TemplateTokenize as usize], ms(6));
        assert_eq!(result[MultimodalPrepare as usize], Duration::ZERO);
        assert_total(&result, 20);
    }

    #[test]
    fn routing_metadata_is_exclusive_within_multimodal_preparation() {
        let result = partition(
            &[
                interval(Preprocess, 0, 20),
                interval(MultimodalPrepare, 2, 18),
                interval(MmRoutingBuild, 6, 12),
            ],
            ms(20),
        );
        assert_eq!(result[Preprocess as usize], ms(4));
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
        assert_eq!(
            partition(&[], Duration::ZERO),
            [Duration::ZERO; TtftStage::ALL.len()]
        );
        assert_eq!(partition(&[], ms(10))[FrontendOverhead as usize], ms(10));
    }

    #[test]
    fn finish_freezes_open_spans_and_ignores_late_writes() {
        let start = Instant::now();
        let timings = Arc::new(TtftTimings::new(start));
        timings.state.lock().intervals.push(Interval {
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
        assert_eq!(timings.state.lock().intervals.len(), 1);
        assert!(timings.state.lock().intervals[0].end.is_none());
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
        assert!(timings.state.lock().intervals[0].end.is_none());
    }

    #[test]
    fn python_stage_names_are_fixed_and_invalid_ages_are_rejected() {
        for stage in [
            Preprocess,
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
        timings.record_ago(Preprocess, ms(1), ms(2));
        assert!(timings.state.lock().intervals.is_empty());
    }

    #[test]
    fn imported_ages_preserve_duration_and_reject_before_request_start() {
        let timings = TtftTimings::new(Instant::now() - ms(100));
        timings.record_ago(TemplateTokenize, ms(7), ms(2));
        timings.record_ago(MultimodalPrepare, Duration::MAX, Duration::MAX);
        let state = timings.state.lock();
        let intervals = &state.intervals;
        assert_eq!(intervals[0].end.unwrap() - intervals[0].start, ms(5));
        assert_eq!(intervals.len(), 1);
    }

    #[test]
    fn registry_ids_labels_and_legacy_mappings_are_unique() {
        let mut labels = std::collections::HashSet::new();
        for (id, stage) in TtftStage::ALL.into_iter().enumerate() {
            assert_eq!(stage as usize, id);
            assert!(labels.insert(stage.labels()));
            assert_eq!(
                TtftStage::from_python_id(id),
                stage.python_enabled().then_some(stage)
            );
            if let Some(legacy_stage) = stage.descriptor().legacy_stage {
                assert_eq!(legacy_stage, stage.labels()[0]);
            }
            if let Some(parent) = stage.descriptor().parent {
                assert!(stage.descriptor().priority > parent.descriptor().priority);
            }
        }
        assert!(TtftStage::from_python_id(TtftStage::ALL.len()).is_none());
        assert_eq!(
            BackendDecode.dispatch_prepare(),
            Some(DispatchPrepareDecode)
        );
        assert_eq!(TemplateRender.dispatch_prepare(), None);
    }

    #[test]
    fn prepare_is_removed_from_dispatch_without_moving_legacy_boundary() {
        let result = partition(
            &[
                interval(BackendDecode, 2, 20),
                interval(DispatchPrepareDecode, 2, 7),
                interval(RouteDecode, 0, 4),
                interval(ResponsePostprocess, 18, 22),
            ],
            ms(22),
        );
        assert_eq!(result[RouteDecode as usize], ms(2));
        assert_eq!(result[DispatchPrepareDecode as usize], ms(5));
        assert_eq!(result[BackendDecode as usize], ms(11));
        assert_eq!(result[ResponsePostprocess as usize], ms(4));
        assert_total(&result, 22);
    }

    #[test]
    fn real_template_tokenize_and_structural_tag_boundaries_are_separate() {
        let start = Instant::now();
        let timings = TtftTimings::new(start);
        timings.record_duration(Preprocess, start, ms(20));
        timings.record_duration(TemplateRender, start + ms(2), ms(3));
        timings.record_duration(Tokenize, start + ms(5), ms(7));
        // Structural tags are built after the preprocessing operation returns.
        timings.record_duration(StructuralTagBuild, start + ms(20), ms(4));
        let result = timings.finish(start + ms(25));
        assert_eq!(result[Preprocess as usize], ms(10));
        assert_eq!(result[TemplateRender as usize], ms(3));
        assert_eq!(result[Tokenize as usize], ms(7));
        assert_eq!(result[StructuralTagBuild as usize], ms(4));
        assert_eq!(result[FrontendOverhead as usize], ms(1));
        assert_total(&result, 25);
        assert_eq!(timings.quality(), TtftQuality::default());
    }

    #[test]
    fn interval_limit_degrades_without_reusing_live_guard_slots() {
        let start = Instant::now();
        let timings = Arc::new(TtftTimings::new(start));
        let span = timings.start_span(BackendDecode);
        for _ in 1..MAX_TTFT_INTERVALS {
            timings.record_duration(TemplateRender, start, ms(1));
        }
        drop(timings.start_span(RouteDecode));
        assert_eq!(timings.state.lock().intervals.len(), MAX_TTFT_INTERVALS);
        drop(span);
        assert!(timings.state.lock().intervals[0].end.is_some());
        let result = timings.finish(start + ms(10));
        assert_eq!(result[Unclassified as usize], ms(10));
        assert_total(&result, 10);
        assert_eq!(
            timings.quality().degraded_reason,
            Some(TtftDegradedReason::IntervalLimit)
        );
        for stage in TtftStage::ALL {
            if stage != Unclassified {
                assert_eq!(result[stage as usize], Duration::ZERO);
            }
        }
    }

    #[test]
    fn imported_batch_has_one_anchor_and_is_validated_atomically() {
        let timings = TtftTimings::new(Instant::now() - ms(100));
        let before = CLOCK_READS.with(|reads| reads.get());
        assert!(timings.record_batch_ago(&[
            (TemplateRender, ms(20), ms(15)),
            (Tokenize, ms(15), ms(7)),
            (StructuralTagBuild, ms(7), ms(1)),
        ]));
        assert_eq!(CLOCK_READS.with(|reads| reads.get()) - before, 1);
        let state = timings.state.lock();
        assert_eq!(state.intervals[0].end, Some(state.intervals[1].start));
        assert_eq!(state.intervals[1].end, Some(state.intervals[2].start));
        drop(state);
        assert!(
            !timings.record_batch_ago(&[(TemplateRender, ms(6), ms(3)), (Tokenize, ms(1), ms(2)),])
        );
        assert_eq!(timings.state.lock().intervals.len(), 3);
        assert_eq!(
            timings.quality().degraded_reason,
            Some(TtftDegradedReason::InvalidInterval)
        );
    }

    #[test]
    fn batch_limit_and_overflow_are_bounded_degradation() {
        let timings = TtftTimings::new(Instant::now());
        let batch =
            vec![(TemplateRender, Duration::ZERO, Duration::ZERO); MAX_TTFT_BATCH_INTERVALS + 1];
        assert!(!timings.record_batch_ago(&batch));
        assert!(timings.state.lock().intervals.is_empty());
        assert_eq!(
            timings.quality().degraded_reason,
            Some(TtftDegradedReason::InvalidBatch)
        );
        let timings = TtftTimings::new(Instant::now());
        timings.record_duration(Tokenize, Instant::now(), Duration::MAX);
        assert_eq!(
            timings.quality().degraded_reason,
            Some(TtftDegradedReason::TimestampOverflow)
        );
    }

    #[test]
    fn opaque_fallback_is_coarse_but_conserves_time() {
        let start = Instant::now();
        let timings = TtftTimings::new(start);
        timings.record_duration(TemplateTokenize, start, ms(3));
        let result = timings.finish(start + ms(4));
        assert_total(&result, 4);
        assert_eq!(result[TemplateTokenize as usize], ms(3));
        assert_eq!(timings.quality().status, TtftQualityStatus::Coarse);
        assert_eq!(
            timings.quality().coarse_reason,
            Some(TtftCoarseReason::OpaqueTemplate)
        );
    }

    #[test]
    fn sibling_overlap_is_deterministic_and_nested_work_is_expected() {
        let intervals = [
            interval(Preprocess, 0, 10),
            interval(Tokenize, 2, 8),
            interval(TemplateRender, 3, 6),
        ];
        let (result, overlap) = partition_with_quality(&intervals, ms(10));
        assert!(overlap);
        assert_eq!(result[TemplateRender as usize], ms(3));
        assert_eq!(result[Tokenize as usize], ms(3));
        assert_total(&result, 10);
        let reversed: Vec<_> = intervals.into_iter().rev().collect();
        assert_eq!(partition_with_quality(&reversed, ms(10)), (result, overlap));
        let (_, overlap) = partition_with_quality(
            &[
                interval(Preprocess, 0, 10),
                interval(MultimodalPrepare, 2, 8),
                interval(MmRoutingBuild, 3, 6),
            ],
            ms(10),
        );
        assert!(!overlap);
    }

    #[test]
    fn frozen_entries_and_drop_read_no_clocks_or_locks() {
        let start = Instant::now();
        let timings = Arc::new(TtftTimings::new(start));
        let span = timings.start_span(Preprocess);
        let frozen = timings.finish(start + ms(10));
        let before = CLOCK_READS.with(|reads| reads.get());
        // Holding the lock also verifies the ordinary frozen paths avoid it.
        let _state = timings.state.lock();
        drop(span);
        drop(timings.start_span(TemplateRender));
        timings.record_duration(Tokenize, start, Duration::MAX);
        timings.record_interval(Tokenize, start, start);
        timings.record_ago(Tokenize, Duration::ZERO, Duration::ZERO);
        timings.mark_degraded(TtftDegradedReason::InvalidBatch);
        timings.mark_coarse(TtftCoarseReason::MissingCoverage);
        assert_eq!(timings.finish(start + ms(100)), frozen);
        assert_eq!(timings.quality(), TtftQuality::default());
        assert_eq!(CLOCK_READS.with(|reads| reads.get()), before);
    }

    #[test]
    fn concurrent_finish_and_import_freeze_one_conserved_snapshot() {
        let start = Instant::now() - ms(100);
        let timings = Arc::new(TtftTimings::new(start));
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let imported = Arc::clone(&timings);
        let worker_barrier = Arc::clone(&barrier);
        let worker = std::thread::spawn(move || {
            worker_barrier.wait();
            imported.record_batch_ago(&[(TemplateRender, ms(20), ms(10))]);
        });
        barrier.wait();
        let result = timings.finish(start + ms(100));
        worker.join().unwrap();
        assert_total(&result, 100);
        assert_eq!(timings.finish(start + ms(200)), result);
    }

    #[test]
    fn dense_retry_timelines_conserve_integer_time_independent_of_insertion_order() {
        for seed in 0..32u64 {
            let mut intervals = Vec::new();
            for index in 0..MAX_TTFT_INTERVALS {
                let stage = TtftStage::ALL[(index + seed as usize) % TtftStage::ALL.len()];
                if stage == Unclassified {
                    continue;
                }
                let start = (index as u64 * 17 + seed * 13) % 120;
                let end = start + (index as u64 * 7 + seed) % 40;
                intervals.push(interval(stage, start, end));
            }
            let result = partition(&intervals, ms(100));
            assert_total(&result, 100);
            intervals.reverse();
            assert_eq!(partition(&intervals, ms(100)), result);
        }
    }
}
