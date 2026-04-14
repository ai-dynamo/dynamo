// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-specific scheduling implementations.

mod kv_event_sink;
#[path = "sglang/mod.rs"]
pub mod sglang;
pub mod vllm;

pub use crate::common::protocols::ForwardPassSnapshot;
use crate::common::protocols::{DirectRequest, FpmPublisher, KvEventPublishers, OutputSignal};
use dynamo_kv_router::protocols::RouterEvent;
pub(crate) use kv_event_sink::{
    CapturedRouterEventBuffer, DeferredFpmBuffer, capture_deferred_kv_publish_sink,
    capture_router_event_sink, publish_deferred_fpm, publish_deferred_kv_events,
};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

/// Welford's online algorithm for count / sum / population-variance.
///
/// Mirrors the Python `WelfordAccumulator` in `forward_pass_metrics.py`.
#[derive(Default)]
pub(crate) struct WelfordAcc {
    pub(crate) count: u32,
    pub(crate) sum: f64,
    mean: f64,
    m2: f64,
}

impl WelfordAcc {
    pub(crate) fn add(&mut self, v: f64) {
        self.count += 1;
        self.sum += v;
        let delta = v - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = v - self.mean;
        self.m2 += delta * delta2;
    }

    pub(crate) fn variance(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.m2 / self.count as f64
    }
}

/// Build a [`ForwardPassSnapshot`] from engine-agnostic iterators.
///
/// Each engine (vLLM, SGLang) calls this with its own iterators, avoiding
/// duplicated variance/accumulation logic.
///
/// - `scheduled_prefills`: `(prompt_len, prefix_tokens, tokens_computed)` per request
/// - `scheduled_decodes`: `sequence_len` per request
/// - `queued_prefills`: `prompt_len` per waiting prefill request
/// - `queued_decodes`: `kv_tokens` per preempted decode request
pub(crate) fn build_fpm_snapshot(
    scheduled_prefills: impl Iterator<Item = (u64, u64, u64)>,
    scheduled_decodes: impl Iterator<Item = u64>,
    queued_prefills: impl Iterator<Item = u64>,
    queued_decodes: impl Iterator<Item = u64>,
    wall_time_secs: f64,
) -> ForwardPassSnapshot {
    let mut prefill_acc = WelfordAcc::default();
    let mut decode_acc = WelfordAcc::default();
    let mut sum_prefill_tokens: u64 = 0;
    let mut sum_prefill_kv_tokens: u64 = 0;

    for (prompt_len, prefix_tokens, tokens_computed) in scheduled_prefills {
        sum_prefill_tokens += tokens_computed;
        sum_prefill_kv_tokens += prefix_tokens;
        prefill_acc.add(prompt_len as f64);
    }

    for sequence_len in scheduled_decodes {
        decode_acc.add(sequence_len as f64);
    }

    let mut queued_prefill_acc = WelfordAcc::default();
    let mut queued_decode_acc = WelfordAcc::default();

    for prompt_len in queued_prefills {
        queued_prefill_acc.add(prompt_len as f64);
    }

    for kv_tokens in queued_decodes {
        queued_decode_acc.add(kv_tokens as f64);
    }

    ForwardPassSnapshot {
        num_prefill_requests: prefill_acc.count,
        sum_prefill_tokens,
        var_prefill_length: prefill_acc.variance(),
        sum_prefill_kv_tokens,
        num_decode_requests: decode_acc.count,
        sum_decode_kv_tokens: decode_acc.sum as u64,
        var_decode_kv_tokens: decode_acc.variance(),
        num_queued_prefill: queued_prefill_acc.count,
        sum_queued_prefill_tokens: queued_prefill_acc.sum as u64,
        var_queued_prefill_length: queued_prefill_acc.variance(),
        num_queued_decode: queued_decode_acc.count,
        sum_queued_decode_kv_tokens: queued_decode_acc.sum as u64,
        var_queued_decode_kv_tokens: queued_decode_acc.variance(),
        wall_time_secs,
    }
}

pub(crate) use sglang::SglangCore;
pub use sglang::SglangScheduler;
pub(crate) use vllm::VllmCore;
pub use vllm::{MockerMetrics, Scheduler};

#[derive(Debug, Clone)]
pub(crate) struct AdmissionEvent {
    pub(crate) uuid: Uuid,
    pub(crate) reused_input_tokens: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct EnginePassResult {
    pub(crate) end_ms: f64,
    pub(crate) completed_requests: usize,
    pub(crate) output_signals: Vec<OutputSignal>,
    pub(crate) admissions: Vec<AdmissionEvent>,
    pub(crate) active_decode_blocks: u64,
    /// Controls when replay/live schedulers should expose this pass's buffered
    /// KV events to the real router or publisher sink.
    pub(crate) router_event_visibility: RouterEventVisibility,
    /// Router-visible KV events emitted during this pass.
    pub(crate) kv_events: Vec<RouterEvent>,
    /// Forward pass metrics snapshot for this iteration.
    pub(crate) fpm: Option<ForwardPassSnapshot>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RouterEventVisibility {
    /// Expose buffered KV events when the pass starts, before the modeled sleep.
    PassStart,
    /// Expose buffered KV events when the pass finishes, before output flush.
    PassEnd,
}

#[allow(clippy::large_enum_variant)]
pub(crate) enum EngineCore {
    Vllm(VllmCore),
    Sglang(SglangCore),
}

impl EngineCore {
    pub(crate) fn receive(&mut self, request: DirectRequest) -> Uuid {
        match self {
            Self::Vllm(core) => core.receive(request),
            Self::Sglang(core) => core.receive(request),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        match self {
            Self::Vllm(core) => core.is_empty(),
            Self::Sglang(core) => core.is_empty(),
        }
    }

    pub(crate) fn num_requests(&self) -> usize {
        match self {
            Self::Vllm(core) => core.num_requests(),
            Self::Sglang(core) => core.num_requests(),
        }
    }

    pub(crate) fn execute_pass(
        &mut self,
        collector: &mut crate::replay::TraceCollector,
        now_ms: f64,
    ) -> EnginePassResult {
        match self {
            Self::Vllm(core) => core.execute_pass(collector, now_ms),
            Self::Sglang(core) => core.execute_pass(collector, now_ms),
        }
    }

    pub(crate) fn execute_hidden_pass(&mut self, now_ms: f64) -> EnginePassResult {
        match self {
            Self::Vllm(core) => core.execute_hidden_pass(now_ms),
            Self::Sglang(core) => core.execute_hidden_pass(now_ms),
        }
    }
}

#[derive(Clone)]
pub(crate) enum EngineScheduler {
    Vllm(Scheduler),
    Sglang(SglangScheduler),
}

impl EngineScheduler {
    pub(crate) fn new_with_admission(
        args: crate::common::protocols::MockEngineArgs,
        dp_rank: u32,
        output_tx: Option<mpsc::UnboundedSender<Vec<OutputSignal>>>,
        kv_event_publishers: KvEventPublishers,
        cancellation_token: Option<CancellationToken>,
        admission_tx: Option<mpsc::UnboundedSender<AdmissionEvent>>,
        fpm_publisher: FpmPublisher,
    ) -> Self {
        match args.engine_type {
            crate::common::protocols::EngineType::Vllm => {
                Self::Vllm(Scheduler::new_with_admission(
                    args,
                    dp_rank,
                    output_tx,
                    kv_event_publishers,
                    cancellation_token,
                    admission_tx,
                    fpm_publisher,
                ))
            }
            crate::common::protocols::EngineType::Sglang => {
                Self::Sglang(SglangScheduler::new_with_admission(
                    args,
                    dp_rank,
                    output_tx,
                    kv_event_publishers,
                    cancellation_token,
                    admission_tx,
                    fpm_publisher,
                ))
            }
        }
    }
}

impl SchedulerHandle for EngineScheduler {
    fn receive(&self, request: DirectRequest) {
        match self {
            Self::Vllm(scheduler) => scheduler.receive(request),
            Self::Sglang(scheduler) => scheduler.receive(request),
        }
    }

    fn request_sender(&self) -> mpsc::UnboundedSender<DirectRequest> {
        match self {
            Self::Vllm(scheduler) => scheduler.request_sender(),
            Self::Sglang(scheduler) => scheduler.request_sender(),
        }
    }

    fn metrics_receiver(&self) -> tokio::sync::watch::Receiver<MockerMetrics> {
        match self {
            Self::Vllm(scheduler) => scheduler.metrics_receiver(),
            Self::Sglang(scheduler) => scheduler.metrics_receiver(),
        }
    }
}

/// Engine-agnostic scheduler interface.
///
/// Both vLLM and SGLang schedulers implement this trait so that the engine
/// wrapper (`MockEngine`) can work with either backend through the same API.
pub trait SchedulerHandle: Send + Sync {
    /// Send a request to the scheduler's waiting queue.
    fn receive(&self, request: DirectRequest);

    /// Get a clone of the request sender channel for direct sending.
    fn request_sender(&self) -> mpsc::UnboundedSender<DirectRequest>;

    /// Get a watch receiver for scheduler metrics (active decode blocks, etc.).
    fn metrics_receiver(&self) -> tokio::sync::watch::Receiver<MockerMetrics>;
}

/// Shared test utilities for scheduler stress tests.
#[cfg(test)]
pub(crate) mod test_utils;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn welford_acc_empty() {
        let acc = WelfordAcc::default();
        assert_eq!(acc.count, 0);
        assert_eq!(acc.sum, 0.0);
        assert_eq!(acc.variance(), 0.0);
    }

    #[test]
    fn welford_acc_single_value() {
        let mut acc = WelfordAcc::default();
        acc.add(42.0);
        assert_eq!(acc.count, 1);
        assert_eq!(acc.sum, 42.0);
        assert_eq!(acc.variance(), 0.0);
    }

    #[test]
    fn welford_acc_population_variance() {
        let mut acc = WelfordAcc::default();
        // Values: 2, 4, 4, 4, 5, 5, 7, 9
        // Mean = 5, Population variance = 4.0
        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            acc.add(v);
        }
        assert_eq!(acc.count, 8);
        assert_eq!(acc.sum, 40.0);
        assert!((acc.variance() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn welford_acc_matches_python() {
        // Reproduce the Python WelfordAccumulator behavior:
        // values = [100, 200, 300], mean = 200,
        // population variance = ((100-200)^2 + (200-200)^2 + (300-200)^2) / 3
        //                     = (10000 + 0 + 10000) / 3 = 6666.666...
        let mut acc = WelfordAcc::default();
        acc.add(100.0);
        acc.add(200.0);
        acc.add(300.0);
        assert_eq!(acc.count, 3);
        assert_eq!(acc.sum, 600.0);
        let expected = 20000.0 / 3.0;
        assert!(
            (acc.variance() - expected).abs() < 1e-10,
            "expected {expected}, got {}",
            acc.variance()
        );
    }
}
