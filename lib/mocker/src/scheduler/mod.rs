// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-specific scheduling implementations.

mod kv_event_sink;
#[path = "sglang/mod.rs"]
pub mod sglang;
pub mod vllm;

use crate::common::protocols::{DirectRequest, KvEventPublishers, OutputSignal};
use dynamo_kv_router::protocols::RouterEvent;
pub(crate) use kv_event_sink::{
    CapturedRouterEventBuffer, capture_deferred_kv_publish_sink, capture_router_event_sink,
    publish_deferred_kv_events,
};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

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
