// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-specific scheduling implementations.

#[path = "sglang/mod.rs"]
pub mod sglang;
pub mod vllm;

use std::sync::{Arc, Mutex};

use crate::common::protocols::{DirectRequest, KvCacheEventSink, OutputSignal};
use anyhow::Result;
use dynamo_kv_router::protocols::{RouterEvent, WorkerId};
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
    pub(crate) kv_events: Vec<RouterEvent>,
}

#[derive(Clone, Default)]
pub(crate) struct KvEventBuffer {
    events: Arc<Mutex<Vec<RouterEvent>>>,
}

impl KvEventBuffer {
    pub(crate) fn push(&self, event: RouterEvent) {
        self.events.lock().unwrap().push(event);
    }

    pub(crate) fn drain(&self) -> Vec<RouterEvent> {
        std::mem::take(&mut *self.events.lock().unwrap())
    }
}

#[derive(Clone)]
struct BufferedKvEventSink {
    worker_id: WorkerId,
    buffer: KvEventBuffer,
}

impl KvCacheEventSink for BufferedKvEventSink {
    fn publish(
        &self,
        event: dynamo_kv_router::protocols::KvCacheEvent,
        _block_token_ids: Option<&[Vec<u32>]>,
    ) -> Result<()> {
        self.buffer.push(RouterEvent::new(self.worker_id, event));
        Ok(())
    }
}

pub(crate) fn capture_kv_event_sink(
    worker_id: WorkerId,
) -> (KvEventBuffer, Arc<dyn KvCacheEventSink>) {
    let buffer = KvEventBuffer::default();
    let sink: Arc<dyn KvCacheEventSink> = Arc::new(BufferedKvEventSink {
        worker_id,
        buffer: buffer.clone(),
    });
    (buffer, sink)
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
        output_tx: Option<mpsc::UnboundedSender<OutputSignal>>,
        kv_event_sink: Option<Arc<dyn KvCacheEventSink>>,
        cancellation_token: Option<CancellationToken>,
        admission_tx: Option<mpsc::UnboundedSender<AdmissionEvent>>,
    ) -> Self {
        match args.engine_type {
            crate::common::protocols::EngineType::Vllm => {
                Self::Vllm(Scheduler::new_with_admission(
                    args,
                    dp_rank,
                    output_tx,
                    kv_event_sink,
                    cancellation_token,
                    admission_tx,
                ))
            }
            crate::common::protocols::EngineType::Sglang => {
                Self::Sglang(SglangScheduler::new_with_admission(
                    args,
                    dp_rank,
                    output_tx,
                    kv_event_sink,
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
