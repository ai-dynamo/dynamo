// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::common::protocols::{KvCacheEventSink, MockEngineArgs, OutputSignal};
use crate::common::running_mean::RunningMean;
use crate::kv_manager::KvManager;
use crate::replay::TraceCollector;
use crate::scheduler::vllm::{SchedulerState, simulate_decode_step, simulate_prefill_step};
use dynamo_kv_router::protocols::{RouterEvent, WorkerId};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

pub(crate) struct ReplayWorkerCore {
    args: MockEngineArgs,
    pub(crate) scheduler: SchedulerState,
    pub(crate) kv_manager: KvManager,
    hit_rates: RunningMean<f32>,
    kv_event_buffer: Option<KvEventBuffer>,
}

#[derive(Clone, Default)]
struct KvEventBuffer {
    events: Arc<Mutex<Vec<RouterEvent>>>,
}

impl KvEventBuffer {
    fn push(&self, event: RouterEvent) {
        self.events.lock().unwrap().push(event);
    }

    fn drain(&self) -> Vec<RouterEvent> {
        std::mem::take(&mut *self.events.lock().unwrap())
    }
}

#[derive(Clone)]
struct ReplayKvCaptureSink {
    worker_id: WorkerId,
    buffer: KvEventBuffer,
}

impl KvCacheEventSink for ReplayKvCaptureSink {
    fn publish(
        &self,
        event: dynamo_kv_router::protocols::KvCacheEvent,
        _block_token_ids: Option<&[Vec<u32>]>,
    ) -> anyhow::Result<()> {
        self.buffer.push(RouterEvent::new(self.worker_id, event));
        Ok(())
    }
}

impl ReplayWorkerCore {
    pub(crate) fn new(args: MockEngineArgs) -> Self {
        Self::new_with_kv_capture(args, None)
    }

    pub(crate) fn new_with_kv_capture(args: MockEngineArgs, worker_id: Option<WorkerId>) -> Self {
        let kv_event_buffer = worker_id.map(|worker_id| {
            let buffer = KvEventBuffer::default();
            let sink: Arc<dyn KvCacheEventSink> = Arc::new(ReplayKvCaptureSink {
                worker_id,
                buffer: buffer.clone(),
            });

            (buffer, sink)
        });
        let (kv_event_buffer, kv_event_sink) = match kv_event_buffer {
            Some((buffer, sink)) => (Some(buffer), Some(sink)),
            None => (None, None),
        };

        Self {
            kv_manager: KvManager::new_with_event_sink(
                args.num_gpu_blocks,
                args.block_size,
                kv_event_sink,
                0,
            ),
            args,
            scheduler: SchedulerState::default(),
            hit_rates: RunningMean::new(1000),
            kv_event_buffer,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.scheduler.is_empty()
    }

    pub(crate) fn receive(
        &mut self,
        request: crate::common::protocols::DirectRequest,
    ) -> uuid::Uuid {
        self.scheduler.receive(request)
    }

    pub(crate) fn num_requests(&self) -> usize {
        self.scheduler.requests.len()
    }

    pub(crate) fn run_prefill_step(
        &mut self,
        collector: &mut TraceCollector,
        now_ms: f64,
    ) -> std::time::Duration {
        simulate_prefill_step(
            &mut self.scheduler,
            &mut self.kv_manager,
            &mut self.hit_rates,
            &self.args,
            Some(collector),
            None,
            now_ms,
            true,
        )
    }

    pub(crate) fn run_decode_step(
        &mut self,
        collector: &mut TraceCollector,
        now_ms: f64,
    ) -> std::time::Duration {
        let output_tx: Option<mpsc::UnboundedSender<OutputSignal>> = None;
        simulate_decode_step(
            &mut self.scheduler,
            &mut self.kv_manager,
            &output_tx,
            &self.args,
            Some(collector),
            now_ms,
            true,
        )
    }

    pub(crate) fn execute_pass(
        &mut self,
        collector: &mut TraceCollector,
        now_ms: f64,
    ) -> ExecutedPass {
        let requests_before = self.scheduler.requests.len();
        let prefill_time = self.run_prefill_step(collector, now_ms);
        let decode_start_ms = now_ms + prefill_time.as_secs_f64() * 1000.0;
        let (output_tx, mut output_rx) = mpsc::unbounded_channel();
        let output_tx = Some(output_tx);
        let decode_time = simulate_decode_step(
            &mut self.scheduler,
            &mut self.kv_manager,
            &output_tx,
            &self.args,
            Some(collector),
            decode_start_ms,
            true,
        );
        let end_ms = decode_start_ms + decode_time.as_secs_f64() * 1000.0;
        let requests_after = self.scheduler.requests.len();
        let mut output_signals = Vec::new();
        while let Ok(signal) = output_rx.try_recv() {
            output_signals.push(signal);
        }

        ExecutedPass {
            end_ms,
            completed_requests: requests_before.saturating_sub(requests_after),
            output_signals,
            kv_events: self
                .kv_event_buffer
                .as_ref()
                .map(KvEventBuffer::drain)
                .unwrap_or_default(),
        }
    }
}

#[derive(Debug)]
pub(crate) struct ExecutedPass {
    pub(crate) end_ms: f64,
    pub(crate) completed_requests: usize,
    pub(crate) output_signals: Vec<OutputSignal>,
    pub(crate) kv_events: Vec<RouterEvent>,
}
