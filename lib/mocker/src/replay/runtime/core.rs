// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::common::protocols::{MockEngineArgs, OutputSignal};
use crate::common::running_mean::RunningMean;
use crate::kv_manager::KvManager;
use crate::replay::TraceCollector;
use crate::scheduler::vllm::{SchedulerState, simulate_decode_step, simulate_prefill_step};
use tokio::sync::mpsc;

pub(crate) struct ReplayWorkerCore {
    args: MockEngineArgs,
    pub(crate) scheduler: SchedulerState,
    pub(crate) kv_manager: KvManager,
    hit_rates: RunningMean<f32>,
}

impl ReplayWorkerCore {
    pub(crate) fn new(args: MockEngineArgs) -> Self {
        Self {
            kv_manager: KvManager::new(args.num_gpu_blocks, args.block_size),
            args,
            scheduler: SchedulerState::default(),
            hit_rates: RunningMean::new(1000),
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
        let decode_time = self.run_decode_step(collector, decode_start_ms);
        let end_ms = decode_start_ms + decode_time.as_secs_f64() * 1000.0;
        let requests_after = self.scheduler.requests.len();

        ExecutedPass {
            end_ms,
            completed_requests: requests_before.saturating_sub(requests_after),
        }
    }
}

#[derive(Debug)]
pub(crate) struct ExecutedPass {
    pub(crate) end_ms: f64,
    pub(crate) completed_requests: usize,
}
