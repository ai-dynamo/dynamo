// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::core::ExecutedPass;
use super::core::ReplayWorkerCore;
use crate::common::protocols::DirectRequest;
use crate::common::protocols::MockEngineArgs;
use crate::replay::TraceCollector;

pub(crate) struct OfflineWorkerState {
    worker_idx: usize,
    core: ReplayWorkerCore,
    busy: bool,
    in_flight: usize,
}

impl OfflineWorkerState {
    pub(crate) fn new(worker_idx: usize, args: MockEngineArgs) -> Self {
        Self {
            worker_idx,
            core: ReplayWorkerCore::new(args),
            busy: false,
            in_flight: 0,
        }
    }

    pub(crate) fn worker_idx(&self) -> usize {
        self.worker_idx
    }

    pub(crate) fn in_flight(&self) -> usize {
        self.in_flight
    }

    pub(crate) fn receive_request(&mut self, request: DirectRequest) {
        self.in_flight += 1;
        self.core.receive(request);
    }

    pub(crate) fn mark_completed(&mut self, completed_requests: usize) {
        self.in_flight = self.in_flight.saturating_sub(completed_requests);
    }

    pub(crate) fn mark_busy(&mut self) {
        self.busy = true;
    }

    pub(crate) fn mark_idle(&mut self) {
        self.busy = false;
    }

    pub(crate) fn is_ready(&self) -> bool {
        !self.busy && !self.core.is_empty()
    }

    pub(crate) fn is_drained(&self) -> bool {
        self.in_flight == 0 && !self.busy && self.core.is_empty()
    }

    pub(crate) fn execute_pass(
        &mut self,
        collector: &mut TraceCollector,
        now_ms: f64,
    ) -> ExecutedPass {
        self.core.execute_pass(collector, now_ms)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WorkerProgressSnapshot {
    pub(crate) waiting_len: usize,
    pub(crate) prefill_len: usize,
    pub(crate) decode_len: usize,
    pub(crate) request_count: usize,
    pub(crate) total_generated_tokens: usize,
    pub(crate) total_allocated_tokens: usize,
    pub(crate) active_blocks: usize,
}
