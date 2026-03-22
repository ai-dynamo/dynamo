// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::core::ExecutedPass;
use super::core::ReplayWorkerCore;
use crate::common::protocols::DirectRequest;
use crate::common::protocols::MockEngineArgs;
use crate::replay::TraceCollector;

pub(crate) struct OfflineWorkerState {
    core: ReplayWorkerCore,
    busy: bool,
    in_flight: usize,
}

impl OfflineWorkerState {
    pub(crate) fn new(worker_idx: usize, args: MockEngineArgs, capture_kv_events: bool) -> Self {
        let core = if capture_kv_events {
            ReplayWorkerCore::new_with_kv_capture(args, Some(worker_idx as u64))
        } else {
            ReplayWorkerCore::new(args)
        };

        Self {
            core,
            busy: false,
            in_flight: 0,
        }
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
