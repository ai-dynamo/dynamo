// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, anyhow, bail};

use crate::common::protocols::DirectRequest;
use crate::common::protocols::MockEngineArgs;
use crate::replay::TraceCollector;
use crate::scheduler::{EngineCore, EnginePassResult};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AggRequestPhase {
    QueuedAtRouter,
    Running,
}

pub(crate) struct AggRequestState {
    request: Option<DirectRequest>,
    pub(in crate::replay::offline) phase: AggRequestPhase,
    pub(in crate::replay::offline) prefill_completed: bool,
}

impl AggRequestState {
    pub(crate) fn new_queued(request: DirectRequest) -> Self {
        Self {
            request: Some(request),
            phase: AggRequestPhase::QueuedAtRouter,
            prefill_completed: false,
        }
    }

    pub(crate) fn new_running() -> Self {
        Self {
            request: None,
            phase: AggRequestPhase::Running,
            prefill_completed: false,
        }
    }

    pub(crate) fn take_queued_request(&mut self, uuid: Uuid) -> Result<DirectRequest> {
        if self.phase != AggRequestPhase::QueuedAtRouter {
            bail!("offline replay expected queued request state for {uuid}");
        }
        let request = self
            .request
            .take()
            .ok_or_else(|| anyhow!("offline replay missing queued request payload for {uuid}"))?;
        self.phase = AggRequestPhase::Running;
        Ok(request)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DisaggPhase {
    QueuedPrefill,
    RunningPrefill,
    QueuedDecode,
    RunningDecode,
    Done,
}

pub(crate) struct DisaggRequestState {
    original: Option<DirectRequest>,
    #[cfg(test)]
    arrival_ms: f64,
    pub(in crate::replay::offline) phase: DisaggPhase,
    prefill_worker_idx: Option<usize>,
    decode_worker_idx: Option<usize>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DisaggRequestSnapshot {
    pub(crate) arrival_ms: f64,
    pub(crate) phase: DisaggPhase,
    pub(crate) prefill_worker_idx: Option<usize>,
    pub(crate) decode_worker_idx: Option<usize>,
}

impl DisaggRequestState {
    pub(crate) fn new(request: DirectRequest, arrival_ms: f64) -> Self {
        #[cfg(not(test))]
        let _ = arrival_ms;
        Self {
            original: Some(request),
            #[cfg(test)]
            arrival_ms,
            phase: DisaggPhase::QueuedPrefill,
            prefill_worker_idx: None,
            decode_worker_idx: None,
        }
    }

    pub(crate) fn original_request(&self) -> Result<&DirectRequest> {
        self.original
            .as_ref()
            .ok_or_else(|| anyhow!("offline disagg replay request payload was already released"))
    }

    pub(crate) fn build_prefill_request(&self) -> Result<DirectRequest> {
        let mut request = self.original_request()?.clone();
        request.max_output_tokens = 1;
        Ok(request)
    }

    pub(crate) fn start_prefill(&mut self, worker_idx: usize) {
        self.phase = DisaggPhase::RunningPrefill;
        self.prefill_worker_idx = Some(worker_idx);
    }

    pub(crate) fn queue_decode(&mut self) {
        self.phase = DisaggPhase::QueuedDecode;
    }

    pub(crate) fn start_decode(&mut self, worker_idx: usize) {
        self.phase = DisaggPhase::RunningDecode;
        self.decode_worker_idx = Some(worker_idx);
    }

    pub(crate) fn mark_done(&mut self) {
        self.phase = DisaggPhase::Done;
        self.original = None;
    }

    #[cfg(test)]
    pub(crate) fn debug_snapshot(&self) -> DisaggRequestSnapshot {
        DisaggRequestSnapshot {
            arrival_ms: self.arrival_ms,
            phase: self.phase,
            prefill_worker_idx: self.prefill_worker_idx,
            decode_worker_idx: self.decode_worker_idx,
        }
    }
}

pub(crate) struct OfflineWorkerState {
    core: EngineCore,
    busy: bool,
    in_flight: usize,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OfflineWorkerSnapshot {
    pub(crate) busy: bool,
    pub(crate) in_flight: usize,
    pub(crate) ready: bool,
    pub(crate) drained: bool,
}

impl OfflineWorkerState {
    pub(crate) fn new(worker_idx: usize, args: MockEngineArgs, capture_kv_events: bool) -> Self {
        let core = match args.engine_type {
            crate::common::protocols::EngineType::Vllm => {
                if capture_kv_events {
                    EngineCore::Vllm(crate::scheduler::VllmCore::new_with_kv_capture(
                        args,
                        worker_idx as u64,
                    ))
                } else {
                    EngineCore::Vllm(crate::scheduler::VllmCore::new(args))
                }
            }
            crate::common::protocols::EngineType::Sglang => {
                if capture_kv_events {
                    EngineCore::Sglang(crate::scheduler::SglangCore::new_with_kv_capture(
                        args,
                        worker_idx as u64,
                    ))
                } else {
                    EngineCore::Sglang(crate::scheduler::SglangCore::new(args))
                }
            }
        };

        Self {
            core,
            busy: false,
            in_flight: 0,
        }
    }

    pub(crate) fn in_flight(&self) -> usize {
        debug_assert!(self.in_flight >= self.core.num_requests());
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
    ) -> EnginePassResult {
        self.core.execute_pass(collector, now_ms)
    }

    pub(crate) fn execute_hidden_pass(&mut self, now_ms: f64) -> EnginePassResult {
        self.core.execute_hidden_pass(now_ms)
    }

    #[cfg(test)]
    pub(crate) fn debug_snapshot(&self) -> OfflineWorkerSnapshot {
        OfflineWorkerSnapshot {
            busy: self.busy,
            in_flight: self.in_flight,
            ready: self.is_ready(),
            drained: self.is_drained(),
        }
    }
}
