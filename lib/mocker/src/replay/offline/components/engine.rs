// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::bail;

use super::super::events::SimulationWorkerStage;
use super::super::runtime_utils::WorkerCompletionPayload;
#[cfg(test)]
use super::super::state::OfflineWorkerSnapshot;
use super::super::state::OfflineWorkerState;
use super::{EngineEffects, EnginePassMode, ScheduledWorkerCompletion};
use crate::common::protocols::DirectRequest;
use crate::replay::TraceCollector;
use crate::scheduler::RouterEventVisibility;

pub(in crate::replay::offline) struct EngineComponent {
    stage: SimulationWorkerStage,
    pass_mode: EnginePassMode,
    workers: Vec<OfflineWorkerState>,
}

impl EngineComponent {
    pub(in crate::replay::offline) fn new(
        stage: SimulationWorkerStage,
        pass_mode: EnginePassMode,
        workers: Vec<OfflineWorkerState>,
    ) -> Self {
        Self {
            stage,
            pass_mode,
            workers,
        }
    }

    pub(in crate::replay::offline) fn dispatch(
        &mut self,
        worker_idx: usize,
        request: DirectRequest,
    ) -> anyhow::Result<()> {
        self.validate_worker_idx(worker_idx)?;
        self.workers[worker_idx].receive_request(request);
        Ok(())
    }

    pub(in crate::replay::offline) fn drive_ready(
        &mut self,
        now_ms: f64,
        mut collector: Option<&mut TraceCollector>,
    ) -> anyhow::Result<EngineEffects> {
        for worker_idx in 0..self.workers.len() {
            if !self.workers[worker_idx].is_ready() {
                continue;
            }

            let executed = match self.pass_mode {
                EnginePassMode::Visible => {
                    let Some(collector) = collector.as_deref_mut() else {
                        bail!("offline replay visible engine pass requires a collector");
                    };
                    self.workers[worker_idx].execute_pass(collector, now_ms)
                }
                EnginePassMode::Hidden => self.workers[worker_idx].execute_hidden_pass(now_ms),
            };

            let mut effects = EngineEffects {
                admissions: executed.admissions,
                ..EngineEffects::default()
            };
            let completion_kv_events =
                if executed.router_event_visibility == RouterEventVisibility::PassStart {
                    effects.pass_start_kv_events = executed.kv_events;
                    Vec::new()
                } else {
                    executed.kv_events
                };
            let payload = WorkerCompletionPayload {
                stage: self.stage,
                worker_idx,
                completed_requests: executed.completed_requests,
                output_signals: executed.output_signals,
                kv_events: completion_kv_events,
            };

            if executed.end_ms == now_ms {
                effects.immediate_completions.push(payload);
                return Ok(effects);
            }

            self.workers[worker_idx].mark_busy();
            effects
                .scheduled_completions
                .push(ScheduledWorkerCompletion {
                    at_ms: executed.end_ms,
                    payload,
                });
            return Ok(effects);
        }

        Ok(EngineEffects::default())
    }

    pub(in crate::replay::offline) fn on_scheduled_completion(
        &mut self,
        payload: WorkerCompletionPayload,
    ) -> anyhow::Result<WorkerCompletionPayload> {
        if payload.stage != self.stage {
            bail!(
                "offline replay completion stage mismatch: expected {:?}, got {:?}",
                self.stage,
                payload.stage
            );
        }
        self.validate_worker_idx(payload.worker_idx)?;
        self.workers[payload.worker_idx].mark_idle();
        self.workers[payload.worker_idx].mark_completed(payload.completed_requests);
        Ok(payload)
    }

    pub(in crate::replay::offline) fn in_flight(&self) -> usize {
        self.workers.iter().map(OfflineWorkerState::in_flight).sum()
    }

    pub(in crate::replay::offline) fn is_drained(&self) -> bool {
        self.workers.iter().all(OfflineWorkerState::is_drained)
    }

    pub(in crate::replay::offline) fn worker_count(&self) -> usize {
        self.workers.len()
    }

    fn validate_worker_idx(&self, worker_idx: usize) -> anyhow::Result<()> {
        if worker_idx >= self.workers.len() {
            bail!("offline replay selected unknown worker index {worker_idx}");
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn debug_snapshots(&self) -> Vec<OfflineWorkerSnapshot> {
        self.workers
            .iter()
            .map(OfflineWorkerState::debug_snapshot)
            .collect()
    }
}
