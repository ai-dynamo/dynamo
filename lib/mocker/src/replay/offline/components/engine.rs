// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use anyhow::bail;

use super::super::events::SimulationWorkerStage;
use super::super::runtime_utils::WorkerCompletionPayload;
#[cfg(test)]
use super::super::state::OfflineWorkerSnapshot;
use super::super::state::OfflineWorkerState;
use super::{EngineEffects, EnginePassMode, ScheduledWorkerCompletion};
use crate::common::protocols::{DirectRequest, MockEngineArgs};
use crate::replay::TraceCollector;
use crate::scheduler::RouterEventVisibility;

pub(in crate::replay::offline) struct EngineComponent {
    stage: SimulationWorkerStage,
    pass_mode: EnginePassMode,
    /// Workers keyed by stable ID (monotonic, never reused).
    workers: BTreeMap<usize, OfflineWorkerState>,
    /// Counter for generating the next stable worker ID.
    next_id: usize,
    /// Workers marked for removal — skipped by round-robin, removed when drained.
    pending_removal: BTreeSet<usize>,
    /// Workers still starting up — excluded from active set until ready.
    pending_startup: BTreeSet<usize>,
    /// Engine args used to construct new workers during scale-up.
    args: MockEngineArgs,
    /// Whether new workers should capture KV events (true when a router is present).
    capture_kv_events: bool,
}

impl EngineComponent {
    pub(in crate::replay::offline) fn new(
        stage: SimulationWorkerStage,
        pass_mode: EnginePassMode,
        workers: Vec<OfflineWorkerState>,
    ) -> Self {
        let count = workers.len();
        let map: BTreeMap<usize, OfflineWorkerState> = workers.into_iter().enumerate().collect();
        Self {
            stage,
            pass_mode,
            workers: map,
            next_id: count,
            pending_removal: BTreeSet::new(),
            pending_startup: BTreeSet::new(),
            args: MockEngineArgs::default(),
            capture_kv_events: false,
        }
    }

    /// Set the engine args and KV capture flag used when adding workers dynamically.
    pub(in crate::replay::offline) fn set_scaling_args(
        &mut self,
        args: MockEngineArgs,
        capture_kv_events: bool,
    ) {
        self.args = args;
        self.capture_kv_events = capture_kv_events;
    }

    /// Add a new worker, returning its stable ID.
    pub(in crate::replay::offline) fn add_worker(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        let worker = OfflineWorkerState::new(id, self.args.clone(), self.capture_kv_events);
        self.workers.insert(id, worker);
        id
    }

    /// Mark a worker for removal. It will be skipped by `drive_ready` and
    /// removed once fully drained.
    pub(in crate::replay::offline) fn mark_for_removal(&mut self, worker_id: usize) {
        self.pending_removal.insert(worker_id);
    }

    /// Remove all marked workers that have fully drained, returning their IDs.
    pub(in crate::replay::offline) fn try_remove_drained(&mut self) -> Vec<usize> {
        let mut removed = Vec::new();
        self.pending_removal.retain(|&id| {
            if let Some(worker) = self.workers.get(&id) {
                if worker.is_drained() {
                    removed.push(id);
                    return false; // remove from pending set
                }
            } else {
                // Worker already gone
                return false;
            }
            true // keep in pending set
        });
        for &id in &removed {
            self.workers.remove(&id);
        }
        removed
    }

    /// Apply a target worker count: add new workers or mark excess for removal.
    /// Returns `(added_ids, newly_marked_ids)` so the caller can update the
    /// router immediately. Newly marked workers should be removed from the
    /// router right away to prevent new requests from landing on them, even
    /// though the workers themselves remain in the engine until fully drained.
    ///
    /// The effective count is `active + pending_startup` — workers that will
    /// be active once all startups complete. On scale-down, pending startup
    /// workers are cancelled first (cheapest: no in-flight work, no router
    /// registration), then active workers are marked for removal.
    pub(in crate::replay::offline) fn apply_target_count(
        &mut self,
        target: usize,
    ) -> (Vec<usize>, Vec<usize>) {
        let active_ids = self.active_worker_ids();
        let effective = active_ids.len() + self.pending_startup.len();
        let mut added = Vec::new();
        let mut newly_marked = Vec::new();

        if target > effective {
            let has_startup_delay = self.startup_time_ms().is_some();
            for _ in 0..(target - effective) {
                let id = self.add_worker();
                if has_startup_delay {
                    self.pending_startup.insert(id);
                }
                added.push(id);
            }
        } else if target < effective {
            let excess = effective - target;

            // Cancel pending startup workers first (reverse order = highest IDs).
            let to_cancel: Vec<usize> = self
                .pending_startup
                .iter()
                .copied()
                .rev()
                .take(excess)
                .collect();
            for &id in &to_cancel {
                self.pending_startup.remove(&id);
                self.workers.remove(&id);
            }

            // Mark active workers for removal if more excess remains.
            let remaining = excess - to_cancel.len();
            for &id in active_ids.iter().rev().take(remaining) {
                self.mark_for_removal(id);
                newly_marked.push(id);
            }
        }

        // Clean up any workers that have already fully drained.
        self.try_remove_drained();
        (added, newly_marked)
    }

    /// Return stable IDs of all active workers — excludes both pending removal
    /// and pending startup.
    pub(in crate::replay::offline) fn active_worker_ids(&self) -> Vec<usize> {
        self.workers
            .keys()
            .filter(|id| !self.pending_removal.contains(id) && !self.pending_startup.contains(id))
            .copied()
            .collect()
    }

    /// Return the configured startup delay in milliseconds, if any.
    pub(in crate::replay::offline) fn startup_time_ms(&self) -> Option<f64> {
        self.args
            .startup_time
            .filter(|&s| s > 0.0)
            .map(|s| s * 1000.0)
    }

    /// Mark a pending-startup worker as ready. Returns `true` if the worker
    /// was actually pending startup (and is now active), `false` if the worker
    /// was already cancelled or unknown (stale event).
    pub(in crate::replay::offline) fn mark_worker_ready(&mut self, worker_id: usize) -> bool {
        self.pending_startup.remove(&worker_id) && self.workers.contains_key(&worker_id)
    }

    pub(in crate::replay::offline) fn dispatch(
        &mut self,
        worker_id: usize,
        request: DirectRequest,
    ) -> anyhow::Result<()> {
        let worker = self
            .workers
            .get_mut(&worker_id)
            .ok_or_else(|| anyhow::anyhow!("offline replay selected unknown worker {worker_id}"))?;
        worker.receive_request(request);
        Ok(())
    }

    pub(in crate::replay::offline) fn drive_ready(
        &mut self,
        now_ms: f64,
        mut collector: Option<&mut TraceCollector>,
    ) -> anyhow::Result<EngineEffects> {
        // Collect worker IDs first to avoid borrow issues.
        let worker_ids: Vec<usize> = self.workers.keys().copied().collect();
        for worker_id in worker_ids {
            let worker = self.workers.get(&worker_id).unwrap();
            if !worker.is_ready() {
                continue;
            }

            let executed = match self.pass_mode {
                EnginePassMode::Visible => {
                    let Some(collector) = collector.as_deref_mut() else {
                        bail!("offline replay visible engine pass requires a collector");
                    };
                    self.workers
                        .get_mut(&worker_id)
                        .unwrap()
                        .execute_pass(collector, now_ms)
                }
                EnginePassMode::Hidden => self
                    .workers
                    .get_mut(&worker_id)
                    .unwrap()
                    .execute_hidden_pass(now_ms),
            };

            let mut effects = EngineEffects {
                admissions: executed.admissions,
                ..EngineEffects::default()
            };
            if let Some(fpm) = executed.fpm {
                effects.fpm_snapshots.push((worker_id, fpm));
            }
            let completion_kv_events =
                if executed.router_event_visibility == RouterEventVisibility::PassStart {
                    effects.pass_start_kv_events = executed.kv_events;
                    Vec::new()
                } else {
                    executed.kv_events
                };
            let payload = WorkerCompletionPayload {
                stage: self.stage,
                worker_idx: worker_id,
                completed_requests: executed.completed_requests,
                output_signals: executed.output_signals,
                kv_events: completion_kv_events,
            };

            if executed.end_ms == now_ms {
                effects.immediate_completions.push(payload);
                return Ok(effects);
            }

            self.workers.get_mut(&worker_id).unwrap().mark_busy();
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
        let worker = self.workers.get_mut(&payload.worker_idx).ok_or_else(|| {
            anyhow::anyhow!(
                "offline replay completion for unknown worker {}",
                payload.worker_idx
            )
        })?;
        worker.mark_idle();
        worker.mark_completed(payload.completed_requests);
        // Eagerly clean up drained workers that are pending removal so they
        // don't linger indefinitely when no further scaling events trigger
        // apply_target_count.
        if self.pending_removal.contains(&payload.worker_idx) {
            self.try_remove_drained();
        }
        Ok(payload)
    }

    pub(in crate::replay::offline) fn in_flight(&self) -> usize {
        self.workers
            .values()
            .map(OfflineWorkerState::in_flight)
            .sum()
    }

    pub(in crate::replay::offline) fn is_drained(&self) -> bool {
        self.workers.values().all(OfflineWorkerState::is_drained)
    }

    pub(in crate::replay::offline) fn worker_count(&self) -> usize {
        self.workers.len()
    }

    #[cfg(test)]
    pub(crate) fn debug_snapshots(&self) -> Vec<OfflineWorkerSnapshot> {
        self.workers
            .values()
            .map(OfflineWorkerState::debug_snapshot)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::protocols::MockEngineArgs;

    fn engine_with_startup(num_workers: usize, startup_time: Option<f64>) -> EngineComponent {
        let args = MockEngineArgs {
            startup_time,
            ..MockEngineArgs::default()
        };
        let workers: Vec<_> = (0..num_workers)
            .map(|i| OfflineWorkerState::new(i, args.clone(), false))
            .collect();
        let mut engine = EngineComponent::new(
            SimulationWorkerStage::Aggregated,
            EnginePassMode::Visible,
            workers,
        );
        engine.set_scaling_args(args, false);
        engine
    }

    #[test]
    fn test_apply_target_count_scale_up_with_startup() {
        let mut engine = engine_with_startup(2, Some(5.0));
        let (added, newly_marked) = engine.apply_target_count(4);

        assert_eq!(added.len(), 2);
        assert!(newly_marked.is_empty());
        // New workers are in pending_startup.
        assert_eq!(engine.active_worker_ids().len(), 2);
        assert_eq!(engine.worker_count(), 4);
    }

    #[test]
    fn test_apply_target_count_scale_up_without_startup() {
        let mut engine = engine_with_startup(2, None);
        let (added, newly_marked) = engine.apply_target_count(4);

        assert_eq!(added.len(), 2);
        assert!(newly_marked.is_empty());
        // Without startup delay, workers are immediately active.
        assert_eq!(engine.active_worker_ids().len(), 4);
        assert_eq!(engine.worker_count(), 4);
    }

    #[test]
    fn test_scale_down_cancels_startup_before_active() {
        let mut engine = engine_with_startup(2, Some(5.0));

        // Scale up to 4 — adds 2 in pending_startup.
        engine.apply_target_count(4);
        assert_eq!(engine.active_worker_ids().len(), 2);
        assert_eq!(engine.worker_count(), 4);

        // Scale down to 3 — should cancel 1 startup worker, not mark any active.
        let (_added, newly_marked) = engine.apply_target_count(3);
        assert!(newly_marked.is_empty());
        assert_eq!(engine.active_worker_ids().len(), 2);
        assert_eq!(engine.worker_count(), 3); // 2 active + 1 still starting

        // Scale down to 2 — should cancel the remaining startup worker.
        let (_added, newly_marked) = engine.apply_target_count(2);
        assert!(newly_marked.is_empty());
        assert_eq!(engine.active_worker_ids().len(), 2);
        assert_eq!(engine.worker_count(), 2);
    }

    #[test]
    fn test_scale_down_past_startup_marks_active() {
        let mut engine = engine_with_startup(3, Some(5.0));

        // Scale up to 5 — adds 2 in pending_startup.
        engine.apply_target_count(5);

        // Scale down to 1 — should cancel 2 startup, mark 2 active.
        let (_added, newly_marked) = engine.apply_target_count(1);
        assert_eq!(newly_marked.len(), 2);
        assert_eq!(engine.active_worker_ids().len(), 1);
    }

    #[test]
    fn test_mark_worker_ready_activates_pending() {
        let mut engine = engine_with_startup(1, Some(5.0));
        let (added, _) = engine.apply_target_count(2);
        let new_id = added[0];

        assert_eq!(engine.active_worker_ids().len(), 1);
        assert!(engine.mark_worker_ready(new_id));
        assert_eq!(engine.active_worker_ids().len(), 2);
    }

    #[test]
    fn test_mark_worker_ready_returns_false_for_cancelled() {
        let mut engine = engine_with_startup(1, Some(5.0));
        let (added, _) = engine.apply_target_count(2);
        let new_id = added[0];

        // Cancel by scaling back down.
        engine.apply_target_count(1);
        // Worker was removed from pending_startup and workers map.
        assert!(!engine.mark_worker_ready(new_id));
    }

    #[test]
    fn test_startup_time_ms_conversion() {
        let engine = engine_with_startup(1, Some(5.0));
        assert_eq!(engine.startup_time_ms(), Some(5000.0));

        let engine = engine_with_startup(1, None);
        assert_eq!(engine.startup_time_ms(), None);

        let engine = engine_with_startup(1, Some(0.0));
        assert_eq!(engine.startup_time_ms(), None); // 0 treated as no delay
    }
}
