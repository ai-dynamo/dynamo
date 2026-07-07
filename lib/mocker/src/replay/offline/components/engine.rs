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
use crate::common::protocols::{DirectRequest, ForwardPassSnapshot, MockEngineArgs};
use crate::replay::TraceCollector;
use crate::scheduler::RouterEventVisibility;
use crate::scheduler::{SchedulerCommand, SchedulerCommandEffects};

fn fpm_has_scheduled_work(snapshot: &ForwardPassSnapshot) -> bool {
    snapshot.num_prefill_requests > 0 || snapshot.num_decode_requests > 0
}

pub(in crate::replay::offline) struct EngineComponent {
    stage: SimulationWorkerStage,
    pass_mode: EnginePassMode,
    /// DP-rank schedulers keyed by stable ID (monotonic, never reused).
    workers: BTreeMap<usize, OfflineWorkerState>,
    /// Mocker worker IDs mapped to their per-rank scheduler IDs.
    worker_groups: BTreeMap<usize, Vec<usize>>,
    /// Counter for generating the next stable scheduler ID.
    next_id: usize,
    /// Counter for generating the next stable mocker worker ID.
    next_worker_id: usize,
    /// Mocker workers marked for removal — skipped by round-robin, removed when drained.
    pending_removal: BTreeSet<usize>,
    /// Mocker workers still starting up — excluded from active set until ready.
    pending_startup: BTreeSet<usize>,
    /// Engine args used to construct new DP-rank schedulers during scale-up.
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
        let worker_groups = (0..count).map(|id| (id, vec![id])).collect();
        Self {
            stage,
            pass_mode,
            workers: map,
            worker_groups,
            next_id: count,
            next_worker_id: count,
            pending_removal: BTreeSet::new(),
            pending_startup: BTreeSet::new(),
            args: MockEngineArgs::default(),
            capture_kv_events: false,
        }
    }

    /// Build one scheduler core per DP rank while retaining the live mocker's
    /// `(worker_id, dp_rank)` topology.
    pub(in crate::replay::offline) fn new_ranked(
        stage: SimulationWorkerStage,
        pass_mode: EnginePassMode,
        args: MockEngineArgs,
        num_workers: usize,
        capture_kv_events: bool,
    ) -> Self {
        let dp_size = args.dp_size.max(1) as usize;
        let mut workers = BTreeMap::new();
        let mut worker_groups = BTreeMap::new();
        for worker_id in 0..num_workers {
            let mut rank_ids = Vec::with_capacity(dp_size);
            for dp_rank in 0..dp_size {
                let rank_id = worker_id * dp_size + dp_rank;
                workers.insert(
                    rank_id,
                    OfflineWorkerState::new_with_rank(
                        rank_id,
                        worker_id as u64,
                        dp_rank as u32,
                        args.clone(),
                        capture_kv_events,
                    ),
                );
                rank_ids.push(rank_id);
            }
            worker_groups.insert(worker_id, rank_ids);
        }
        Self {
            stage,
            pass_mode,
            workers,
            worker_groups,
            next_id: num_workers.saturating_mul(dp_size),
            next_worker_id: num_workers,
            pending_removal: BTreeSet::new(),
            pending_startup: BTreeSet::new(),
            args,
            capture_kv_events,
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

    /// Add a new mocker worker and all of its DP-rank schedulers, returning
    /// the stable mocker worker ID.
    pub(in crate::replay::offline) fn add_worker(&mut self) -> usize {
        let worker_id = self.next_worker_id;
        self.next_worker_id += 1;
        let mut rank_ids = Vec::with_capacity(self.args.dp_size.max(1) as usize);
        for dp_rank in 0..self.args.dp_size.max(1) {
            let rank_id = self.next_id;
            self.next_id += 1;
            let worker = OfflineWorkerState::new_with_rank(
                rank_id,
                worker_id as u64,
                dp_rank,
                self.args.clone(),
                self.capture_kv_events,
            );
            self.workers.insert(rank_id, worker);
            rank_ids.push(rank_id);
        }
        self.worker_groups.insert(worker_id, rank_ids);
        worker_id
    }

    /// Mark a worker for removal. Round-robin routing skips marked workers;
    /// in router mode the caller must also remove the worker from the router
    /// (see `apply_target_count`). The worker remains eligible for
    /// `drive_ready` until its existing work drains.
    pub(in crate::replay::offline) fn mark_for_removal(&mut self, worker_id: usize) {
        self.pending_removal.insert(worker_id);
    }

    /// Remove all marked workers that have fully drained, returning their IDs.
    pub(in crate::replay::offline) fn try_remove_drained(&mut self) -> Vec<usize> {
        let mut removed = Vec::new();
        self.pending_removal.retain(|&id| {
            if let Some(rank_ids) = self.worker_groups.get(&id) {
                if rank_ids.iter().all(|rank_id| {
                    self.workers
                        .get(rank_id)
                        .is_none_or(OfflineWorkerState::is_drained)
                }) {
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
            if let Some(rank_ids) = self.worker_groups.remove(&id) {
                for rank_id in rank_ids {
                    self.workers.remove(&rank_id);
                }
            }
        }
        removed
    }

    /// Apply a target worker count: add new workers or mark excess for removal.
    /// Returns `(added_ids, newly_marked_ids, removed_ids)` so the caller can
    /// update the router immediately. Newly marked workers should be removed
    /// from routing eligibility right away; removed workers have already
    /// drained and their retained router state can be finalized.
    ///
    /// The effective count is `active + pending_startup` — workers that will
    /// be active once all startups complete. On scale-down, pending startup
    /// workers are cancelled first (cheapest: no in-flight work, no router
    /// registration), then active workers are marked for removal.
    pub(in crate::replay::offline) fn apply_target_count(
        &mut self,
        target: usize,
    ) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        let active_ids = self.active_group_ids();
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
                if let Some(rank_ids) = self.worker_groups.remove(&id) {
                    for rank_id in rank_ids {
                        self.workers.remove(&rank_id);
                    }
                }
            }

            // Mark active workers for removal if more excess remains.
            let remaining = excess - to_cancel.len();
            for &id in active_ids.iter().rev().take(remaining) {
                self.mark_for_removal(id);
                newly_marked.push(id);
            }
        }

        // Clean up any workers that have already fully drained.
        let removed = self.try_remove_drained();
        (added, newly_marked, removed)
    }

    /// Return stable scheduler IDs of all active DP ranks — excludes ranks in
    /// mocker workers pending removal or startup.
    pub(in crate::replay::offline) fn active_worker_ids(&self) -> Vec<usize> {
        self.active_group_ids()
            .into_iter()
            .flat_map(|worker_id| self.worker_groups[&worker_id].iter().copied())
            .collect()
    }

    /// Return stable mocker worker IDs that are active for new admissions.
    pub(in crate::replay::offline) fn active_group_ids(&self) -> Vec<usize> {
        self.worker_groups
            .keys()
            .filter(|id| !self.pending_removal.contains(id) && !self.pending_startup.contains(id))
            .copied()
            .collect()
    }

    pub(in crate::replay::offline) fn rank_id(
        &self,
        worker_id: usize,
        dp_rank: u32,
    ) -> Option<usize> {
        self.worker_groups
            .get(&worker_id)
            .and_then(|rank_ids| rank_ids.get(dp_rank as usize))
            .copied()
    }

    /// Return the logical mocker worker and DP rank represented by a stable
    /// scheduler ID.
    pub(in crate::replay::offline) fn rank_identity(&self, rank_id: usize) -> Option<(usize, u32)> {
        let (worker_id, dp_rank) = self.workers.get(&rank_id)?.rank_identity();
        Some((usize::try_from(worker_id).ok()?, dp_rank))
    }

    pub(in crate::replay::offline) fn has_active_workers(&self) -> bool {
        !self.active_group_ids().is_empty()
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
        self.pending_startup.remove(&worker_id) && self.worker_groups.contains_key(&worker_id)
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

    pub(in crate::replay::offline) fn apply_command(
        &mut self,
        worker_id: usize,
        command: SchedulerCommand,
    ) -> anyhow::Result<SchedulerCommandEffects> {
        let worker = self
            .workers
            .get_mut(&worker_id)
            .ok_or_else(|| anyhow::anyhow!("offline replay selected unknown worker {worker_id}"))?;
        worker.apply_command(command)
    }

    pub(in crate::replay::offline) fn worker_is_busy(
        &self,
        worker_id: usize,
    ) -> anyhow::Result<bool> {
        let worker = self
            .workers
            .get(&worker_id)
            .ok_or_else(|| anyhow::anyhow!("offline replay selected unknown worker {worker_id}"))?;
        Ok(worker.is_busy())
    }

    pub(in crate::replay::offline) fn drive_ready(
        &mut self,
        now_ms: f64,
        mut collector: Option<&mut TraceCollector>,
    ) -> anyhow::Result<EngineEffects> {
        let worker_groups: Vec<Vec<usize>> = self.worker_groups.values().cloned().collect();
        for rank_ids in worker_groups {
            // A logical attention-DP worker advances in group-owned epochs.
            // A rank that received work mid-epoch must wait until every sibling
            // has crossed the prior completion boundary.
            if rank_ids
                .iter()
                .any(|rank_id| self.workers.get(rank_id).unwrap().is_busy())
            {
                continue;
            }
            if !rank_ids
                .iter()
                .any(|rank_id| self.workers.get(rank_id).unwrap().is_ready())
            {
                continue;
            }

            let mut executed_by_rank = BTreeMap::new();
            for &rank_id in &rank_ids {
                if !self.workers.get(&rank_id).unwrap().is_ready() {
                    continue;
                }
                let executed = match self.pass_mode {
                    EnginePassMode::Visible => {
                        let Some(collector) = collector.as_deref_mut() else {
                            bail!("offline replay visible engine pass requires a collector");
                        };
                        self.workers
                            .get_mut(&rank_id)
                            .unwrap()
                            .execute_pass(collector, now_ms)
                    }
                    EnginePassMode::Hidden => self
                        .workers
                        .get_mut(&rank_id)
                        .unwrap()
                        .execute_hidden_pass(now_ms),
                };
                executed_by_rank.insert(rank_id, executed);
            }

            let group_end_ms = executed_by_rank
                .values()
                .map(|executed| executed.end_ms)
                .fold(now_ms, f64::max);
            let group_wall_time_secs = (group_end_ms - now_ms).max(0.0) / 1000.0;
            let mut effects = EngineEffects::default();

            for &rank_id in &rank_ids {
                let Some(mut executed) = executed_by_rank.remove(&rank_id) else {
                    if group_end_ms > now_ms {
                        // Empty ranks still participate in the barrier so work
                        // arriving mid-epoch cannot start ahead of a sibling.
                        self.workers.get_mut(&rank_id).unwrap().mark_busy();
                        effects
                            .scheduled_completions
                            .push(ScheduledWorkerCompletion {
                                at_ms: group_end_ms,
                                payload: WorkerCompletionPayload {
                                    stage: self.stage,
                                    worker_idx: rank_id,
                                    completed_requests: 0,
                                    output_signals: Vec::new(),
                                    lifecycle_events: Vec::new(),
                                    kv_events: Vec::new(),
                                    fpm: Some(ForwardPassSnapshot {
                                        wall_time_secs: group_wall_time_secs,
                                        ..Default::default()
                                    }),
                                    accept_length_output_tokens: 0,
                                    accept_length_decode_forwards: 0,
                                },
                            });
                    }
                    continue;
                };

                if let Some(fpm) = executed.fpm.as_mut() {
                    fpm.wall_time_secs = group_wall_time_secs;
                }
                if self.pass_mode == EnginePassMode::Visible {
                    collector
                        .as_deref_mut()
                        .expect("visible pass collector checked before execution")
                        .align_pass_token_times(&executed.output_signals, group_end_ms);
                }

                let admitted_requests = !executed.admissions.is_empty();
                effects.admissions.extend(executed.admissions);
                let published_pass_start_kv = executed.router_event_visibility
                    == RouterEventVisibility::PassStart
                    && !executed.kv_events.is_empty();
                let completion_kv_events =
                    if executed.router_event_visibility == RouterEventVisibility::PassStart {
                        effects.pass_start_kv_events.extend(executed.kv_events);
                        Vec::new()
                    } else {
                        executed.kv_events
                    };
                let payload = WorkerCompletionPayload {
                    stage: self.stage,
                    worker_idx: rank_id,
                    completed_requests: executed.completed_requests,
                    output_signals: executed.output_signals,
                    lifecycle_events: executed.lifecycle_events,
                    kv_events: completion_kv_events,
                    fpm: executed.fpm,
                    accept_length_output_tokens: executed.accept_length_output_tokens,
                    accept_length_decode_forwards: executed.accept_length_decode_forwards,
                };

                if group_end_ms > now_ms {
                    self.workers.get_mut(&rank_id).unwrap().mark_busy();
                    effects
                        .scheduled_completions
                        .push(ScheduledWorkerCompletion {
                            at_ms: group_end_ms,
                            payload,
                        });
                    continue;
                }

                // NOTE: Keep both lifecycle extremes in view when changing this gate.
                // Tight-spin/livelock occurs when an effect-free, queued-only,
                // zero-duration pass is repeatedly treated as progress at the same
                // virtual timestamp. Dead-end/lost-wakeup occurs when replay declares
                // quiescence while workers still own unfinished requests but have no
                // concrete future event, deadline, or dependency notification. Stop
                // same-time iteration when no observable state changed, but only after
                // the owning subsystem can account for every unfinished request's
                // future wakeup; an empty event queue alone is not quiescence.
                let made_progress = admitted_requests
                    || published_pass_start_kv
                    || payload.completed_requests > 0
                    || !payload.output_signals.is_empty()
                    || !payload.lifecycle_events.is_empty()
                    || !payload.kv_events.is_empty()
                    || payload.fpm.as_ref().is_some_and(fpm_has_scheduled_work);
                if made_progress {
                    effects.immediate_completions.push(payload);
                }
            }

            if !effects.is_empty() {
                return Ok(effects);
            }
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
        let mut payload = payload;
        payload
            .lifecycle_events
            .extend(worker.retry_pending_destinations());
        payload.kv_events.extend(worker.drain_kv_events());
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
        self.worker_groups.len()
    }

    #[cfg(test)]
    pub(in crate::replay::offline) fn rank_id_capacity(&self) -> usize {
        self.next_id
    }

    #[cfg(feature = "kvbm-offload")]
    pub(in crate::replay::offline) fn earliest_offload_deadline(&self) -> Option<f64> {
        self.workers
            .values()
            .filter_map(OfflineWorkerState::earliest_offload_deadline)
            .reduce(f64::min)
    }

    #[cfg(feature = "kvbm-offload")]
    pub(in crate::replay::offline) fn tick_offload_engines(
        &mut self,
        now_ms: f64,
    ) -> crate::scheduler::OffloadTickEffects {
        let mut effects = crate::scheduler::OffloadTickEffects {
            kv_events: Vec::new(),
            lifecycle_events: Vec::new(),
        };
        for worker in self.workers.values_mut() {
            let worker_effects = if worker.is_busy() {
                worker.tick_offload_transport_only(now_ms)
            } else {
                worker.tick_offload_only(now_ms)
            };
            effects.kv_events.extend(worker_effects.kv_events);
            effects
                .lifecycle_events
                .extend(worker_effects.lifecycle_events);
        }
        effects
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
    use crate::common::handoff::HandoffId;
    use crate::common::perf_model::{AicCallback, PerfModel};
    use crate::common::protocols::{
        DirectRequest, EngineType, MockEngineArgs, SglangArgs, WorkerType,
    };
    use crate::scheduler::{SchedulerCommand, SchedulerCommandResult, SchedulerLifecycleEvent};
    use std::sync::Arc;
    use uuid::Uuid;

    #[cfg(feature = "kvbm-offload")]
    use dynamo_kv_router::protocols::StorageTier;

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

    fn take_only_completion(mut effects: EngineEffects) -> WorkerCompletionPayload {
        if let Some(payload) = effects.immediate_completions.pop() {
            assert!(effects.scheduled_completions.is_empty());
            return payload;
        }
        assert_eq!(effects.scheduled_completions.len(), 1);
        effects.scheduled_completions.pop().unwrap().payload
    }

    fn decode_engine_with_chunking(enable_chunked_prefill: bool) -> EngineComponent {
        let args = MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(16)
            .max_num_batched_tokens(Some(4))
            .max_num_seqs(Some(1))
            .enable_chunked_prefill(enable_chunked_prefill)
            .worker_type(WorkerType::Decode)
            .build()
            .unwrap();
        EngineComponent::new(
            SimulationWorkerStage::Decode,
            EnginePassMode::Visible,
            vec![OfflineWorkerState::new(0, args, false)],
        )
    }

    struct LengthLatency;

    impl AicCallback for LengthLatency {
        fn predict_prefill(&self, _batch_size: usize, effective_isl: usize, _prefix: usize) -> f64 {
            effective_isl as f64
        }

        fn predict_decode(&self, _batch_size: usize, _isl: usize, _osl: usize) -> f64 {
            1.0
        }
    }

    fn ranked_timing_engine(dp_size: u32) -> EngineComponent {
        let args = MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(64)
            .max_num_batched_tokens(Some(64))
            .max_num_seqs(Some(1))
            .enable_prefix_caching(false)
            .speedup_ratio(1.0)
            .dp_size(dp_size)
            .perf_model(Arc::new(PerfModel::from_aic_callback(Arc::new(
                LengthLatency,
            ))))
            .build()
            .unwrap();
        EngineComponent::new_ranked(
            SimulationWorkerStage::Aggregated,
            EnginePassMode::Visible,
            args,
            1,
            false,
        )
    }

    fn timed_request(uuid: Uuid, input_length: usize) -> DirectRequest {
        DirectRequest {
            tokens: vec![1; input_length],
            max_output_tokens: 1,
            uuid: Some(uuid),
            ..Default::default()
        }
    }

    #[test]
    fn attention_dp_group_completes_at_slowest_rank_boundary() {
        let mut engine = ranked_timing_engine(2);
        let fast = Uuid::from_u128(30);
        let slow = Uuid::from_u128(31);
        engine.dispatch(0, timed_request(fast, 4)).unwrap();
        engine.dispatch(1, timed_request(slow, 8)).unwrap();
        let mut collector = TraceCollector::default();
        collector.on_arrival(fast, 0.0, 4, 1);
        collector.on_arrival(slow, 0.0, 8, 1);

        let effects = engine.drive_ready(0.0, Some(&mut collector)).unwrap();

        assert_eq!(effects.scheduled_completions.len(), 2);
        assert!(
            effects
                .scheduled_completions
                .iter()
                .all(|completion| (completion.at_ms - 9.0).abs() < f64::EPSILON)
        );
        assert!(effects.scheduled_completions.iter().all(|completion| {
            (completion.payload.fpm.as_ref().unwrap().wall_time_secs - 0.009).abs() < f64::EPSILON
        }));
        assert_eq!(collector.request_latencies(fast).unwrap().0, 9.0);
        assert_eq!(collector.request_latencies(slow).unwrap().0, 9.0);
    }

    #[test]
    fn attention_dp_empty_rank_blocks_mid_epoch_arrival() {
        let mut engine = ranked_timing_engine(2);
        let slow = Uuid::from_u128(40);
        engine.dispatch(0, timed_request(slow, 8)).unwrap();
        let mut collector = TraceCollector::default();
        collector.on_arrival(slow, 0.0, 8, 1);

        let first_epoch = engine.drive_ready(0.0, Some(&mut collector)).unwrap();
        assert_eq!(first_epoch.scheduled_completions.len(), 2);
        assert!(engine.debug_snapshots().iter().all(|rank| rank.busy));
        let idle_fpm = first_epoch
            .scheduled_completions
            .iter()
            .find(|completion| completion.payload.worker_idx == 1)
            .unwrap()
            .payload
            .fpm
            .as_ref()
            .expect("idle DP rank must emit an empty FPM");
        assert!(!fpm_has_scheduled_work(idle_fpm));
        assert_eq!(idle_fpm.num_queued_prefill, 0);
        assert_eq!(idle_fpm.num_queued_decode, 0);
        assert!((idle_fpm.wall_time_secs - 0.009).abs() < f64::EPSILON);

        let mid_epoch = Uuid::from_u128(41);
        collector.on_arrival(mid_epoch, 4.0, 4, 1);
        engine.dispatch(1, timed_request(mid_epoch, 4)).unwrap();
        assert!(
            engine
                .drive_ready(4.0, Some(&mut collector))
                .unwrap()
                .is_empty()
        );

        for completion in first_epoch.scheduled_completions {
            engine.on_scheduled_completion(completion.payload).unwrap();
        }
        let second_epoch = engine.drive_ready(9.0, Some(&mut collector)).unwrap();
        assert_eq!(second_epoch.scheduled_completions.len(), 2);
        assert!(
            second_epoch
                .scheduled_completions
                .iter()
                .all(|completion| (completion.at_ms - 14.0).abs() < f64::EPSILON)
        );
        assert_eq!(collector.request_latencies(mid_epoch).unwrap().0, 10.0);
    }

    #[test]
    fn single_rank_group_keeps_local_completion_time() {
        let mut engine = ranked_timing_engine(1);
        let uuid = Uuid::from_u128(50);
        engine.dispatch(0, timed_request(uuid, 4)).unwrap();
        let mut collector = TraceCollector::default();
        collector.on_arrival(uuid, 0.0, 4, 1);

        let effects = engine.drive_ready(0.0, Some(&mut collector)).unwrap();

        assert_eq!(effects.scheduled_completions.len(), 1);
        assert_eq!(effects.scheduled_completions[0].at_ms, 5.0);
        assert_eq!(collector.request_latencies(uuid).unwrap().0, 5.0);
    }

    #[test]
    fn kv_visibility_follows_backend_contract_and_fpm_waits_for_completion() {
        let make_engine = |engine_type| {
            let args = MockEngineArgs::builder()
                .engine_type(engine_type)
                .num_gpu_blocks(8)
                .block_size(4)
                .max_num_batched_tokens(Some(16))
                .max_num_seqs(Some(1))
                .enable_prefix_caching(true)
                .speedup_ratio(0.0)
                .sglang(Some(SglangArgs {
                    page_size: Some(4),
                    chunked_prefill_size: Some(16),
                    ..Default::default()
                }))
                .build()
                .unwrap();
            EngineComponent::new(
                SimulationWorkerStage::Decode,
                EnginePassMode::Visible,
                vec![OfflineWorkerState::new(0, args, true)],
            )
        };
        let request = |uuid| DirectRequest {
            tokens: vec![1; 8],
            max_output_tokens: 1,
            uuid: Some(uuid),
            ..Default::default()
        };
        let mut collector = TraceCollector::default();

        let mut vllm = make_engine(EngineType::Vllm);
        vllm.dispatch(0, request(Uuid::from_u128(10))).unwrap();
        let vllm_start = vllm.drive_ready(0.0, Some(&mut collector)).unwrap();
        assert!(!vllm_start.pass_start_kv_events.is_empty());
        let vllm_end = take_only_completion(vllm_start);
        assert!(vllm_end.kv_events.is_empty());
        assert!(vllm_end.fpm.is_some());

        let mut sglang = make_engine(EngineType::Sglang);
        sglang.dispatch(0, request(Uuid::from_u128(11))).unwrap();
        let sglang_start = sglang.drive_ready(0.0, Some(&mut collector)).unwrap();
        assert!(sglang_start.pass_start_kv_events.is_empty());
        let sglang_end = take_only_completion(sglang_start);
        assert!(!sglang_end.kv_events.is_empty());
        assert!(sglang_end.fpm.is_some());
    }

    #[test]
    fn test_apply_target_count_scale_up_with_startup() {
        let mut engine = engine_with_startup(2, Some(5.0));
        let (added, newly_marked, _) = engine.apply_target_count(4);

        assert_eq!(added.len(), 2);
        assert!(newly_marked.is_empty());
        // New workers are in pending_startup.
        assert_eq!(engine.active_worker_ids().len(), 2);
        assert_eq!(engine.worker_count(), 4);
    }

    #[test]
    fn test_apply_target_count_scale_up_without_startup() {
        let mut engine = engine_with_startup(2, None);
        let (added, newly_marked, _) = engine.apply_target_count(4);

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
        let (_added, newly_marked, _) = engine.apply_target_count(3);
        assert!(newly_marked.is_empty());
        assert_eq!(engine.active_worker_ids().len(), 2);
        assert_eq!(engine.worker_count(), 3); // 2 active + 1 still starting

        // Scale down to 2 — should cancel the remaining startup worker.
        let (_added, newly_marked, _) = engine.apply_target_count(2);
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
        let (_added, newly_marked, _) = engine.apply_target_count(1);
        assert_eq!(newly_marked.len(), 2);
        assert_eq!(engine.active_worker_ids().len(), 1);
    }

    #[test]
    fn test_mark_worker_ready_activates_pending() {
        let mut engine = engine_with_startup(1, Some(5.0));
        let (added, _, _) = engine.apply_target_count(2);
        let new_id = added[0];

        assert_eq!(engine.active_worker_ids().len(), 1);
        assert!(engine.mark_worker_ready(new_id));
        assert_eq!(engine.active_worker_ids().len(), 2);
    }

    #[test]
    fn productive_zero_duration_pass_reaches_terminal_completion() {
        let mut engine = decode_engine_with_chunking(true);
        let uuid = Uuid::from_u128(20);
        engine
            .dispatch(
                0,
                DirectRequest {
                    tokens: vec![1; 12],
                    max_output_tokens: 1,
                    uuid: Some(uuid),
                    ..Default::default()
                },
            )
            .unwrap();
        let mut collector = TraceCollector::default();

        let first = engine.drive_ready(0.0, Some(&mut collector)).unwrap();
        assert_eq!(first.admissions.len(), 1);
        let first = take_only_completion(first);
        assert_eq!(first.completed_requests, 0);
        engine.on_scheduled_completion(first).unwrap();

        let second = engine.drive_ready(0.0, Some(&mut collector)).unwrap();
        assert!(second.admissions.is_empty());
        assert!(second.pass_start_kv_events.is_empty());
        assert_eq!(second.immediate_completions.len(), 1);
        assert!(second.scheduled_completions.is_empty());
        let second = take_only_completion(second);
        assert_eq!(second.fpm.as_ref().unwrap().num_prefill_requests, 1);
        assert_eq!(second.completed_requests, 0);
        assert!(second.output_signals.is_empty());
        assert!(second.lifecycle_events.is_empty());
        assert!(second.kv_events.is_empty());
        engine.on_scheduled_completion(second).unwrap();

        let final_pass = engine.drive_ready(0.0, Some(&mut collector)).unwrap();
        let final_pass = take_only_completion(final_pass);
        assert_eq!(final_pass.completed_requests, 1);
        assert!(
            final_pass
                .output_signals
                .iter()
                .any(|signal| signal.uuid == uuid && signal.completed)
        );
        engine.on_scheduled_completion(final_pass).unwrap();
        assert!(engine.is_drained());
    }

    #[test]
    fn queued_only_zero_duration_pass_does_not_report_progress() {
        let mut engine = decode_engine_with_chunking(false);
        engine
            .dispatch(
                0,
                DirectRequest {
                    tokens: vec![1; 8],
                    max_output_tokens: 1,
                    uuid: Some(Uuid::from_u128(21)),
                    ..Default::default()
                },
            )
            .unwrap();
        let mut collector = TraceCollector::default();

        let first = engine.drive_ready(0.0, Some(&mut collector)).unwrap();
        assert!(first.is_empty());
        assert_eq!(
            engine.debug_snapshots(),
            vec![OfflineWorkerSnapshot {
                busy: false,
                in_flight: 1,
                ready: true,
                drained: false,
            }]
        );

        let retry = engine.drive_ready(0.0, Some(&mut collector)).unwrap();
        assert!(retry.is_empty());
    }

    #[test]
    fn test_mark_worker_ready_returns_false_for_cancelled() {
        let mut engine = engine_with_startup(1, Some(5.0));
        let (added, _, _) = engine.apply_target_count(2);
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

    #[test]
    fn pending_destination_keeps_scaled_down_worker_alive_until_cleanup() {
        let args = MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(64)
            .max_num_batched_tokens(Some(64))
            .max_num_seqs(Some(1))
            .speedup_ratio(1.0)
            .decode_speedup_ratio(1.0)
            .worker_type(WorkerType::Decode)
            .build()
            .unwrap();
        let worker = OfflineWorkerState::new(0, args.clone(), false);
        let mut engine = EngineComponent::new(
            SimulationWorkerStage::Decode,
            EnginePassMode::Visible,
            vec![worker],
        );
        engine.set_scaling_args(args, false);

        engine
            .dispatch(
                0,
                DirectRequest {
                    tokens: vec![1; 4],
                    max_output_tokens: 1,
                    uuid: Some(Uuid::from_u128(1)),
                    ..Default::default()
                },
            )
            .unwrap();
        let mut collector = TraceCollector::default();
        let mut pass = engine.drive_ready(0.0, Some(&mut collector)).unwrap();
        assert_eq!(pass.scheduled_completions.len(), 1);

        let handoff_id = HandoffId::from(Uuid::from_u128(2));
        let effects = engine
            .apply_command(
                0,
                SchedulerCommand::ReserveDestination {
                    handoff_id,
                    request: DirectRequest {
                        tokens: vec![2; 4],
                        max_output_tokens: 1,
                        uuid: Some(Uuid::from_u128(2)),
                        ..Default::default()
                    },
                },
            )
            .unwrap();
        assert!(matches!(
            effects.result,
            SchedulerCommandResult::DestinationAccepted {
                request_id
            } if request_id == Uuid::from_u128(2)
        ));
        assert!(effects.lifecycle_events.is_empty());

        let (_, newly_marked, _) = engine.apply_target_count(0);
        assert_eq!(newly_marked, vec![0]);
        assert!(engine.active_worker_ids().is_empty());
        assert_eq!(engine.worker_count(), 1);

        let completion = pass.scheduled_completions.pop().unwrap();
        let payload = engine.on_scheduled_completion(completion.payload).unwrap();
        assert!(payload.lifecycle_events.iter().any(|event| matches!(
            event,
            SchedulerLifecycleEvent::DestinationReserved {
                handoff_id: observed,
                request_id,
                ..
            } if *observed == handoff_id && *request_id == Uuid::from_u128(2)
        )));
        assert_eq!(engine.worker_count(), 1);

        let effects = engine
            .apply_command(0, SchedulerCommand::CancelDestination { handoff_id })
            .unwrap();
        assert_eq!(effects.result, SchedulerCommandResult::Applied);
        assert_eq!(engine.try_remove_drained(), vec![0]);
        assert_eq!(engine.worker_count(), 0);
    }

    #[cfg(feature = "kvbm-offload")]
    #[test]
    fn busy_worker_advances_offload_without_destination_admission() {
        let args = MockEngineArgs::builder()
            .num_gpu_blocks(1)
            .block_size(4)
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(1))
            .enable_prefix_caching(true)
            .worker_type(WorkerType::Decode)
            .speedup_ratio(1.0)
            .kv_bytes_per_token(Some(5_000_000))
            .num_g2_blocks(Some(4))
            .bandwidth_g1_to_g2_gbps(Some(1.0))
            .build()
            .unwrap();
        let worker = OfflineWorkerState::new(0, args, true);
        let mut engine = EngineComponent::new(
            SimulationWorkerStage::Decode,
            EnginePassMode::Visible,
            vec![worker],
        );
        engine
            .dispatch(
                0,
                DirectRequest {
                    tokens: vec![1; 4],
                    max_output_tokens: 1,
                    uuid: Some(Uuid::from_u128(101)),
                    ..Default::default()
                },
            )
            .unwrap();
        let mut collector = TraceCollector::default();
        let mut seed = engine.drive_ready(0.0, Some(&mut collector)).unwrap();
        assert!(
            seed.pass_start_kv_events
                .iter()
                .any(|event| { event.storage_tier == StorageTier::Device })
        );
        assert_eq!(
            seed.immediate_completions.len() + seed.scheduled_completions.len(),
            1,
            "seed request should produce exactly one completion boundary"
        );
        let completion = seed.immediate_completions.pop().unwrap_or_else(|| {
            seed.scheduled_completions
                .pop()
                .expect("seed request completion must be present")
                .payload
        });
        // Model a reservation attempt at the t=0 boundary, before the GPU
        // compute interval becomes externally busy.
        engine.workers.get_mut(&0).unwrap().mark_idle();

        let handoff_id = HandoffId::from(Uuid::from_u128(102));
        let reserve = engine
            .apply_command(
                0,
                SchedulerCommand::ReserveDestination {
                    handoff_id,
                    request: DirectRequest {
                        tokens: vec![2; 4],
                        max_output_tokens: 1,
                        uuid: Some(Uuid::from_u128(103)),
                        ..Default::default()
                    },
                },
            )
            .unwrap();
        assert!(reserve.lifecycle_events.is_empty());
        let deadline = engine
            .earliest_offload_deadline()
            .expect("reservation eviction should start G1 to G2 DMA");
        assert!((deadline - 20.0).abs() < 0.01);

        engine.workers.get_mut(&0).unwrap().mark_busy();
        assert_eq!(engine.earliest_offload_deadline(), Some(deadline));
        let transport = engine.tick_offload_engines(deadline);
        assert!(transport.lifecycle_events.is_empty());
        assert!(
            transport
                .kv_events
                .iter()
                .any(|event| event.storage_tier == StorageTier::HostPinned)
        );

        let boundary = engine.on_scheduled_completion(completion).unwrap();
        assert!(boundary.lifecycle_events.iter().any(|event| matches!(
            event,
            SchedulerLifecycleEvent::DestinationReserved {
                handoff_id: observed,
                request_id,
                ..
            } if *observed == handoff_id && *request_id == Uuid::from_u128(103)
        )));
    }
}
