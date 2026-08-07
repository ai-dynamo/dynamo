// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Actor-local state for bounded soft-drain warm/cold routing.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use tokio::time::Instant;

use crate::protocols::{WorkerId, WorkerWithDpRank};

const COLD_POOL_RECHECK_INTERVAL: Duration = Duration::from_millis(100);

pub(crate) fn periodic_recheck_interval(configured: Duration) -> Duration {
    configured.min(COLD_POOL_RECHECK_INTERVAL)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ColdPoolConfig {
    /// Raw requests below this length enter the Warm lane. Longer requests
    /// enter the Cold lane only when their best effective prefill is also at
    /// or above this threshold.
    pub(crate) request_threshold: usize,
    /// Requested number of Cold workers before topology clamping. The effective
    /// pool is capped at one third of the registered workers, with at least one
    /// Cold worker whenever the topology can also retain a Warm worker.
    pub(crate) workers: usize,
    /// Maximum time Cold waits for a member without active Warm work.
    pub(crate) soft_drain_timeout: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ColdPoolLane {
    Warm,
    Cold,
}

pub(crate) struct ColdPoolState {
    config: ColdPoolConfig,
    cold_worker_ids: HashSet<WorkerId>,
    pending_cold: usize,
    active_cold_requests: HashMap<String, WorkerWithDpRank>,
    active_cold_by_worker: HashMap<WorkerWithDpRank, usize>,
}

impl ColdPoolState {
    pub(crate) fn new(config: ColdPoolConfig) -> Self {
        Self {
            config,
            cold_worker_ids: HashSet::new(),
            pending_cold: 0,
            active_cold_requests: HashMap::new(),
            active_cold_by_worker: HashMap::new(),
        }
    }

    pub(crate) fn classify(
        &self,
        raw_isl_tokens: usize,
        best_effective_prefill_tokens: impl FnOnce() -> usize,
    ) -> Option<ColdPoolLane> {
        if raw_isl_tokens < self.config.request_threshold {
            return Some(ColdPoolLane::Warm);
        }
        if best_effective_prefill_tokens() >= self.config.request_threshold {
            return Some(ColdPoolLane::Cold);
        }
        None
    }

    pub(crate) fn soft_drain_expired(&self, enqueue_at: Instant, now: Instant) -> bool {
        now.saturating_duration_since(enqueue_at) >= self.config.soft_drain_timeout
    }

    pub(crate) fn cold_worker_ids(&self) -> &HashSet<WorkerId> {
        &self.cold_worker_ids
    }

    pub(crate) fn effective_worker_count(&self, worker_count: usize) -> usize {
        if worker_count < 2 {
            return 0;
        }

        let topology_cap = (worker_count / 3).max(1);
        self.config.workers.clamp(1, topology_cap)
    }

    pub(crate) fn reconcile_membership(&mut self, worker_ids: impl IntoIterator<Item = WorkerId>) {
        let mut worker_ids: Vec<_> = worker_ids.into_iter().collect();
        worker_ids.sort_unstable();
        worker_ids.dedup();
        let effective = self.effective_worker_count(worker_ids.len());
        worker_ids.sort_unstable_by(|left, right| {
            rendezvous_score(*right)
                .cmp(&rendezvous_score(*left))
                .then_with(|| left.cmp(right))
        });
        let next: HashSet<_> = worker_ids.into_iter().take(effective).collect();
        if next != self.cold_worker_ids {
            self.cold_worker_ids = next;
        }
    }

    pub(crate) fn on_queued(&mut self, lane: ColdPoolLane) {
        if lane == ColdPoolLane::Cold {
            self.pending_cold += 1;
        }
    }

    pub(crate) fn on_dequeued(&mut self, lane: ColdPoolLane) {
        if lane == ColdPoolLane::Cold {
            debug_assert!(self.pending_cold > 0, "Cold Pool pending counter underflow");
            self.pending_cold = self.pending_cold.saturating_sub(1);
        }
    }

    pub(crate) fn on_dispatched(
        &mut self,
        request_id: String,
        lane: ColdPoolLane,
        worker: WorkerWithDpRank,
    ) {
        if lane != ColdPoolLane::Cold {
            return;
        }
        self.active_cold_requests.insert(request_id, worker);
        *self.active_cold_by_worker.entry(worker).or_default() += 1;
    }

    pub(crate) fn reconcile_active_cold(
        &mut self,
        mut is_active: impl FnMut(&String, WorkerWithDpRank) -> bool,
    ) {
        self.active_cold_requests
            .retain(|request_id, worker| is_active(request_id, *worker));
        self.active_cold_by_worker.clear();
        for worker in self.active_cold_requests.values().copied() {
            *self.active_cold_by_worker.entry(worker).or_default() += 1;
        }
    }

    pub(crate) fn active_cold(&self, worker: WorkerWithDpRank) -> usize {
        self.active_cold_by_worker
            .get(&worker)
            .copied()
            .unwrap_or(0)
    }

    pub(crate) fn active_warm(&self, worker: WorkerWithDpRank, total_active: usize) -> usize {
        total_active.saturating_sub(self.active_cold(worker))
    }

    pub(crate) fn warm_avoid_worker_ids(&self) -> HashSet<WorkerId> {
        if self.pending_cold > 0 {
            return self.cold_worker_ids.clone();
        }
        self.active_cold_by_worker
            .iter()
            .filter_map(|(worker, active)| (*active > 0).then_some(worker.worker_id))
            .collect()
    }
}

/// SplitMix64 gives every Worker ID a stable, well-distributed HRW score.
fn rendezvous_score(worker_id: WorkerId) -> u64 {
    let mut value = worker_id.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(workers: usize) -> ColdPoolConfig {
        ColdPoolConfig {
            request_threshold: 128_000,
            workers,
            soft_drain_timeout: Duration::from_secs(1),
        }
    }

    #[test]
    fn membership_is_stable_and_leaves_one_warm_worker() {
        let mut state = ColdPoolState::new(config(10));
        state.reconcile_membership([1, 2, 3, 4]);
        assert_eq!(state.cold_worker_ids.len(), 1);

        let first = state.cold_worker_ids.clone();
        state.reconcile_membership([4, 3, 2, 1]);
        assert_eq!(state.cold_worker_ids, first);

        state.reconcile_membership([1]);
        assert!(state.cold_worker_ids.is_empty());
    }

    #[test]
    fn effective_worker_count_is_bounded_by_topology() {
        let state = ColdPoolState::new(config(10));
        let expected = [
            (0, 0),
            (1, 0),
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 1),
            (6, 2),
            (8, 2),
            (9, 3),
        ];

        for (worker_count, cold_count) in expected {
            assert_eq!(state.effective_worker_count(worker_count), cold_count);
        }

        let configured_one = ColdPoolState::new(config(1));
        assert_eq!(configured_one.effective_worker_count(9), 1);
    }

    #[test]
    fn pressure_tracks_pending_and_active_cold() {
        let mut state = ColdPoolState::new(config(1));
        state.reconcile_membership([1, 2]);
        let cold_worker = *state.cold_worker_ids.iter().next().unwrap();
        let lane = state.classify(400_000, || 400_000).unwrap();

        state.on_queued(lane);
        assert_eq!(state.warm_avoid_worker_ids(), state.cold_worker_ids);
        state.on_dequeued(lane);
        state.on_dispatched(
            "cold-1".to_string(),
            lane,
            WorkerWithDpRank::new(cold_worker, 0),
        );
        assert_eq!(state.warm_avoid_worker_ids(), state.cold_worker_ids);

        state.reconcile_active_cold(|_, _| false);
        assert!(state.warm_avoid_worker_ids().is_empty());
    }

    #[test]
    fn classification_bypasses_cache_affine_long_contexts() {
        let state = ColdPoolState::new(config(1));

        assert_eq!(
            state.classify(127_999, || panic!(
                "Warm classification must not inspect cache"
            )),
            Some(ColdPoolLane::Warm)
        );
        assert_eq!(
            state.classify(128_000, || 128_000),
            Some(ColdPoolLane::Cold)
        );
        assert_eq!(
            state.classify(400_000, || 128_000),
            Some(ColdPoolLane::Cold)
        );
        assert_eq!(state.classify(400_000, || 127_999), None);
    }

    #[test]
    fn soft_drain_timeout_is_bounded_by_enqueue_age() {
        let state = ColdPoolState::new(config(1));
        let enqueued = Instant::now();
        assert!(!state.soft_drain_expired(enqueued, enqueued + Duration::from_millis(999)));
        assert!(state.soft_drain_expired(enqueued, enqueued + Duration::from_secs(1)));
    }

    #[test]
    fn bounded_drain_caps_the_existing_recheck_interval() {
        assert_eq!(
            periodic_recheck_interval(Duration::from_secs(60)),
            Duration::from_millis(100)
        );
        assert_eq!(
            periodic_recheck_interval(Duration::from_millis(50)),
            Duration::from_millis(50)
        );
    }
}
