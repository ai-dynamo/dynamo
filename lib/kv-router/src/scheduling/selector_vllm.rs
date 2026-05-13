// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Port of vLLM's DP load-balancer selection (`DPLBAsyncMPClient::get_core_engine_for_request`).
//!
//! Algorithm:
//!   for each engine i in rotated order starting at `start_index`:
//!     score(i) = waiting(i) * waiting_weight + running(i)
//!   pick the engine with the minimum score (first-seen wins on ties).
//!
//! Differences from upstream vLLM:
//!   * `eng_start_index` in vLLM is fixed per client (each client gets a different
//!     offset so they break ties differently). The dynamo selector serves many
//!     concurrent in-flight calls from one process, so we advance the index
//!     once per `select_worker` call. Net effect is identical at the
//!     fleet level: ties cycle through workers instead of always landing
//!     on the same first-encountered minimum.
//!   * vLLM optimistically bumps `current_counts[eng_index][0] += client_count`
//!     after each pick to keep balance between coordinator updates (every
//!     ~100 ms). We do not bump inside `select_worker` because selectors are
//!     required to be side-effect free (see `scheduling/CLAUDE.md`). The
//!     equivalent bump should happen at admission: call
//!     [`VllmDPLBSelector::record_admit`] from the same code path that books
//!     state after a successful selection.
//!   * No KV-cache awareness — this selector deliberately ignores
//!     `tier_overlap_blocks`, `effective_overlap_blocks`, and
//!     `shared_cache_hits`. Use [`DefaultWorkerSelector`] when prefix-cache
//!     hits should influence routing.
//!
//! [`DefaultWorkerSelector`]: super::selector::DefaultWorkerSelector

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::RwLock;
use rustc_hash::FxHashMap;

use super::config::{KvRouterConfig, SelectorKind};
use super::selector::{DefaultWorkerSelector, WorkerSelector};
use super::types::{KvSchedulerError, SchedulingRequest, pinned_worker_config};
use crate::protocols::{WorkerConfigLike, WorkerId, WorkerSelectionResult, WorkerWithDpRank};

/// vLLM default: `score = 4 * waiting + running`. The 4x weighting reflects
/// that a queued request has more pending work than one already in the running
/// batch (prefill vs continuing decode).
pub const DEFAULT_WAITING_WEIGHT: u32 = 4;

#[derive(Debug, Default, Clone, Copy)]
struct Counts {
    waiting: u32,
    running: u32,
}

/// vLLM-style DP load-balancing selector.
///
/// Holds internal per-worker `(waiting, running)` request counts. External code
/// (the admission/scheduling path) maintains the counts by calling
/// [`Self::record_admit`], [`Self::record_running`], and [`Self::record_finish`].
/// `select_worker` reads the counts but never mutates them.
#[derive(Debug)]
pub struct VllmDPLBSelector {
    waiting_weight: u32,
    start_index: AtomicUsize,
    counts: RwLock<FxHashMap<WorkerWithDpRank, Counts>>,
    worker_type: &'static str,
}

impl VllmDPLBSelector {
    pub fn new(waiting_weight: u32, worker_type: &'static str) -> Self {
        Self {
            waiting_weight,
            start_index: AtomicUsize::new(0),
            counts: RwLock::new(FxHashMap::default()),
            worker_type,
        }
    }

    /// Bump the waiting count for `worker`. Call after a successful
    /// `select_worker` to mirror vLLM's optimistic-increment behavior, so the
    /// next selection sees the bump.
    pub fn record_admit(&self, worker: WorkerWithDpRank) {
        let mut counts = self.counts.write();
        counts.entry(worker).or_default().waiting += 1;
    }

    /// Move a request from waiting → running for `worker` (e.g. when prefill
    /// completes and the request joins the running batch).
    pub fn record_running(&self, worker: WorkerWithDpRank) {
        let mut counts = self.counts.write();
        let entry = counts.entry(worker).or_default();
        entry.waiting = entry.waiting.saturating_sub(1);
        entry.running += 1;
    }

    /// Decrement the running count for `worker`. Call when a request finishes.
    pub fn record_finish(&self, worker: WorkerWithDpRank) {
        let mut counts = self.counts.write();
        if let Some(entry) = counts.get_mut(&worker) {
            entry.running = entry.running.saturating_sub(1);
        }
    }

    /// Forget all counts for a worker (e.g. on worker disconnect).
    pub fn forget_worker(&self, worker_id: WorkerId) {
        let mut counts = self.counts.write();
        counts.retain(|w, _| w.worker_id != worker_id);
    }

    fn counts_for(&self, worker: WorkerWithDpRank) -> Counts {
        self.counts.read().get(&worker).copied().unwrap_or_default()
    }

    fn score(&self, worker: WorkerWithDpRank) -> u64 {
        let c = self.counts_for(worker);
        (c.waiting as u64) * (self.waiting_weight as u64) + (c.running as u64)
    }

    /// Test-only setter so unit tests can stage `(waiting, running)` directly.
    #[cfg(test)]
    fn set_counts(&self, worker: WorkerWithDpRank, waiting: u32, running: u32) {
        self.counts
            .write()
            .insert(worker, Counts { waiting, running });
    }
}

impl<C: WorkerConfigLike> WorkerSelector<C> for VllmDPLBSelector {
    fn select_worker(
        &self,
        workers: &HashMap<WorkerId, C>,
        request: &SchedulingRequest,
        block_size: u32,
    ) -> Result<WorkerSelectionResult, KvSchedulerError> {
        assert!(request.isl_tokens > 0);
        request.validate_worker_constraints()?;

        let request_blocks = request.request_blocks(block_size);

        if let Some(worker) = request.pinned_worker {
            pinned_worker_config(workers, worker)?;
            tracing::info!(
                worker_type = self.worker_type,
                worker_id = worker.worker_id,
                dp_rank = worker.dp_rank,
                "Selected pinned worker (vllm_dplb)",
            );
            return Ok(WorkerSelectionResult {
                worker,
                required_blocks: request_blocks,
                effective_overlap_blocks: 0.0,
                cached_tokens: 0,
            });
        }

        // Expand allowed workers across their DP ranks, then sort to make
        // the rotation index meaningful. Without a stable order, scanning
        // by `(start + i) % n` would not produce consistent tie-breaking.
        let mut candidates: Vec<WorkerWithDpRank> = workers
            .iter()
            .filter(|(worker_id, _)| request.is_worker_allowed(**worker_id))
            .flat_map(|(worker_id, config)| {
                let start = config.data_parallel_start_rank();
                let end = start + config.data_parallel_size();
                (start..end).map(move |dp_rank| WorkerWithDpRank::new(*worker_id, dp_rank))
            })
            .collect();
        if candidates.is_empty() {
            return Err(KvSchedulerError::NoEndpoints);
        }
        candidates.sort_unstable();

        let n = candidates.len();
        // Relaxed is fine — the value is only used to bias tie-breaking; we
        // do not need cross-thread ordering with any other operation.
        let start = self.start_index.fetch_add(1, Ordering::Relaxed) % n;

        let mut best_idx = start;
        let mut best_score = self.score(candidates[start]);
        for i in 1..n {
            let idx = (start + i) % n;
            let score = self.score(candidates[idx]);
            if score < best_score {
                best_score = score;
                best_idx = idx;
            }
        }
        let best_worker = candidates[best_idx];

        tracing::info!(
            worker_type = self.worker_type,
            worker_id = best_worker.worker_id,
            dp_rank = best_worker.dp_rank,
            score = best_score,
            "Selected worker (vllm_dplb)",
        );

        Ok(WorkerSelectionResult {
            worker: best_worker,
            required_blocks: request_blocks,
            effective_overlap_blocks: 0.0,
            cached_tokens: 0,
        })
    }

    fn on_admit(&self, worker: WorkerWithDpRank) {
        self.record_admit(worker);
    }

    fn on_running(&self, worker: WorkerWithDpRank) {
        self.record_running(worker);
    }

    fn on_finish(&self, worker: WorkerWithDpRank) {
        self.record_finish(worker);
    }

    fn on_forget_worker(&self, worker_id: WorkerId) {
        self.forget_worker(worker_id);
    }
}

/// Config-driven enum dispatch over [`DefaultWorkerSelector`] and
/// [`VllmDPLBSelector`]. Use this as the `Sel` type parameter when the
/// concrete selector should be picked at runtime from [`KvRouterConfig`].
#[derive(Debug)]
pub enum AnyWorkerSelector {
    Default(DefaultWorkerSelector),
    VllmDplb(VllmDPLBSelector),
}

impl AnyWorkerSelector {
    /// Build the selector variant requested by `kv_router_config.selector_kind`.
    pub fn from_config(kv_router_config: KvRouterConfig, worker_type: &'static str) -> Self {
        match kv_router_config.selector_kind {
            SelectorKind::Default => Self::Default(DefaultWorkerSelector::new(
                Some(kv_router_config),
                worker_type,
            )),
            SelectorKind::VllmDplb => Self::VllmDplb(VllmDPLBSelector::new(
                kv_router_config.vllm_dplb_waiting_weight,
                worker_type,
            )),
        }
    }

    /// Access the underlying `VllmDPLBSelector` if this variant was chosen.
    /// Useful for tests and for any caller that needs to inspect the
    /// selector's internal counters.
    pub fn as_vllm_dplb(&self) -> Option<&VllmDPLBSelector> {
        match self {
            Self::VllmDplb(s) => Some(s),
            _ => None,
        }
    }
}

impl<C: WorkerConfigLike> WorkerSelector<C> for AnyWorkerSelector {
    fn select_worker(
        &self,
        workers: &std::collections::HashMap<WorkerId, C>,
        request: &SchedulingRequest,
        block_size: u32,
    ) -> Result<WorkerSelectionResult, KvSchedulerError> {
        match self {
            Self::Default(s) => s.select_worker(workers, request, block_size),
            Self::VllmDplb(s) => s.select_worker(workers, request, block_size),
        }
    }

    fn on_admit(&self, worker: WorkerWithDpRank) {
        match self {
            Self::Default(s) => <DefaultWorkerSelector as WorkerSelector<C>>::on_admit(s, worker),
            Self::VllmDplb(s) => <VllmDPLBSelector as WorkerSelector<C>>::on_admit(s, worker),
        }
    }

    fn on_running(&self, worker: WorkerWithDpRank) {
        match self {
            Self::Default(s) => <DefaultWorkerSelector as WorkerSelector<C>>::on_running(s, worker),
            Self::VllmDplb(s) => <VllmDPLBSelector as WorkerSelector<C>>::on_running(s, worker),
        }
    }

    fn on_finish(&self, worker: WorkerWithDpRank) {
        match self {
            Self::Default(s) => <DefaultWorkerSelector as WorkerSelector<C>>::on_finish(s, worker),
            Self::VllmDplb(s) => <VllmDPLBSelector as WorkerSelector<C>>::on_finish(s, worker),
        }
    }

    fn on_forget_worker(&self, worker_id: WorkerId) {
        match self {
            Self::Default(s) => {
                <DefaultWorkerSelector as WorkerSelector<C>>::on_forget_worker(s, worker_id)
            }
            Self::VllmDplb(s) => {
                <VllmDPLBSelector as WorkerSelector<C>>::on_forget_worker(s, worker_id)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::SimpleWorkerConfig;

    fn make_request(isl: usize) -> SchedulingRequest {
        let (tx, _rx) = tokio::sync::oneshot::channel();
        SchedulingRequest {
            maybe_request_id: Some("test".into()),
            token_seq: None,
            isl_tokens: isl,
            tier_overlap_blocks: Default::default(),
            effective_overlap_blocks: HashMap::new(),
            effective_cached_tokens: HashMap::new(),
            decode_blocks: FxHashMap::default(),
            prefill_tokens: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            update_states: false,
            lora_name: None,
            priority_jump: 0.0,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            shared_cache_hits: None,
            resp_tx: Some(tx),
        }
    }

    fn make_workers(ids: &[WorkerId]) -> HashMap<WorkerId, SimpleWorkerConfig> {
        ids.iter()
            .map(|id| (*id, SimpleWorkerConfig::default()))
            .collect()
    }

    /// Worker 1 has (waiting=2, running=0) → score = 8.
    /// Worker 2 has (waiting=0, running=5) → score = 5.
    /// Worker 2 wins. Matches vLLM's `waiting * 4 + running`.
    #[test]
    fn scoring_matches_vllm_formula() {
        let selector = VllmDPLBSelector::new(DEFAULT_WAITING_WEIGHT, "test");
        selector.set_counts(WorkerWithDpRank::from_worker_id(1), 2, 0);
        selector.set_counts(WorkerWithDpRank::from_worker_id(2), 0, 5);

        let workers = make_workers(&[1, 2]);
        let request = make_request(16);
        let result = selector.select_worker(&workers, &request, 16).unwrap();
        assert_eq!(result.worker, WorkerWithDpRank::from_worker_id(2));
    }

    /// All workers tied at zero load — rotation should walk through them in
    /// sorted order across successive calls.
    #[test]
    fn rotated_tie_break_on_equal_load() {
        let selector = VllmDPLBSelector::new(DEFAULT_WAITING_WEIGHT, "test");
        let workers = make_workers(&[1, 2, 3]);
        let request = make_request(16);

        let picks: Vec<WorkerId> = (0..6)
            .map(|_| {
                selector
                    .select_worker(&workers, &request, 16)
                    .unwrap()
                    .worker
                    .worker_id
            })
            .collect();
        // Sorted candidate order is [1, 2, 3]; with start incrementing each
        // call we should see 1, 2, 3, 1, 2, 3.
        assert_eq!(picks, vec![1, 2, 3, 1, 2, 3]);
    }

    /// Optimistic increment after a pick should steer the next call elsewhere.
    #[test]
    fn record_admit_bumps_waiting() {
        let selector = VllmDPLBSelector::new(DEFAULT_WAITING_WEIGHT, "test");
        let workers = make_workers(&[1, 2]);
        let request = make_request(16);

        let first = selector
            .select_worker(&workers, &request, 16)
            .unwrap()
            .worker;
        selector.record_admit(first);
        // After bumping waiting on `first`, the second pick should land on
        // the other worker (its score is still 0, vs first's 4).
        let second = selector
            .select_worker(&workers, &request, 16)
            .unwrap()
            .worker;
        assert_ne!(first, second);
    }

    #[test]
    fn pinned_worker_short_circuits_scoring() {
        let selector = VllmDPLBSelector::new(DEFAULT_WAITING_WEIGHT, "test");
        // Make worker 1 look heavily loaded — it should still win when pinned.
        selector.set_counts(WorkerWithDpRank::from_worker_id(1), 999, 999);
        let workers = make_workers(&[1, 2]);

        let mut request = make_request(16);
        request.pinned_worker = Some(WorkerWithDpRank::from_worker_id(1));

        let result = selector.select_worker(&workers, &request, 16).unwrap();
        assert_eq!(result.worker, WorkerWithDpRank::from_worker_id(1));
    }

    #[test]
    fn allowed_worker_ids_filters_candidates() {
        let selector = VllmDPLBSelector::new(DEFAULT_WAITING_WEIGHT, "test");
        let workers = make_workers(&[1, 2, 3]);
        let mut request = make_request(16);
        request.allowed_worker_ids = Some([2u64].into_iter().collect());

        // Whatever start_index is, only worker 2 is allowed.
        for _ in 0..5 {
            let result = selector.select_worker(&workers, &request, 16).unwrap();
            assert_eq!(result.worker.worker_id, 2);
        }
    }

    #[test]
    fn no_allowed_candidates_returns_no_endpoints() {
        let selector = VllmDPLBSelector::new(DEFAULT_WAITING_WEIGHT, "test");
        let workers = make_workers(&[1, 2]);
        let mut request = make_request(16);
        request.allowed_worker_ids = Some([99u64].into_iter().collect());

        let err = selector.select_worker(&workers, &request, 16).unwrap_err();
        assert!(matches!(err, KvSchedulerError::NoEndpoints));
    }

    /// `record_running` moves a request from waiting to running; combined
    /// score should drop because waiting is weighted 4x.
    #[test]
    fn record_running_lowers_score() {
        let selector = VllmDPLBSelector::new(DEFAULT_WAITING_WEIGHT, "test");
        let w = WorkerWithDpRank::from_worker_id(1);
        selector.set_counts(w, 1, 0);
        assert_eq!(selector.score(w), 4);
        selector.record_running(w);
        assert_eq!(selector.score(w), 1);
    }

    #[test]
    fn forget_worker_clears_state() {
        let selector = VllmDPLBSelector::new(DEFAULT_WAITING_WEIGHT, "test");
        let w = WorkerWithDpRank::from_worker_id(1);
        selector.set_counts(w, 5, 5);
        selector.forget_worker(1);
        assert_eq!(selector.score(w), 0);
    }

    #[test]
    fn any_worker_selector_from_config_picks_vllm_dplb() {
        let config = KvRouterConfig {
            selector_kind: SelectorKind::VllmDplb,
            vllm_dplb_waiting_weight: 4,
            ..Default::default()
        };
        let sel = AnyWorkerSelector::from_config(config, "test");
        assert!(sel.as_vllm_dplb().is_some());
    }

    #[test]
    fn any_worker_selector_from_config_defaults_to_default_kind() {
        let sel = AnyWorkerSelector::from_config(KvRouterConfig::default(), "test");
        assert!(sel.as_vllm_dplb().is_none());
    }

    /// Lifecycle hooks should reach the inner VllmDPLBSelector through
    /// AnyWorkerSelector dispatch.
    #[test]
    fn any_worker_selector_dispatches_hooks_to_inner_vllm_dplb() {
        let config = KvRouterConfig {
            selector_kind: SelectorKind::VllmDplb,
            vllm_dplb_waiting_weight: 4,
            ..Default::default()
        };
        let sel = AnyWorkerSelector::from_config(config, "test");
        let worker = WorkerWithDpRank::from_worker_id(7);
        <AnyWorkerSelector as WorkerSelector<SimpleWorkerConfig>>::on_admit(&sel, worker);
        // The on_admit hook should have bumped the inner selector's waiting count.
        let inner = sel.as_vllm_dplb().unwrap();
        assert_eq!(inner.score(worker), 4);
        <AnyWorkerSelector as WorkerSelector<SimpleWorkerConfig>>::on_running(&sel, worker);
        // waiting 1 -> 0, running 0 -> 1, score = 1.
        assert_eq!(inner.score(worker), 1);
        <AnyWorkerSelector as WorkerSelector<SimpleWorkerConfig>>::on_finish(&sel, worker);
        assert_eq!(inner.score(worker), 0);
    }

    /// Custom weight: with weight=1, waiting and running count equally.
    #[test]
    fn custom_waiting_weight() {
        let selector = VllmDPLBSelector::new(1, "test");
        selector.set_counts(WorkerWithDpRank::from_worker_id(1), 3, 0);
        selector.set_counts(WorkerWithDpRank::from_worker_id(2), 0, 2);
        let workers = make_workers(&[1, 2]);
        let request = make_request(16);
        // worker 1 score = 3, worker 2 score = 2 → worker 2 wins.
        let result = selector.select_worker(&workers, &request, 16).unwrap();
        assert_eq!(result.worker.worker_id, 2);
    }
}
