// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Narrow Queue integration for bounded soft-drain routing.

use std::collections::{HashMap, HashSet};

use tokio::time::Instant;

use super::SchedulerQueueActor;
use crate::protocols::{WorkerConfigLike, WorkerId, WorkerSelectionResult, WorkerWithDpRank};
use crate::scheduling::cold_pool::{ColdPoolLane, ColdPoolState};
use crate::scheduling::filter::RoutingEligibility;
use crate::scheduling::overlap_refresh::OverlapScoresRefresh;
use crate::scheduling::policy_config::PolicyClassConfig;
use crate::scheduling::selector::WorkerSelector;
use crate::scheduling::types::{KvSchedulerError, SchedulingContext, SchedulingRequest};
use crate::sequences::SequencePublisher;

impl<
    P: SequencePublisher + 'static,
    C: WorkerConfigLike + Send + Sync + 'static,
    Sel: WorkerSelector<C> + Send + 'static,
    RF: OverlapScoresRefresh + Send + Sync + 'static,
> SchedulerQueueActor<P, C, Sel, RF>
{
    pub(super) fn reconcile_cold_pool(&mut self) {
        let Some(state) = self.cold_pool_state.as_mut() else {
            return;
        };
        let workers = self.workers_with_configs.borrow();
        state.reconcile_membership(workers.keys().copied());
        state.reconcile_active_cold(|request_id, worker| {
            self.slots.request_worker(request_id) == Some(worker)
        });
    }

    pub(super) fn prepare_cold_pool_request(
        &mut self,
        request: &SchedulingRequest,
    ) -> Option<ColdPoolLane> {
        if !request.mode.is_tracked() {
            return None;
        }
        self.reconcile_cold_pool();
        let state = self.cold_pool_state.as_ref()?;
        let workers = self.workers_with_configs.borrow();
        let lane = state.classify(request.isl_tokens, || {
            SchedulingContext::new(request, &workers).best_effective_prefill_tokens()
        })?;
        if lane == ColdPoolLane::Cold {
            let candidates = cold_worker_ids(state, request, &workers);
            if candidates.is_empty()
                || !eligibility_for(request, Some(&candidates))
                    .any_eligible_worker_rank(&workers, |_, _| true)
            {
                return None;
            }
        }
        Some(lane)
    }

    pub(super) fn cold_pool_request_is_dispatchable(
        &self,
        class: &PolicyClassConfig,
        request: &SchedulingRequest,
        lane: Option<ColdPoolLane>,
        enqueue_at: Instant,
        now: Instant,
    ) -> bool {
        let active_tokens = self.slots.active_tokens(now);
        let active_counts = self.slots.active_request_counts();
        let workers = self.workers_with_configs.borrow();
        Self::cold_pool_request_is_dispatchable_with(
            self.cold_pool_state.as_ref(),
            class,
            request,
            lane,
            Some(enqueue_at),
            &active_tokens,
            &active_counts,
            &workers,
            now,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn cold_pool_request_is_dispatchable_with(
        state: Option<&ColdPoolState>,
        class: &PolicyClassConfig,
        request: &SchedulingRequest,
        lane: Option<ColdPoolLane>,
        enqueue_at: Option<Instant>,
        active_tokens: &HashMap<WorkerWithDpRank, usize>,
        active_counts: &HashMap<WorkerWithDpRank, usize>,
        workers: &HashMap<WorkerId, C>,
        now: Instant,
    ) -> bool {
        let Some((state, lane)) = state.zip(lane) else {
            return !Self::all_workers_prefill_busy_with(
                active_tokens,
                workers,
                class,
                request.eligibility(),
            );
        };

        match lane {
            ColdPoolLane::Warm => {
                let preferred = warm_preferred_worker_ids(state, request, workers);
                let eligibility = preferred.as_ref().map_or_else(
                    || request.eligibility(),
                    |ids| eligibility_for(request, Some(ids)),
                );
                !Self::all_workers_prefill_busy_with(active_tokens, workers, class, eligibility)
            }
            ColdPoolLane::Cold => {
                let fallback = cold_worker_ids(state, request, workers);
                if fallback.is_empty() {
                    // Cold containment is no longer structurally possible.
                    // Fall back to ordinary queue admission so busy workers
                    // still queue while invalid eligibility reaches selector.
                    return !Self::all_workers_prefill_busy_with(
                        active_tokens,
                        workers,
                        class,
                        request.eligibility(),
                    );
                }
                let preferred =
                    clean_cold_worker_ids(state, request, workers, active_counts, &fallback);
                if !preferred.is_empty()
                    && !Self::all_workers_prefill_busy_with(
                        active_tokens,
                        workers,
                        class,
                        eligibility_for(request, Some(&preferred)),
                    )
                {
                    return true;
                }

                let expired = enqueue_at.is_none_or(|at| state.soft_drain_expired(at, now));
                expired
                    && !Self::all_workers_prefill_busy_with(
                        active_tokens,
                        workers,
                        class,
                        eligibility_for(request, Some(&fallback)),
                    )
            }
        }
    }

    pub(super) fn select_worker_with_cold_pool(
        &self,
        request: &SchedulingRequest,
        lane: Option<ColdPoolLane>,
        class: Option<&PolicyClassConfig>,
        workers: &HashMap<WorkerId, C>,
        overloaded_worker_ids: Option<&HashSet<WorkerId>>,
    ) -> Result<WorkerSelectionResult, KvSchedulerError> {
        let Some((state, lane)) = self.cold_pool_state.as_ref().zip(lane) else {
            return self.selector.select_worker(
                workers,
                request,
                request.eligibility_with_overloaded(overloaded_worker_ids),
                self.block_size,
            );
        };

        let active_counts = self.slots.active_request_counts();
        let (preferred, fallback) = match lane {
            ColdPoolLane::Warm => (warm_preferred_worker_ids(state, request, workers), None),
            ColdPoolLane::Cold => {
                let fallback = cold_worker_ids(state, request, workers);
                if fallback.is_empty() {
                    return self.selector.select_worker(
                        workers,
                        request,
                        request.eligibility_with_overloaded(overloaded_worker_ids),
                        self.block_size,
                    );
                }
                let preferred =
                    clean_cold_worker_ids(state, request, workers, &active_counts, &fallback);
                ((!preferred.is_empty()).then_some(preferred), Some(fallback))
            }
        };

        let skip_preferred = lane == ColdPoolLane::Cold
            && preferred.as_ref().is_some_and(|preferred| {
                preferred_workers_prefill_busy(
                    request,
                    workers,
                    class.expect("Cold Pool selection must have a resolved policy class"),
                    preferred,
                )
            });
        if !skip_preferred && let Some(preferred) = preferred.as_ref() {
            match self.selector.select_worker(
                workers,
                request,
                eligibility_for_with_overload(request, Some(preferred), overloaded_worker_ids),
                self.block_size,
            ) {
                Ok(selection) => return Ok(selection),
                Err(
                    KvSchedulerError::NoEndpoints | KvSchedulerError::AllEligibleWorkersOverloaded,
                ) => {}
                Err(error) => return Err(error),
            }
        }

        match fallback {
            Some(mut fallback) => {
                if skip_preferred && let Some(preferred) = preferred {
                    fallback.retain(|worker| !preferred.contains(worker));
                }
                self.selector.select_worker(
                    workers,
                    request,
                    eligibility_for_with_overload(request, Some(&fallback), overloaded_worker_ids),
                    self.block_size,
                )
            }
            None => self.selector.select_worker(
                workers,
                request,
                request.eligibility_with_overloaded(overloaded_worker_ids),
                self.block_size,
            ),
        }
    }

    pub(super) fn on_cold_pool_request_queued(&mut self, lane: Option<ColdPoolLane>) {
        if let Some((state, lane)) = self.cold_pool_state.as_mut().zip(lane) {
            state.on_queued(lane);
        }
    }

    pub(super) fn on_cold_pool_request_dequeued(&mut self, lane: Option<ColdPoolLane>) {
        if let Some((state, lane)) = self.cold_pool_state.as_mut().zip(lane) {
            state.on_dequeued(lane);
        }
    }

    pub(super) fn on_cold_pool_request_dispatched(
        &mut self,
        request_id: String,
        lane: Option<ColdPoolLane>,
        worker: WorkerWithDpRank,
    ) {
        if let Some((state, lane)) = self.cold_pool_state.as_mut().zip(lane) {
            state.on_dispatched(request_id, lane, worker);
        }
    }
}

fn preferred_workers_prefill_busy<C: WorkerConfigLike>(
    request: &SchedulingRequest,
    workers: &HashMap<WorkerId, C>,
    class: &PolicyClassConfig,
    preferred: &HashSet<WorkerId>,
) -> bool {
    let eligibility = eligibility_for(request, Some(preferred));
    let mut checked_any = false;
    let has_available = eligibility.any_eligible_worker_rank(workers, |worker, config| {
        checked_any = true;
        let max_batched = config
            .max_num_batched_tokens()
            .unwrap_or(super::DEFAULT_MAX_BATCHED_TOKENS);
        let active_tokens = request.worker_load_for(worker).active_prefill_tokens;
        !class.worker_is_busy(active_tokens, max_batched)
    });
    checked_any && !has_available
}

fn warm_preferred_worker_ids<C: WorkerConfigLike>(
    state: &ColdPoolState,
    request: &SchedulingRequest,
    workers: &HashMap<WorkerId, C>,
) -> Option<HashSet<WorkerId>> {
    let avoid = state.warm_avoid_worker_ids();
    if avoid.is_empty() {
        return None;
    }
    let preferred: HashSet<_> = workers
        .keys()
        .copied()
        .filter(|worker_id| request.eligibility().caller_allows_worker_id(*worker_id))
        .filter(|worker_id| !avoid.contains(worker_id))
        .collect();
    let has_structural_candidate =
        eligibility_for(request, Some(&preferred)).any_eligible_worker_rank(workers, |_, _| true);
    has_structural_candidate.then_some(preferred)
}

fn cold_worker_ids<C: WorkerConfigLike>(
    state: &ColdPoolState,
    request: &SchedulingRequest,
    workers: &HashMap<WorkerId, C>,
) -> HashSet<WorkerId> {
    state
        .cold_worker_ids()
        .iter()
        .copied()
        .filter(|worker_id| workers.contains_key(worker_id))
        .filter(|worker_id| request.eligibility().caller_allows_worker_id(*worker_id))
        .collect()
}

fn clean_cold_worker_ids<C: WorkerConfigLike>(
    state: &ColdPoolState,
    request: &SchedulingRequest,
    workers: &HashMap<WorkerId, C>,
    active_counts: &HashMap<WorkerWithDpRank, usize>,
    fallback: &HashSet<WorkerId>,
) -> HashSet<WorkerId> {
    let eligibility = eligibility_for(request, Some(fallback));
    let mut clean = HashSet::new();
    let mut has_active_warm = HashSet::new();
    eligibility.for_each_eligible_worker_rank(workers, |worker, _| {
        clean.insert(worker.worker_id);
        let total_active = active_counts.get(&worker).copied().unwrap_or(0);
        if state.active_warm(worker, total_active) > 0 {
            has_active_warm.insert(worker.worker_id);
        }
    });
    clean.retain(|worker_id| !has_active_warm.contains(worker_id));
    clean
}

fn eligibility_for<'a>(
    request: &'a SchedulingRequest,
    allowed_worker_ids: Option<&'a HashSet<WorkerId>>,
) -> RoutingEligibility<'a> {
    eligibility_for_with_overload(request, allowed_worker_ids, None)
}

fn eligibility_for_with_overload<'a>(
    request: &'a SchedulingRequest,
    allowed_worker_ids: Option<&'a HashSet<WorkerId>>,
    overloaded_worker_ids: Option<&'a HashSet<WorkerId>>,
) -> RoutingEligibility<'a> {
    RoutingEligibility::new(
        allowed_worker_ids,
        overloaded_worker_ids,
        request.pinned_worker,
        &request.routing_constraints,
    )
}
