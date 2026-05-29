// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use dynamo_tokens::SequenceHash;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use super::config::RouterConfigOverride;
use crate::protocols::{
    DpRank, RouterBackpressureReason, RoutingConstraints, SharedCacheHits, WorkerConfigLike,
    WorkerId, WorkerWithDpRank,
};
use crate::sequences::PrefillTokenDeltas;

pub type OverloadedWorkerProvider =
    Arc<dyn Fn() -> Option<HashSet<WorkerId>> + Send + Sync + 'static>;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TierOverlapBlocks {
    #[serde(default)]
    pub device: FxHashMap<WorkerWithDpRank, usize>,
    #[serde(default)]
    pub host_pinned: FxHashMap<WorkerWithDpRank, usize>,
    #[serde(default)]
    pub disk: FxHashMap<WorkerWithDpRank, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PotentialLoad {
    pub worker_id: WorkerId,
    pub dp_rank: DpRank,
    pub potential_prefill_tokens: usize,
    pub potential_decode_blocks: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum KvSchedulerError {
    #[error("no endpoints available to route work")]
    NoEndpoints,

    #[error(
        "router backpressure: {reason:?} (queued_isl_tokens={queued_isl_tokens}, max_queued_isl_tokens={max_queued_isl_tokens:?})"
    )]
    Backpressure {
        reason: RouterBackpressureReason,
        queued_isl_tokens: usize,
        max_queued_isl_tokens: Option<usize>,
    },

    #[error("all eligible workers are overloaded")]
    AllEligibleWorkersOverloaded,

    #[error("pinned worker {worker_id} is overloaded")]
    PinnedWorkerOverloaded { worker_id: WorkerId },

    #[error("pinned worker {worker_id} is not in allowed worker set")]
    PinnedWorkerNotAllowed { worker_id: WorkerId },

    #[error("endpoint subscriber shutdown")]
    SubscriberShutdown,

    #[error("failed to initialize event publisher: {0}")]
    InitFailed(String),
}

impl KvSchedulerError {
    pub fn is_overload(&self) -> bool {
        matches!(
            self,
            Self::Backpressure { .. }
                | Self::AllEligibleWorkersOverloaded
                | Self::PinnedWorkerOverloaded { .. }
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WorkerEligibilityError {
    #[error("worker {worker_id} is not in allowed worker set")]
    WorkerNotAllowed { worker_id: WorkerId },

    #[error("worker {worker_id} is unavailable")]
    WorkerUnavailable { worker_id: WorkerId },

    #[error("worker {worker_id} dp_rank {dp_rank} is outside [{start}, {end})")]
    DpRankUnavailable {
        worker_id: WorkerId,
        dp_rank: DpRank,
        start: DpRank,
        end: DpRank,
    },

    #[error("worker {worker_id} is overloaded")]
    WorkerOverloaded { worker_id: WorkerId },

    #[error("worker {worker_id} does not satisfy routing constraints")]
    RoutingConstraintsUnsatisfied { worker_id: WorkerId },
}

#[derive(Debug)]
pub struct SchedulingResponse {
    pub best_worker: WorkerWithDpRank,
    pub effective_overlap_blocks: f64,
    pub cached_tokens: usize,
}

pub struct SchedulingRequest {
    // Request identity and payload.
    pub maybe_request_id: Option<String>,
    pub token_seq: Option<Vec<SequenceHash>>,
    pub isl_tokens: usize,
    pub lora_name: Option<String>,
    pub expected_output_tokens: Option<u32>,

    // Routing constraints and request-level config.
    pub pinned_worker: Option<WorkerWithDpRank>,
    pub allowed_worker_ids: Option<HashSet<WorkerId>>,
    pub routing_constraints: RoutingConstraints,
    pub router_config_override: Option<RouterConfigOverride>,
    pub track_prefill_tokens: bool,
    pub priority_jump: f64,

    // Overlap and cache signals.
    pub tier_overlap_blocks: TierOverlapBlocks,
    pub effective_overlap_blocks: HashMap<WorkerWithDpRank, f64>,
    pub effective_cached_tokens: HashMap<WorkerWithDpRank, usize>,
    pub shared_cache_hits: Option<SharedCacheHits>,

    // Load state computed during admission.
    pub decode_blocks: FxHashMap<WorkerWithDpRank, usize>,
    pub prefill_tokens: FxHashMap<WorkerWithDpRank, usize>,

    // Scheduling side effects and lifecycle controls.
    pub update_states: bool,
    pub resp_tx: Option<tokio::sync::oneshot::Sender<Result<SchedulingResponse, KvSchedulerError>>>,
}

#[derive(Clone, Copy)]
pub struct RoutingEligibility<'a> {
    allowed_worker_ids: Option<&'a HashSet<WorkerId>>,
    overloaded_worker_ids: Option<&'a HashSet<WorkerId>>,
    pinned_worker: Option<WorkerWithDpRank>,
    routing_constraints: &'a RoutingConstraints,
}

impl<'a> RoutingEligibility<'a> {
    #[inline]
    pub fn new(
        allowed_worker_ids: Option<&'a HashSet<WorkerId>>,
        overloaded_worker_ids: Option<&'a HashSet<WorkerId>>,
        pinned_worker: Option<WorkerWithDpRank>,
        routing_constraints: &'a RoutingConstraints,
    ) -> Self {
        Self {
            allowed_worker_ids,
            overloaded_worker_ids,
            pinned_worker,
            routing_constraints,
        }
    }

    #[inline]
    pub fn pinned_worker(&self) -> Option<WorkerWithDpRank> {
        self.pinned_worker
    }

    #[inline]
    pub fn caller_allows_worker_id(&self, worker_id: WorkerId) -> bool {
        self.allowed_worker_ids
            .is_none_or(|worker_ids| worker_ids.contains(&worker_id))
    }

    #[inline]
    pub fn is_worker_overloaded(&self, worker_id: WorkerId) -> bool {
        self.overloaded_worker_ids
            .is_some_and(|worker_ids| worker_ids.contains(&worker_id))
    }

    #[inline]
    pub fn allows_worker_id(&self, worker_id: WorkerId) -> bool {
        self.caller_allows_worker_id(worker_id) && !self.is_worker_overloaded(worker_id)
    }

    #[inline]
    pub fn allows_worker_ignoring_overload<C: WorkerConfigLike>(
        &self,
        worker_id: WorkerId,
        config: &C,
    ) -> bool {
        self.caller_allows_worker_id(worker_id)
            && self
                .routing_constraints
                .is_compatible_with_worker_taints(config.taints())
    }

    #[inline]
    pub fn allows_worker<C: WorkerConfigLike>(&self, worker_id: WorkerId, config: &C) -> bool {
        self.allows_worker_id(worker_id)
            && self
                .routing_constraints
                .is_compatible_with_worker_taints(config.taints())
    }

    #[inline]
    pub fn has_eligible_worker<'w, C, I>(&self, workers: I) -> bool
    where
        C: WorkerConfigLike + 'w,
        I: IntoIterator<Item = (WorkerId, &'w C)>,
    {
        for (worker_id, config) in workers {
            if !self.allows_worker_id(worker_id) {
                continue;
            }

            if self.allows_worker(worker_id, config) {
                return true;
            }
        }

        false
    }

    #[inline]
    pub fn has_eligible_worker_ignoring_overload<'w, C, I>(&self, workers: I) -> bool
    where
        C: WorkerConfigLike + 'w,
        I: IntoIterator<Item = (WorkerId, &'w C)>,
    {
        for (worker_id, config) in workers {
            if self.allows_worker_ignoring_overload(worker_id, config) {
                return true;
            }
        }

        false
    }

    #[inline]
    pub fn validate_worker_rank<'w, C: WorkerConfigLike>(
        &self,
        workers: &'w HashMap<WorkerId, C>,
        worker: WorkerWithDpRank,
    ) -> Result<&'w C, WorkerEligibilityError> {
        if !self.caller_allows_worker_id(worker.worker_id) {
            return Err(WorkerEligibilityError::WorkerNotAllowed {
                worker_id: worker.worker_id,
            });
        }

        let config = worker_config_for_rank(workers, worker)?;
        if !self
            .routing_constraints
            .is_compatible_with_worker_taints(config.taints())
        {
            return Err(WorkerEligibilityError::RoutingConstraintsUnsatisfied {
                worker_id: worker.worker_id,
            });
        }

        if self.is_worker_overloaded(worker.worker_id) {
            return Err(WorkerEligibilityError::WorkerOverloaded {
                worker_id: worker.worker_id,
            });
        }

        Ok(config)
    }

    pub fn any_eligible_worker_rank<C, F>(
        &self,
        workers: &HashMap<WorkerId, C>,
        mut predicate: F,
    ) -> bool
    where
        C: WorkerConfigLike,
        F: FnMut(WorkerWithDpRank, &C) -> bool,
    {
        if let Some(worker) = self.pinned_worker {
            let Ok(config) = self.validate_worker_rank(workers, worker) else {
                return false;
            };
            return predicate(worker, config);
        }

        for (&worker_id, config) in workers {
            if !self.allows_worker(worker_id, config) {
                continue;
            }

            let dp_start = config.data_parallel_start_rank();
            let dp_end = dp_start + config.data_parallel_size();
            for dp_rank in dp_start..dp_end {
                if predicate(WorkerWithDpRank::new(worker_id, dp_rank), config) {
                    return true;
                }
            }
        }

        false
    }

    pub fn for_each_eligible_worker_rank<C, F>(&self, workers: &HashMap<WorkerId, C>, mut visit: F)
    where
        C: WorkerConfigLike,
        F: FnMut(WorkerWithDpRank, &C),
    {
        self.any_eligible_worker_rank(workers, |worker, config| {
            visit(worker, config);
            false
        });
    }

    #[inline]
    pub(crate) fn validate_pinned_worker_allowed(&self) -> Result<(), KvSchedulerError> {
        let Some(pinned_worker) = self.pinned_worker else {
            return Ok(());
        };

        if self.caller_allows_worker_id(pinned_worker.worker_id) {
            return Ok(());
        }

        Err(KvSchedulerError::PinnedWorkerNotAllowed {
            worker_id: pinned_worker.worker_id,
        })
    }

    #[inline]
    pub(crate) fn bypasses_capacity_check(&self) -> bool {
        self.pinned_worker.is_none() && self.allowed_worker_ids.is_some()
    }
}

#[derive(Clone, Copy)]
pub struct SchedulingContext<'a, C> {
    request: &'a SchedulingRequest,
    eligibility: RoutingEligibility<'a>,
    workers: &'a HashMap<WorkerId, C>,
}

impl<'a, C: WorkerConfigLike> SchedulingContext<'a, C> {
    pub fn new(request: &'a SchedulingRequest, workers: &'a HashMap<WorkerId, C>) -> Self {
        Self {
            request,
            eligibility: request.eligibility(),
            workers,
        }
    }

    pub fn request(&self) -> &'a SchedulingRequest {
        self.request
    }

    pub fn best_effective_prefill_tokens(&self) -> usize {
        let cached_tokens = match self.eligibility.pinned_worker() {
            Some(worker) => self.request.effective_cached_tokens_for(worker),
            None => self
                .request
                .effective_cached_tokens
                .iter()
                .filter(|(worker, _)| {
                    self.workers.get(&worker.worker_id).is_some_and(|config| {
                        self.eligibility.allows_worker(worker.worker_id, config)
                    })
                })
                .map(|(_, cached_tokens)| *cached_tokens)
                .max()
                .unwrap_or(0),
        };

        self.request.isl_tokens.saturating_sub(cached_tokens)
    }
}

impl SchedulingRequest {
    #[inline]
    pub fn eligibility(&self) -> RoutingEligibility<'_> {
        self.eligibility_with_overloaded(None)
    }

    #[inline]
    pub fn eligibility_with_overloaded<'a>(
        &'a self,
        overloaded_worker_ids: Option<&'a HashSet<WorkerId>>,
    ) -> RoutingEligibility<'a> {
        RoutingEligibility::new(
            self.allowed_worker_ids.as_ref(),
            overloaded_worker_ids,
            self.pinned_worker,
            &self.routing_constraints,
        )
    }

    pub(crate) fn prefill_token_deltas(&self) -> PrefillTokenDeltas {
        if !self.track_prefill_tokens {
            return PrefillTokenDeltas::none();
        }

        let by_worker = self
            .effective_cached_tokens
            .iter()
            .map(|(worker, cached_tokens)| {
                let delta = self
                    .isl_tokens
                    .checked_sub(*cached_tokens)
                    .unwrap_or_else(|| {
                        tracing::error!(
                            "prefill_tokens < 0 with ISL {} < cached_tokens {}, returning 0",
                            self.isl_tokens,
                            cached_tokens
                        );
                        0
                    });
                (*worker, delta)
            })
            .collect();

        PrefillTokenDeltas::new(self.isl_tokens, by_worker)
    }

    pub(crate) fn effective_cached_tokens_for(&self, worker: WorkerWithDpRank) -> usize {
        self.effective_cached_tokens
            .get(&worker)
            .copied()
            .unwrap_or(0)
    }

    pub(crate) fn effective_overlap_blocks_for(&self, worker: WorkerWithDpRank) -> f64 {
        self.effective_overlap_blocks
            .get(&worker)
            .copied()
            .unwrap_or(0.0)
    }

    #[cfg(test)]
    pub(crate) fn prefill_tokens_for(&self, worker: WorkerWithDpRank) -> usize {
        let default_prefill_tokens = if self.track_prefill_tokens {
            self.isl_tokens
        } else {
            0
        };
        self.prefill_tokens
            .get(&worker)
            .copied()
            .unwrap_or(default_prefill_tokens)
    }

    /// Prompt-side load before applying this request's cache-hit credits.
    pub(crate) fn raw_prefill_tokens_for(&self, worker: WorkerWithDpRank) -> usize {
        if !self.track_prefill_tokens {
            return 0;
        }

        match self.prefill_tokens.get(&worker).copied() {
            Some(projected_tokens) => {
                projected_tokens.saturating_add(self.effective_cached_tokens_for(worker))
            }
            None => self.isl_tokens,
        }
    }

    pub(crate) fn request_blocks(&self, block_size: u32) -> u64 {
        self.isl_tokens.div_ceil(block_size as usize) as u64
    }

    pub fn respond(&mut self, result: Result<SchedulingResponse, KvSchedulerError>) {
        let Some(tx) = self.resp_tx.take() else {
            tracing::error!("respond called multiple times on same request");
            return;
        };
        if tx.send(result).is_err() {
            tracing::error!("failed to send response to requestor");
        }
    }
}

fn worker_config_for_rank<C: WorkerConfigLike>(
    workers: &HashMap<WorkerId, C>,
    worker: WorkerWithDpRank,
) -> Result<&C, WorkerEligibilityError> {
    let Some(config) = workers.get(&worker.worker_id) else {
        return Err(WorkerEligibilityError::WorkerUnavailable {
            worker_id: worker.worker_id,
        });
    };
    let dp_start_rank = config.data_parallel_start_rank();
    let dp_end_rank = dp_start_rank + config.data_parallel_size();
    if !(dp_start_rank..dp_end_rank).contains(&worker.dp_rank) {
        return Err(WorkerEligibilityError::DpRankUnavailable {
            worker_id: worker.worker_id,
            dp_rank: worker.dp_rank,
            start: dp_start_rank,
            end: dp_end_rank,
        });
    }

    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct TestWorkerConfig {
        dp_start: DpRank,
        dp_size: DpRank,
        taints: HashSet<String>,
    }

    impl Default for TestWorkerConfig {
        fn default() -> Self {
            Self {
                dp_start: 0,
                dp_size: 1,
                taints: HashSet::new(),
            }
        }
    }

    impl WorkerConfigLike for TestWorkerConfig {
        fn data_parallel_start_rank(&self) -> u32 {
            self.dp_start
        }

        fn data_parallel_size(&self) -> u32 {
            self.dp_size
        }

        fn max_num_batched_tokens(&self) -> Option<u64> {
            None
        }

        fn total_kv_blocks(&self) -> Option<u64> {
            None
        }

        fn taints(&self) -> &HashSet<String> {
            &self.taints
        }
    }

    fn workers() -> HashMap<WorkerId, TestWorkerConfig> {
        HashMap::from([(
            7,
            TestWorkerConfig {
                dp_start: 2,
                dp_size: 3,
                taints: HashSet::from(["zone-a".to_string()]),
            },
        )])
    }

    #[test]
    fn routing_eligibility_accepts_allowed_rank_matching_constraints() {
        let workers = workers();
        let allowed = HashSet::from([7]);
        let constraints = RoutingConstraints {
            required_taints: HashSet::from(["zone-a".to_string()]),
            preferred_taints: HashMap::new(),
        };
        let eligibility = RoutingEligibility::new(Some(&allowed), None, None, &constraints);

        let result = eligibility.validate_worker_rank(&workers, WorkerWithDpRank::new(7, 3));

        assert!(result.is_ok());
    }

    #[test]
    fn routing_eligibility_rejects_disallowed_worker() {
        let workers = workers();
        let allowed = HashSet::from([8]);
        let constraints = RoutingConstraints::default();
        let eligibility = RoutingEligibility::new(Some(&allowed), None, None, &constraints);

        let result = eligibility.validate_worker_rank(&workers, WorkerWithDpRank::new(7, 3));

        assert_eq!(
            result.err(),
            Some(WorkerEligibilityError::WorkerNotAllowed { worker_id: 7 })
        );
    }

    #[test]
    fn routing_eligibility_rejects_overloaded_worker() {
        let workers = workers();
        let overloaded = HashSet::from([7]);
        let constraints = RoutingConstraints::default();
        let eligibility = RoutingEligibility::new(None, Some(&overloaded), None, &constraints);

        let result = eligibility.validate_worker_rank(&workers, WorkerWithDpRank::new(7, 3));

        assert_eq!(
            result.err(),
            Some(WorkerEligibilityError::WorkerOverloaded { worker_id: 7 })
        );
    }

    #[test]
    fn routing_eligibility_prefers_allow_list_error_before_overload() {
        let workers = workers();
        let allowed = HashSet::from([8]);
        let overloaded = HashSet::from([7]);
        let constraints = RoutingConstraints::default();
        let eligibility =
            RoutingEligibility::new(Some(&allowed), Some(&overloaded), None, &constraints);

        let result = eligibility.validate_worker_rank(&workers, WorkerWithDpRank::new(7, 3));

        assert_eq!(
            result.err(),
            Some(WorkerEligibilityError::WorkerNotAllowed { worker_id: 7 })
        );
    }

    #[test]
    fn routing_eligibility_rejects_out_of_range_dp_rank() {
        let workers = workers();
        let constraints = RoutingConstraints::default();
        let eligibility = RoutingEligibility::new(None, None, None, &constraints);

        let result = eligibility.validate_worker_rank(&workers, WorkerWithDpRank::new(7, 5));

        assert_eq!(
            result.err(),
            Some(WorkerEligibilityError::DpRankUnavailable {
                worker_id: 7,
                dp_rank: 5,
                start: 2,
                end: 5,
            })
        );
    }

    #[test]
    fn routing_eligibility_rejects_unsatisfied_required_taints() {
        let workers = workers();
        let constraints = RoutingConstraints {
            required_taints: HashSet::from(["zone-b".to_string()]),
            preferred_taints: HashMap::new(),
        };
        let eligibility = RoutingEligibility::new(None, None, None, &constraints);

        let result = eligibility.validate_worker_rank(&workers, WorkerWithDpRank::new(7, 3));

        assert_eq!(
            result.err(),
            Some(WorkerEligibilityError::RoutingConstraintsUnsatisfied { worker_id: 7 })
        );
    }

    #[test]
    fn routing_eligibility_applies_allowed_overloaded_and_taints() {
        let allowed_worker_ids = HashSet::from([1, 2]);
        let overloaded_worker_ids = HashSet::from([2]);
        let routing_constraints = RoutingConstraints {
            required_taints: HashSet::from(["mdc-a".to_string()]),
            preferred_taints: HashMap::new(),
        };
        let eligibility = RoutingEligibility::new(
            Some(&allowed_worker_ids),
            Some(&overloaded_worker_ids),
            None,
            &routing_constraints,
        );

        let compatible = TestWorkerConfig {
            taints: HashSet::from(["mdc-a".to_string()]),
            ..Default::default()
        };
        let incompatible = TestWorkerConfig {
            taints: HashSet::from(["mdc-b".to_string()]),
            ..Default::default()
        };

        assert!(eligibility.allows_worker(1, &compatible));
        assert!(!eligibility.allows_worker(2, &compatible));
        assert!(!eligibility.allows_worker(3, &compatible));
        assert!(!eligibility.allows_worker(1, &incompatible));
        assert!(eligibility.has_eligible_worker([(1, &compatible), (2, &compatible)]));
        assert!(!eligibility.has_eligible_worker([(2, &compatible)]));
        assert!(eligibility.has_eligible_worker_ignoring_overload([(2, &compatible)]));
        assert!(
            eligibility.has_eligible_worker_ignoring_overload([(1, &compatible), (2, &compatible)])
        );
    }

    #[test]
    fn routing_eligibility_expands_all_eligible_dp_ranks() {
        let workers = HashMap::from([
            (
                7,
                TestWorkerConfig {
                    dp_start: 2,
                    dp_size: 3,
                    taints: HashSet::from(["zone-a".to_string()]),
                },
            ),
            (
                8,
                TestWorkerConfig {
                    dp_start: 0,
                    dp_size: 2,
                    taints: HashSet::from(["zone-b".to_string()]),
                },
            ),
        ]);
        let allowed = HashSet::from([7]);
        let constraints = RoutingConstraints {
            required_taints: HashSet::from(["zone-a".to_string()]),
            preferred_taints: HashMap::new(),
        };
        let eligibility = RoutingEligibility::new(Some(&allowed), None, None, &constraints);
        let mut ranks = Vec::new();

        eligibility.for_each_eligible_worker_rank(&workers, |worker, _| ranks.push(worker));

        assert_eq!(
            ranks,
            vec![
                WorkerWithDpRank::new(7, 2),
                WorkerWithDpRank::new(7, 3),
                WorkerWithDpRank::new(7, 4),
            ]
        );
    }

    #[test]
    fn routing_eligibility_rank_expansion_skips_overloaded_workers() {
        let workers = HashMap::from([
            (
                7,
                TestWorkerConfig {
                    dp_start: 2,
                    dp_size: 2,
                    taints: HashSet::from(["zone-a".to_string()]),
                },
            ),
            (
                8,
                TestWorkerConfig {
                    dp_start: 4,
                    dp_size: 2,
                    taints: HashSet::from(["zone-a".to_string()]),
                },
            ),
        ]);
        let overloaded = HashSet::from([7]);
        let constraints = RoutingConstraints {
            required_taints: HashSet::from(["zone-a".to_string()]),
            preferred_taints: HashMap::new(),
        };
        let eligibility = RoutingEligibility::new(None, Some(&overloaded), None, &constraints);
        let mut ranks = Vec::new();

        eligibility.for_each_eligible_worker_rank(&workers, |worker, _| ranks.push(worker));
        ranks.sort_by_key(|worker| (worker.worker_id, worker.dp_rank));

        assert_eq!(
            ranks,
            vec![WorkerWithDpRank::new(8, 4), WorkerWithDpRank::new(8, 5)]
        );
    }

    #[test]
    fn routing_eligibility_pinned_expansion_yields_exact_rank() {
        let workers = workers();
        let constraints = RoutingConstraints::default();
        let eligibility =
            RoutingEligibility::new(None, None, Some(WorkerWithDpRank::new(7, 3)), &constraints);
        let mut ranks = Vec::new();

        eligibility.for_each_eligible_worker_rank(&workers, |worker, _| ranks.push(worker));

        assert_eq!(ranks, vec![WorkerWithDpRank::new(7, 3)]);
    }

    #[test]
    fn routing_eligibility_pinned_expansion_rejects_bad_rank() {
        let workers = workers();
        let constraints = RoutingConstraints::default();
        let eligibility =
            RoutingEligibility::new(None, None, Some(WorkerWithDpRank::new(7, 5)), &constraints);
        let mut ranks = Vec::new();

        eligibility.for_each_eligible_worker_rank(&workers, |worker, _| ranks.push(worker));

        assert!(ranks.is_empty());
    }
}
