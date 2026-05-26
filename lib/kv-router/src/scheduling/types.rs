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

    #[allow(dead_code)]
    pub(crate) fn is_worker_allowed(&self, worker_id: WorkerId) -> bool {
        self.allowed_worker_ids
            .as_ref()
            .is_none_or(|ids| ids.contains(&worker_id))
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

pub fn pinned_worker_config<C: WorkerConfigLike>(
    workers: &HashMap<WorkerId, C>,
    worker: WorkerWithDpRank,
) -> Result<&C, KvSchedulerError> {
    let Some(config) = workers.get(&worker.worker_id) else {
        return Err(KvSchedulerError::NoEndpoints);
    };
    let dp_start_rank = config.data_parallel_start_rank();
    let dp_end_rank = dp_start_rank + config.data_parallel_size();
    if !(dp_start_rank..dp_end_rank).contains(&worker.dp_rank) {
        return Err(KvSchedulerError::NoEndpoints);
    }

    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct TestWorkerConfig {
        taints: HashSet<String>,
    }

    impl WorkerConfigLike for TestWorkerConfig {
        fn data_parallel_start_rank(&self) -> u32 {
            0
        }

        fn data_parallel_size(&self) -> u32 {
            1
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
        };
        let incompatible = TestWorkerConfig {
            taints: HashSet::from(["mdc-b".to_string()]),
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
}
