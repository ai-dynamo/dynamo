// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! First-party worker-selection policies linked by Dynamo's Python router extension.
//!
//! Standalone EPP and Rust hosts link and register this crate explicitly.
//!
//! This crate owns policy algorithms and policy-local state. The selection host retains discovery,
//! eligibility, and request accounting. The optional custom catalog uses the same registry API,
//! but never needs to replace these stock policies.

use std::sync::Arc;

use dynamo_kv_router::{
    KvRouterConfig,
    scheduling::WorkerSelectionPolicyError,
    selector::{
        CacheFreeCandidateTable, CacheFreePolicyDecision, CacheFreeRequestContext,
        CacheFreeWorkerPicker, WorkerCandidate, WorkerInputView, WorkerInputs, WorkerPicker,
        WorkerScorer, WorkerSelectionContext, WorkerSelectionPolicy, WorkerSelectionRequirements,
    },
    services::policy_registry::{
        WorkerSelectionPolicyProvider, WorkerSelectionPolicyRegistry,
        WorkerSelectionPolicyRegistryError,
    },
};
use dynamo_runtime::fast_picker::{FastPicker, reservoir_least_index_by};

/// Register the stock policy types linked by Dynamo's Python router extension.
///
/// Each resolved policy instance creates one picker for its routing partition, so mutable state
/// such as the round-robin cursor is not shared between models or worker roles.
pub fn register(
    registry: &mut WorkerSelectionPolicyRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    for (name, policy) in [
        ("round_robin", FirstPartyRoutingPolicy::RoundRobin),
        ("random", FirstPartyRoutingPolicy::Random),
        (
            "power_of_two_choices",
            FirstPartyRoutingPolicy::PowerOfTwoChoices,
        ),
        ("least_loaded", FirstPartyRoutingPolicy::LeastLoaded),
    ] {
        registry.register(name, policy_provider(policy))?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FirstPartyRoutingPolicy {
    RoundRobin,
    Random,
    PowerOfTwoChoices,
    LeastLoaded,
}

fn policy_provider(policy: FirstPartyRoutingPolicy) -> WorkerSelectionPolicyProvider {
    Arc::new(move |parameters| {
        let _: EmptyParameters = parameters.deserialize()?;
        Ok(Arc::new(move |config, worker_type, _partition| {
            worker_selection_policy(config.clone(), worker_type.as_str(), policy)
        }))
    })
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct EmptyParameters {}

fn worker_selection_policy(
    config: KvRouterConfig,
    worker_type: &'static str,
    policy: FirstPartyRoutingPolicy,
) -> WorkerSelectionPolicy {
    WorkerSelectionPolicy::new(
        config,
        worker_type,
        vec![Box::new(FirstPartyWorkerScorer { policy })],
        Box::new(FirstPartyWorkerPicker {
            policy,
            last_round_robin_worker: None,
        }),
    )
    .with_cache_free_picker(Box::new(FirstPartyCacheFreePicker::new(policy)))
}

/// Direct cache-free implementation for the first-party simple policies.
///
/// This intentionally bypasses generic filter/score/pick materialization: round-robin and random
/// select an index directly, P2C reads only its two sampled load counters, and least-loaded uses
/// the host-maintained minimum. None of these algorithms may walk the candidate table.
struct FirstPartyCacheFreePicker {
    policy: FirstPartyRoutingPolicy,
    fast_picker: FastPicker,
}

impl FirstPartyCacheFreePicker {
    const fn new(policy: FirstPartyRoutingPolicy) -> Self {
        Self {
            policy,
            fast_picker: FastPicker::new(),
        }
    }

    const fn requirements_for(policy: FirstPartyRoutingPolicy) -> WorkerSelectionRequirements {
        match policy {
            FirstPartyRoutingPolicy::RoundRobin | FirstPartyRoutingPolicy::Random => {
                WorkerSelectionRequirements::STATIC
            }
            FirstPartyRoutingPolicy::PowerOfTwoChoices | FirstPartyRoutingPolicy::LeastLoaded => {
                WorkerSelectionRequirements::ACTIVE_REQUEST_LOAD
            }
        }
    }

    fn checked_index(
        index: Option<usize>,
        candidate_count: usize,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        index
            .filter(|&index| index < candidate_count)
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible workers"))
    }
}

impl CacheFreeWorkerPicker for FirstPartyCacheFreePicker {
    fn requirements(&self) -> WorkerSelectionRequirements {
        Self::requirements_for(self.policy)
    }

    fn select(
        &self,
        request: &CacheFreeRequestContext<'_>,
        candidate_table: &dyn CacheFreeCandidateTable,
    ) -> Result<CacheFreePolicyDecision, WorkerSelectionPolicyError> {
        let candidate_count = candidate_table.len();
        if candidate_count == 0 {
            return Err(WorkerSelectionPolicyError::failed("no eligible workers"));
        }

        let index = match self.policy {
            FirstPartyRoutingPolicy::RoundRobin => self
                .fast_picker
                .round_robin_index(candidate_count, !request.is_advisory())
                .expect("candidate count was checked above"),
            FirstPartyRoutingPolicy::Random => FastPicker::random_index(candidate_count)
                .expect("candidate count was checked above"),
            FirstPartyRoutingPolicy::PowerOfTwoChoices => {
                FastPicker::power_of_two_choices_index(candidate_count, |index| {
                    candidate_table.active_requests(index) as u64
                })
                .expect("candidate count was checked above")
            }
            FirstPartyRoutingPolicy::LeastLoaded => {
                Self::checked_index(candidate_table.least_loaded_index(), candidate_count)?
            }
        };

        Ok(CacheFreePolicyDecision::unfiltered(index))
    }
}

struct FirstPartyWorkerScorer {
    policy: FirstPartyRoutingPolicy,
}

impl WorkerScorer for FirstPartyWorkerScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        match self.policy {
            FirstPartyRoutingPolicy::RoundRobin | FirstPartyRoutingPolicy::Random => {
                WorkerInputs::NONE
            }
            FirstPartyRoutingPolicy::PowerOfTwoChoices | FirstPartyRoutingPolicy::LeastLoaded => {
                WorkerInputs::ACTIVE_REQUEST_LOAD
            }
        }
    }

    fn score(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        candidate: &WorkerCandidate,
    ) -> Result<f64, WorkerSelectionPolicyError> {
        Ok(candidate.load().map_or(0, |load| load.active_requests()) as f64)
    }
}

struct FirstPartyWorkerPicker {
    policy: FirstPartyRoutingPolicy,
    // Candidate row order is unspecified. Holding an identity rather than a row number gives
    // round-robin deterministic behavior across HashMap iteration order and worker churn.
    last_round_robin_worker: Option<dynamo_kv_router::protocols::WorkerWithDpRank>,
}

impl FirstPartyWorkerPicker {
    fn round_robin_row(&mut self, input: WorkerInputView<'_>, advisory: bool) -> usize {
        let mut first = None;
        let mut successor = None;
        for (row, candidate) in input.candidates().iter().enumerate() {
            let worker = candidate.worker();
            if first.is_none_or(|current: usize| worker < input.candidates()[current].worker()) {
                first = Some(row);
            }
            if self
                .last_round_robin_worker
                .is_some_and(|last| worker > last)
                && successor
                    .is_none_or(|current: usize| worker < input.candidates()[current].worker())
            {
                successor = Some(row);
            }
        }
        let row = successor
            .or(first)
            .expect("picker only runs with candidates");
        if !advisory {
            self.last_round_robin_worker = Some(input.candidates()[row].worker());
        }
        row
    }

    fn p2c_row(&self, input: WorkerInputView<'_>) -> usize {
        let candidates = input.candidates();
        FastPicker::power_of_two_choices_index_by(candidates.len(), |first, second| {
            candidates[first].cost() <= candidates[second].cost()
        })
        .expect("picker only runs with candidates")
    }

    fn least_loaded_row(&self, input: WorkerInputView<'_>) -> usize {
        let candidates = input.candidates();
        reservoir_least_index_by(
            candidates.len(),
            |left, right| candidates[left].cost().total_cmp(&candidates[right].cost()),
            |upper| fastrand::usize(..upper),
        )
        .expect("picker only runs with candidates")
    }
}

impl WorkerPicker for FirstPartyWorkerPicker {
    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        if input.candidates().is_empty() {
            return Err(WorkerSelectionPolicyError::failed("no eligible workers"));
        }
        Ok(match self.policy {
            FirstPartyRoutingPolicy::RoundRobin => {
                self.round_robin_row(input, context.is_advisory())
            }
            FirstPartyRoutingPolicy::Random => FastPicker::random_index(input.candidates().len())
                .expect("picker only runs with candidates"),
            FirstPartyRoutingPolicy::PowerOfTwoChoices => self.p2c_row(input),
            FirstPartyRoutingPolicy::LeastLoaded => self.least_loaded_row(input),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use dynamo_kv_router::{
        protocols::{WorkerConfigLike, WorkerWithDpRank},
        scheduling::{OverlapSignals, ScheduleMode, SchedulingRequest},
        selector::{CacheFreeCandidateTable, CacheFreeRequestContext, WorkerSelector},
    };

    #[derive(Default)]
    struct WorkerConfig {
        taints: HashSet<String>,
    }

    impl WorkerConfigLike for WorkerConfig {
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

    fn workers() -> HashMap<u64, WorkerConfig> {
        HashMap::from([(7, WorkerConfig::default()), (11, WorkerConfig::default())])
    }

    fn request(mode: ScheduleMode) -> SchedulingRequest {
        SchedulingRequest {
            mode,
            token_seq: None,
            isl_tokens: 16,
            lora_name: None,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: Default::default(),
            router_config_override: None,
            track_prefill_tokens: false,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            overlap: OverlapSignals::default(),
            router_hint_candidates: None,
            retain_router_hint_chain: false,
            shared_cache_hits: None,
            worker_loads: Default::default(),
            resp_tx: None,
        }
    }

    struct CountingCacheFreeRows {
        rows: Vec<(WorkerWithDpRank, usize)>,
        least_loaded_index: Option<usize>,
        worker_reads: AtomicUsize,
        active_request_reads: AtomicUsize,
        least_loaded_reads: AtomicUsize,
    }

    impl CountingCacheFreeRows {
        fn with_workers(count: usize, least_loaded_index: Option<usize>) -> Self {
            Self {
                rows: (0..count)
                    .map(|index| (WorkerWithDpRank::from_worker_id(index as u64), index))
                    .collect(),
                least_loaded_index,
                worker_reads: AtomicUsize::new(0),
                active_request_reads: AtomicUsize::new(0),
                least_loaded_reads: AtomicUsize::new(0),
            }
        }

        fn reads(&self) -> (usize, usize, usize) {
            (
                self.worker_reads.load(Ordering::Relaxed),
                self.active_request_reads.load(Ordering::Relaxed),
                self.least_loaded_reads.load(Ordering::Relaxed),
            )
        }
    }

    impl CacheFreeCandidateTable for CountingCacheFreeRows {
        fn len(&self) -> usize {
            self.rows.len()
        }

        fn worker(&self, index: usize) -> WorkerWithDpRank {
            self.worker_reads.fetch_add(1, Ordering::Relaxed);
            self.rows[index].0
        }

        fn active_requests(&self, index: usize) -> usize {
            self.active_request_reads.fetch_add(1, Ordering::Relaxed);
            self.rows[index].1
        }

        fn least_loaded_index(&self) -> Option<usize> {
            self.least_loaded_reads.fetch_add(1, Ordering::Relaxed);
            self.least_loaded_index
        }
    }

    fn cache_free_policy(
        policy: FirstPartyRoutingPolicy,
    ) -> dynamo_kv_router::CacheFreeWorkerSelectionPolicy {
        worker_selection_policy(KvRouterConfig::default(), "decode", policy)
            .into_cache_free()
            .unwrap()
    }

    #[test]
    fn round_robin_keeps_advisory_queries_and_partitions_independent() {
        let first = worker_selection_policy(
            KvRouterConfig::default(),
            "decode",
            FirstPartyRoutingPolicy::RoundRobin,
        );
        let second = worker_selection_policy(
            KvRouterConfig::default(),
            "decode",
            FirstPartyRoutingPolicy::RoundRobin,
        );
        let workers = workers();
        let advisory = request(ScheduleMode::QueryOnly {
            request_id: Some("advisory".to_string()),
        });

        assert_eq!(
            first
                .select_worker(&workers, &advisory, advisory.eligibility(), 16)
                .unwrap()
                .worker,
            WorkerWithDpRank::from_worker_id(7)
        );
        assert_eq!(
            first
                .select_worker(&workers, &advisory, advisory.eligibility(), 16)
                .unwrap()
                .worker,
            WorkerWithDpRank::from_worker_id(7)
        );

        let committed = request(ScheduleMode::Tracked {
            request_id: "committed".to_string(),
        });
        assert_eq!(
            first
                .select_worker(&workers, &committed, committed.eligibility(), 16)
                .unwrap()
                .worker,
            WorkerWithDpRank::from_worker_id(7)
        );
        assert_eq!(
            second
                .select_worker(&workers, &committed, committed.eligibility(), 16)
                .unwrap()
                .worker,
            WorkerWithDpRank::from_worker_id(7)
        );
        assert_eq!(
            first
                .select_worker(&workers, &committed, committed.eligibility(), 16)
                .unwrap()
                .worker,
            WorkerWithDpRank::from_worker_id(11)
        );
    }

    #[test]
    fn static_and_load_aware_policies_declare_different_requirements() {
        let static_policy = worker_selection_policy(
            KvRouterConfig::default(),
            "decode",
            FirstPartyRoutingPolicy::RoundRobin,
        );
        let load_aware_policy = worker_selection_policy(
            KvRouterConfig::default(),
            "decode",
            FirstPartyRoutingPolicy::LeastLoaded,
        );

        assert!(!static_policy.requirements().needs_cache_index());
        assert!(!static_policy.requirements().needs_active_request_load());
        assert!(!load_aware_policy.requirements().needs_cache_index());
        assert!(load_aware_policy.requirements().needs_active_request_load());
    }

    #[test]
    fn cache_free_simple_policies_do_not_scan_large_worker_sets() {
        const WORKERS: usize = 1024;
        let request = CacheFreeRequestContext::new("request", 16, false);

        let rows = CountingCacheFreeRows::with_workers(WORKERS, Some(739));
        let round_robin = cache_free_policy(FirstPartyRoutingPolicy::RoundRobin);
        assert_eq!(round_robin.select(&request, &rows).unwrap().index, 0);
        assert_eq!(round_robin.select(&request, &rows).unwrap().index, 1);
        assert_eq!(rows.reads(), (0, 0, 0));

        let rows = CountingCacheFreeRows::with_workers(WORKERS, Some(739));
        let random = cache_free_policy(FirstPartyRoutingPolicy::Random);
        assert!(random.select(&request, &rows).unwrap().index < WORKERS);
        assert_eq!(rows.reads(), (0, 0, 0));

        let rows = CountingCacheFreeRows::with_workers(WORKERS, Some(739));
        let p2c = cache_free_policy(FirstPartyRoutingPolicy::PowerOfTwoChoices);
        assert!(p2c.select(&request, &rows).unwrap().index < WORKERS);
        assert_eq!(rows.reads(), (0, 2, 0));

        let rows = CountingCacheFreeRows::with_workers(WORKERS, Some(739));
        let least_loaded = cache_free_policy(FirstPartyRoutingPolicy::LeastLoaded);
        assert_eq!(least_loaded.select(&request, &rows).unwrap().index, 739);
        assert_eq!(rows.reads(), (0, 0, 1));
    }
}
