// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Always-linked, first-party worker-selection policies.
//!
//! This crate owns policy algorithms and policy-local state. The selection host retains discovery,
//! eligibility, and request accounting. The optional custom catalog uses the same registry API,
//! but never needs to replace these stock policies.

use std::sync::Arc;

use dynamo_kv_router::{
    KvRouterConfig,
    scheduling::WorkerSelectionPolicyError,
    selector::{
        WorkerCandidate, WorkerInputView, WorkerInputs, WorkerPicker, WorkerScorer,
        WorkerSelectionContext, WorkerSelectionPolicy,
    },
    services::policy_registry::{
        WorkerSelectionPolicyProvider, WorkerSelectionPolicyRegistry,
        WorkerSelectionPolicyRegistryError,
    },
};

/// Register the worker-selection policies shipped in every Dynamo artifact.
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
        let first = fastrand::usize(..candidates.len());
        if candidates.len() == 1 {
            return first;
        }
        let second = (first + 1 + fastrand::usize(..candidates.len() - 1)) % candidates.len();
        if candidates[first].cost() <= candidates[second].cost() {
            first
        } else {
            second
        }
    }

    fn least_loaded_row(&self, input: WorkerInputView<'_>) -> usize {
        let mut best = 0;
        let mut ties = 1usize;
        for row in 1..input.candidates().len() {
            match input.candidates()[row]
                .cost()
                .total_cmp(&input.candidates()[best].cost())
            {
                std::cmp::Ordering::Less => {
                    best = row;
                    ties = 1;
                }
                std::cmp::Ordering::Equal => {
                    ties += 1;
                    if fastrand::usize(..ties) == 0 {
                        best = row;
                    }
                }
                std::cmp::Ordering::Greater => {}
            }
        }
        best
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
            FirstPartyRoutingPolicy::Random => fastrand::usize(..input.candidates().len()),
            FirstPartyRoutingPolicy::PowerOfTwoChoices => self.p2c_row(input),
            FirstPartyRoutingPolicy::LeastLoaded => self.least_loaded_row(input),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use super::*;
    use dynamo_kv_router::{
        protocols::{WorkerConfigLike, WorkerWithDpRank},
        scheduling::{OverlapSignals, ScheduleMode, SchedulingRequest},
        selector::WorkerSelector,
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
}
