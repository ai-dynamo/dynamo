// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_kv_router::WorkerType;
use dynamo_kv_router::plugins::RouterPluginRegistry;
use dynamo_kv_router::plugins::worker_selection::{
    WorkerCandidates, WorkerInputView, WorkerInputs, WorkerPicker, WorkerScorer,
    WorkerSelectionContext, WorkerSelectionPolicy, WorkerSelectionPolicyError,
    WorkerSelectionPolicyFactory, WorkerSelectionPolicyParameters,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistryError,
};

use super::THUNDERAGENT_CLASSIFIER_TYPE;

struct ThunderAgentScorer;

impl WorkerScorer for ThunderAgentScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD
    }

    fn score(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError> {
        let block_size = f64::from(context.block_size());
        for (candidate, cost) in candidates.iter().zip(costs) {
            let load = candidate
                .load()
                .ok_or_else(|| WorkerSelectionPolicyError::failed("worker load unavailable"))?;
            *cost = load.active_prefill_tokens() as f64 + load.decode_cost_blocks() * block_size;
        }
        Ok(())
    }
}

struct ThunderAgentPicker;

impl WorkerPicker for ThunderAgentPicker {
    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let candidates = input.candidates();
        if let Some(target) = context.affinity_target()
            && let Some(row) = candidates.iter().position(|candidate| {
                candidate.worker().worker_id == target.worker_id
                    && target
                        .dp_rank
                        .is_none_or(|dp_rank| candidate.worker().dp_rank == dp_rank)
            })
        {
            return Ok(row);
        }

        candidates
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| {
                left.cost()
                    .total_cmp(&right.cost())
                    .then_with(|| left.worker().cmp(&right.worker()))
            })
            .map(|(row, _)| row)
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ThunderAgentSelectionParameters {}

fn worker_selection_provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let _: ThunderAgentSelectionParameters = parameters.deserialize()?;
    Ok(Arc::new(
        move |router, worker_type: WorkerType, _partition| {
            WorkerSelectionPolicy::new(
                router.clone(),
                worker_type.as_str(),
                vec![Box::new(ThunderAgentScorer)],
                Box::new(ThunderAgentPicker),
            )
        },
    ))
}

pub(crate) fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register_worker_selection(
        THUNDERAGENT_CLASSIFIER_TYPE,
        Arc::new(worker_selection_provider),
    )
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use dynamo_kv_router::protocols::{RoutingConstraints, WorkerConfigLike, WorkerWithDpRank};
    use dynamo_kv_router::scheduling::{OverlapSignals, ScheduleMode, SchedulingRequest};
    use dynamo_kv_router::{
        KvRouterConfig, WorkerLoadProjection, WorkerSelectionInput, WorkerSelectionPolicy,
        WorkerSelector,
    };

    use super::*;

    struct TestWorker;

    impl WorkerConfigLike for TestWorker {
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
    }

    fn request(target: WorkerWithDpRank) -> SchedulingRequest {
        SchedulingRequest {
            mode: ScheduleMode::QueryOnly {
                request_id: Some("request".into()),
            },
            token_seq: None,
            isl_tokens: 16,
            lora_name: None,
            expected_output_tokens: None,
            affinity_target: Some(target.into()),
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: RoutingConstraints::default(),
            router_config_override: None,
            track_prefill_tokens: true,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            overlap: OverlapSignals::default(),
            kv_transfer_candidates: None,
            retain_kv_transfer_chain: false,
            shared_cache_hits: None,
            worker_loads: Default::default(),
            resp_tx: None,
        }
    }

    fn policy() -> WorkerSelectionPolicy {
        WorkerSelectionPolicy::new(
            KvRouterConfig::default(),
            "test",
            vec![Box::new(ThunderAgentScorer)],
            Box::new(ThunderAgentPicker),
        )
    }

    #[test]
    fn honors_the_classifier_target_when_eligible() {
        let worker_1 = WorkerWithDpRank::new(1, 0);
        let worker_2 = WorkerWithDpRank::new(2, 0);
        let workers = HashMap::from([(1, TestWorker), (2, TestWorker)]);
        let request = request(worker_2);

        let selected = policy()
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                request.eligibility(),
                16,
            ))
            .unwrap();

        assert_eq!(selected.worker, worker_2);
        assert_ne!(selected.worker, worker_1);
    }

    #[test]
    fn falls_back_when_the_classifier_target_is_unavailable() {
        let unavailable = WorkerWithDpRank::new(2, 0);
        let worker = WorkerWithDpRank::new(4, 0);
        let workers = HashMap::from([(1, TestWorker), (3, TestWorker), (4, TestWorker)]);
        let mut request = request(unavailable);
        // Scores at block size 16 are 161, 100, and 84; both load terms affect selection.
        for (id, prefill_tokens, decode_blocks) in [(1, 1, 10), (3, 100, 0), (4, 20, 4)] {
            request.worker_loads.insert(
                WorkerWithDpRank::new(id, 0),
                WorkerLoadProjection {
                    active_prefill_tokens: prefill_tokens,
                    active_decode_blocks: decode_blocks,
                    ..Default::default()
                },
            );
        }

        let selected = policy()
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                request.eligibility(),
                16,
            ))
            .unwrap();

        assert_eq!(selected.worker, worker);
    }
}
