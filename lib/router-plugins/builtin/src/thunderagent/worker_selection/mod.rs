// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_kv_router::plugins::worker_selection::*;

use super::THUNDERAGENT_CLASSIFIER_TYPE;

struct ThunderAgentScorer;

impl WorkerScorer for ThunderAgentScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD
    }

    fn score(
        &self,
        request: &WorkerSelectionRequest<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError> {
        let block_size = f64::from(request.block_size());
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
        &self,
        request: &WorkerSelectionRequest<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let candidates = input.candidates();
        if let Some(target) = request.affinity_target()
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
    Ok(Arc::new(move |context| {
        WorkerSelectionPolicy::new(
            context.worker_type().as_str(),
            vec![Box::new(ThunderAgentScorer)],
            Box::new(ThunderAgentPicker),
        )
    }))
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
    use dynamo_kv_router::plugins::testing::{CaseWorker, SelectionCase};

    use super::*;

    fn case(target: WorkerWithDpRank) -> SelectionCase {
        SelectionCase::new(16, 16)
            .request_id("request")
            .affinity_target(target)
    }

    fn policy() -> WorkerSelectionPolicy {
        WorkerSelectionPolicy::new(
            "test",
            vec![Box::new(ThunderAgentScorer)],
            Box::new(ThunderAgentPicker),
        )
    }

    #[test]
    fn honors_the_classifier_target_when_eligible() {
        let worker_1 = WorkerWithDpRank::new(1, 0);
        let worker_2 = WorkerWithDpRank::new(2, 0);

        let selected = case(worker_2)
            .worker(1, CaseWorker::new())
            .worker(2, CaseWorker::new())
            .select(&policy())
            .unwrap();

        assert_eq!(selected, worker_2);
        assert_ne!(selected, worker_1);
    }

    #[test]
    fn falls_back_when_the_classifier_target_is_unavailable() {
        let unavailable = WorkerWithDpRank::new(2, 0);
        let worker = WorkerWithDpRank::new(4, 0);
        let mut case = case(unavailable);
        // Scores at block size 16 are 161, 100, and 84; both load terms affect selection.
        for (id, prefill_tokens, decode_blocks) in [(1, 1, 10), (3, 100, 0), (4, 20, 4)] {
            case.set_worker(
                id,
                CaseWorker::new()
                    .active_prefill_tokens(prefill_tokens)
                    .active_decode_blocks(decode_blocks),
            );
        }

        let selected = case.select(&policy()).unwrap();

        assert_eq!(selected, worker);
    }
}
