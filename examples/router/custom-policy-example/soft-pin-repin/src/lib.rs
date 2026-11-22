// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Custom worker-selection policy that demonstrates overload-aware soft affinity.
//!
//! The picker retains an eligible affinity target until its active request count
//! exceeds the configured threshold. It then selects the least-loaded alternative,
//! and Dynamo rebinds the session to the dispatched worker.

use std::sync::Arc;

use dynamo_kv_router::plugins::worker_selection::*;

struct SoftPinRepinPicker {
    max_active_requests: usize,
}

impl SoftPinRepinPicker {
    fn matches_target(candidate: &ScoredWorkerCandidate, target: WorkerAffinityTarget) -> bool {
        let worker = candidate.worker();
        worker.worker_id == target.worker_id
            && target.dp_rank.is_none_or(|rank| worker.dp_rank == rank)
    }

    fn least_loaded_row(
        candidates: &[ScoredWorkerCandidate],
        loads: &[WorkerLoadInput],
        excluded_target: Option<WorkerAffinityTarget>,
    ) -> Option<usize> {
        candidates
            .iter()
            .zip(loads)
            .enumerate()
            .filter(|(_, (candidate, _))| {
                excluded_target.is_none_or(|target| !Self::matches_target(candidate, target))
            })
            .min_by_key(|(_, (candidate, load))| (load.active_requests(), candidate.worker()))
            .map(|(row, _)| row)
    }
}

impl WorkerPicker for SoftPinRepinPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD
    }

    fn pick(
        &self,
        request: &WorkerSelectionRequest<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let candidates = input.candidates();
        let loads = input.load().ok_or_else(|| {
            WorkerSelectionPolicyError::failed("active load input is unavailable")
        })?;

        if let Some(target) = request.affinity_target()
            && let Some(target_row) = candidates
                .iter()
                .zip(loads)
                .enumerate()
                .filter(|(_, (candidate, _))| Self::matches_target(candidate, target))
                .min_by_key(|(_, (candidate, load))| (load.active_requests(), candidate.worker()))
                .map(|(row, _)| row)
        {
            if loads[target_row].active_requests() <= self.max_active_requests {
                return Ok(target_row);
            }

            return Ok(
                Self::least_loaded_row(candidates, loads, Some(target)).unwrap_or(target_row)
            );
        }

        Self::least_loaded_row(candidates, loads, None)
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Parameters {
    max_active_requests: usize,
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let parameters: Parameters = parameters.deserialize()?;
    let max_active_requests = parameters.max_active_requests;

    Ok(Arc::new(move |context| {
        WorkerSelectionPolicy::new(
            context.worker_type().as_str(),
            Vec::new(),
            Box::new(SoftPinRepinPicker {
                max_active_requests,
            }),
        )
    }))
}

pub fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register_worker_selection("soft-pin-repin", Arc::new(provider))
}

#[cfg(test)]
mod tests {
    use dynamo_kv_router::plugins::testing::{CaseWorker, SelectionCase};

    use super::*;

    const A: u64 = 29;
    const B: u64 = 41;

    fn worker(id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::from_worker_id(id)
    }

    /// A request, bound to `affinity_target` when given, against workers given as
    /// `(worker_id, active_requests)`.
    fn case(affinity_target: Option<WorkerWithDpRank>, workers: &[(u64, usize)]) -> SelectionCase {
        let mut case = SelectionCase::new(16, 16);
        if let Some(target) = affinity_target {
            case = case.affinity_target(target);
        }
        for &(id, active_requests) in workers {
            case.set_worker(
                id,
                CaseWorker::new()
                    .active_requests(active_requests)
                    .total_kv_blocks(1024),
            );
        }
        case
    }

    fn policy(max_active_requests: usize) -> WorkerSelectionPolicy {
        WorkerSelectionPolicy::new(
            "test",
            Vec::new(),
            Box::new(SoftPinRepinPicker {
                max_active_requests,
            }),
        )
    }

    #[test]
    fn repins_only_after_the_affinity_target_exceeds_the_threshold() {
        let selected = case(None, &[(A, 0), (B, 2)]).select(&policy(0)).unwrap();
        assert_eq!(selected, worker(A));

        let retained = case(Some(selected), &[(A, 0), (B, 0)])
            .select(&policy(0))
            .unwrap();
        assert_eq!(retained, worker(A));

        let repinned = case(Some(selected), &[(A, 1), (B, 0)])
            .select(&policy(0))
            .unwrap();
        assert_eq!(repinned, worker(B));
    }

    #[test]
    fn retains_the_only_eligible_affinity_target() {
        let selected = case(Some(worker(A)), &[(A, 1)]).select(&policy(0)).unwrap();
        assert_eq!(selected, worker(A));
    }
}
