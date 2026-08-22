// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use dynamo_kv_router::{
    protocols::{WorkerId, WorkerSelectionResult, WorkerWithDpRank},
    scheduling::{KvSchedulerError, RoutingEligibility, SchedulingRequest},
    selector::{WorkerInputs, WorkerSelector},
};
use dynamo_runtime::pipeline::StatelessRoutePicker;

use crate::local_model::runtime_config::ModelRuntimeConfig;

use super::BuiltinRoutingPolicy;

/// Cache-free builtin policy implemented through the worker-selection trait.
pub(super) struct StatelessWorkerSelector {
    picker: StatelessRoutePicker,
}

impl StatelessWorkerSelector {
    pub(super) fn new(policy: BuiltinRoutingPolicy) -> Option<Self> {
        let picker = match policy {
            BuiltinRoutingPolicy::RoundRobin => StatelessRoutePicker::round_robin(),
            BuiltinRoutingPolicy::Random => StatelessRoutePicker::random(),
            BuiltinRoutingPolicy::PowerOfTwoChoices | BuiltinRoutingPolicy::LeastLoaded => {
                return None;
            }
        };
        Some(Self { picker })
    }

    pub(super) fn select_worker_id(
        &self,
        worker_ids: &[WorkerId],
    ) -> Result<WorkerId, KvSchedulerError> {
        <Self as WorkerSelector<ModelRuntimeConfig>>::select_worker_from_ids(self, worker_ids)
            .map(|selection| selection.worker.worker_id)
    }

    pub(super) fn peek_worker_id(&self, worker_ids: &[WorkerId]) -> Option<WorkerId> {
        self.picker
            .peek_index(worker_ids.len())
            .map(|row| worker_ids[row])
    }

    fn select(
        &self,
        candidates: &[WorkerWithDpRank],
    ) -> Result<WorkerSelectionResult, KvSchedulerError> {
        let row = self
            .picker
            .select_index(candidates.len())
            .ok_or(KvSchedulerError::NoEndpoints)?;
        Ok(selection(candidates[row]))
    }
}

impl WorkerSelector<ModelRuntimeConfig> for StatelessWorkerSelector {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::NONE
    }

    fn select_worker(
        &self,
        workers: &HashMap<WorkerId, ModelRuntimeConfig>,
        _request: &SchedulingRequest,
        eligibility: RoutingEligibility<'_>,
        _block_size: u32,
    ) -> Result<WorkerSelectionResult, KvSchedulerError> {
        let mut candidates = Vec::new();
        eligibility.for_each_eligible_worker_rank(workers, |worker, _| candidates.push(worker));
        candidates.sort_unstable();
        self.select(&candidates)
    }

    fn select_worker_from_ids(
        &self,
        worker_ids: &[WorkerId],
    ) -> Result<WorkerSelectionResult, KvSchedulerError> {
        let row = self
            .picker
            .select_index(worker_ids.len())
            .ok_or(KvSchedulerError::NoEndpoints)?;
        Ok(selection(WorkerWithDpRank::from_worker_id(worker_ids[row])))
    }
}

fn selection(worker: WorkerWithDpRank) -> WorkerSelectionResult {
    WorkerSelectionResult {
        worker,
        required_blocks: 0,
        effective_overlap_blocks: 0.0,
        cached_tokens: 0,
        potential_decode_blocks: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_robin_uses_id_only_selector_trait_path() {
        let selector = StatelessWorkerSelector::new(BuiltinRoutingPolicy::RoundRobin).unwrap();
        assert_eq!(selector.required_worker_inputs(), WorkerInputs::NONE);
        assert_eq!(selector.select_worker_id(&[10, 20]).unwrap(), 10);
        assert_eq!(selector.select_worker_id(&[10, 20]).unwrap(), 20);
        assert_eq!(selector.select_worker_id(&[10, 20]).unwrap(), 10);
    }

    #[test]
    fn random_uses_id_only_selector_trait_path() {
        let selector = StatelessWorkerSelector::new(BuiltinRoutingPolicy::Random).unwrap();
        assert_eq!(selector.required_worker_inputs(), WorkerInputs::NONE);
        for _ in 0..32 {
            assert!(matches!(
                selector.select_worker_id(&[10, 20]).unwrap(),
                10 | 20
            ));
        }
    }
}
