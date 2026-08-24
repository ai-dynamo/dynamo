// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_router::{
    protocols::{WorkerSelectionResult, WorkerWithDpRank},
    scheduling::KvSchedulerError,
    selector::{HostedSelectionInputs, WorkerInputs, WorkerSelectionInput, WorkerSelector},
};
use dynamo_runtime::pipeline::{BuiltinRoutePicker, RouterMode};

use crate::local_model::runtime_config::ModelRuntimeConfig;

/// First-party selector hosted directly by [`RoutingHost`](super::RoutingHost).
pub(super) struct BuiltinWorkerSelector {
    mode: RouterMode,
    picker: BuiltinRoutePicker,
}

impl BuiltinWorkerSelector {
    pub(super) fn new(mode: RouterMode) -> Option<Self> {
        let picker = match mode {
            RouterMode::RoundRobin => BuiltinRoutePicker::round_robin(),
            RouterMode::Random => BuiltinRoutePicker::random(),
            RouterMode::PowerOfTwoChoices => BuiltinRoutePicker::power_of_two_choices(),
            RouterMode::LeastLoaded => BuiltinRoutePicker::least_loaded(),
            _ => return None,
        };
        Some(Self { mode, picker })
    }

    pub(super) fn peek_worker(
        &self,
        input: WorkerSelectionInput<'_, ModelRuntimeConfig>,
    ) -> Result<u64, KvSchedulerError> {
        let (worker_ids, occupancy) = self.hosted_inputs(input)?;
        self.picker
            .peek_worker(worker_ids, |worker_id| {
                occupancy.map_or(0, |occupancy| occupancy(worker_id))
            })
            .ok_or(KvSchedulerError::NoEndpoints)
    }

    fn hosted_inputs<'a>(
        &self,
        input: WorkerSelectionInput<'a, ModelRuntimeConfig>,
    ) -> Result<HostedSelectionInputs<'a>, KvSchedulerError> {
        let (worker_ids, occupancy) = input.into_hosted()?;
        if self
            .required_worker_inputs()
            .contains(WorkerInputs::OCCUPANCY)
            && occupancy.is_none()
        {
            return Err(dynamo_kv_router::WorkerSelectionPolicyError::failed(
                "selector requires hosted OCCUPANCY input",
            )
            .into());
        }
        Ok((worker_ids, occupancy))
    }
}

impl WorkerSelector<ModelRuntimeConfig> for BuiltinWorkerSelector {
    fn required_worker_inputs(&self) -> WorkerInputs {
        if self.mode.requires_occupancy() {
            WorkerInputs::OCCUPANCY
        } else {
            WorkerInputs::NONE
        }
    }

    fn select_worker(
        &self,
        input: WorkerSelectionInput<'_, ModelRuntimeConfig>,
    ) -> Result<WorkerSelectionResult, KvSchedulerError> {
        let (worker_ids, occupancy) = self.hosted_inputs(input)?;
        let worker_id = self
            .picker
            .select_worker(worker_ids, |worker_id| {
                occupancy.map_or(0, |occupancy| occupancy(worker_id))
            })
            .ok_or(KvSchedulerError::NoEndpoints)?;
        Ok(selection(worker_id))
    }
}

fn selection(worker_id: u64) -> WorkerSelectionResult {
    WorkerSelectionResult {
        worker: WorkerWithDpRank::from_worker_id(worker_id),
        required_blocks: 0,
        effective_overlap_blocks: 0.0,
        cached_tokens: 0,
        potential_decode_blocks: 0,
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    fn select(selector: &BuiltinWorkerSelector, worker_ids: &[u64]) -> u64 {
        selector
            .select_worker(WorkerSelectionInput::hosted(worker_ids, None))
            .unwrap()
            .worker
            .worker_id
    }

    #[test]
    fn round_robin_uses_hosted_selector_input() {
        let selector = BuiltinWorkerSelector::new(RouterMode::RoundRobin).unwrap();
        assert_eq!(selector.required_worker_inputs(), WorkerInputs::NONE);
        assert_eq!(select(&selector, &[10, 20]), 10);
        assert_eq!(select(&selector, &[10, 20]), 20);
        assert_eq!(select(&selector, &[10, 20]), 10);
    }

    #[test]
    fn random_uses_hosted_selector_input() {
        let selector = BuiltinWorkerSelector::new(RouterMode::Random).unwrap();
        assert_eq!(selector.required_worker_inputs(), WorkerInputs::NONE);
        for _ in 0..32 {
            assert!(matches!(select(&selector, &[10, 20]), 10 | 20));
        }
    }

    #[test]
    fn occupancy_policies_require_lazy_occupancy_input() {
        for mode in [RouterMode::PowerOfTwoChoices, RouterMode::LeastLoaded] {
            let selector = BuiltinWorkerSelector::new(mode).unwrap();
            assert_eq!(selector.required_worker_inputs(), WorkerInputs::OCCUPANCY);
            assert!(
                selector
                    .select_worker(WorkerSelectionInput::hosted(&[10, 20], None))
                    .is_err()
            );
        }
    }

    #[test]
    fn least_loaded_reads_hosted_occupancy() {
        let selector = BuiltinWorkerSelector::new(RouterMode::LeastLoaded).unwrap();
        let occupancy = |worker_id| if worker_id == 10 { 4 } else { 1 };
        let selected = selector
            .select_worker(WorkerSelectionInput::hosted(&[10, 20], Some(&occupancy)))
            .unwrap();
        assert_eq!(selected.worker.worker_id, 20);
    }

    #[test]
    fn power_of_two_choices_reads_only_two_occupancies() {
        let selector = BuiltinWorkerSelector::new(RouterMode::PowerOfTwoChoices).unwrap();
        let reads = Cell::new(0);
        let occupancy = |_| {
            reads.set(reads.get() + 1);
            0
        };

        selector
            .select_worker(WorkerSelectionInput::hosted(
                &[10, 20, 30, 40],
                Some(&occupancy),
            ))
            .unwrap();

        assert_eq!(reads.get(), 2);
    }
}
