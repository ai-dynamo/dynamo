// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Custom worker-selection policy with separate prefill and decode behavior.

use std::sync::Arc;

use dynamo_kv_router::services::selection::{
    WorkerSelectionPolicyFactory, WorkerSelectionPolicyParameters,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistry,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::{
    KvRouterConfig, WorkerCandidate, WorkerInputView, WorkerInputs, WorkerPicker, WorkerScorer,
    WorkerSelectionContext, WorkerSelectionPolicy, WorkerSelectionPolicyError,
};

const DECODE_WORKER_TYPE: &str = "decode";

struct PrefillLoadScorer;

impl WorkerScorer for PrefillLoadScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD
    }

    fn score(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        candidate: &WorkerCandidate,
    ) -> Result<f64, WorkerSelectionPolicyError> {
        let load = candidate
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        Ok(load.active_prefill_tokens() as f64)
    }
}

struct DecodeLoadScorer;

impl WorkerScorer for DecodeLoadScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD
    }

    fn score(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        candidate: &WorkerCandidate,
    ) -> Result<f64, WorkerSelectionPolicyError> {
        let load = candidate
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        Ok(load.decode_cost_blocks())
    }
}

fn lowest_cost_row(input: WorkerInputView<'_>) -> Result<usize, WorkerSelectionPolicyError> {
    input
        .candidates()
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| left.cost().total_cmp(&right.cost()))
        .map(|(row, _)| row)
        .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))
}

struct PrefillPicker;

impl WorkerPicker for PrefillPicker {
    fn pick(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        lowest_cost_row(input)
    }
}

struct DecodePicker;

impl WorkerPicker for DecodePicker {
    fn pick(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        lowest_cost_row(input)
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Parameters {}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let _: Parameters = parameters.deserialize()?;

    Ok(Arc::new(
        |config: &KvRouterConfig, worker_type, _partition| {
            let (scorers, picker): (Vec<Box<dyn WorkerScorer>>, Box<dyn WorkerPicker>) =
                if worker_type == DECODE_WORKER_TYPE {
                    (vec![Box::new(DecodeLoadScorer)], Box::new(DecodePicker))
                } else {
                    (vec![Box::new(PrefillLoadScorer)], Box::new(PrefillPicker))
                };

            WorkerSelectionPolicy::new(config.clone(), worker_type, scorers, picker)
        },
    ))
}

pub fn register(
    registry: &mut WorkerSelectionPolicyRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register("disaggregated-load", Arc::new(provider))
}
