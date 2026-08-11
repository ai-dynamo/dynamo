// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Factory and registration for the `disaggregated-load` policy.
//!
//! The factory selects prefill or decode components for each routing partition.

mod picker;
mod scorer;

use std::sync::Arc;

use dynamo_kv_router::services::selection::{
    WorkerSelectionPolicyFactory, WorkerSelectionPolicyParameters,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistry,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::{KvRouterConfig, WorkerPicker, WorkerScorer, WorkerSelectionPolicy};
use picker::{DecodePicker, PrefillPicker};
use scorer::{DecodeLoadScorer, PrefillLoadScorer};

const DECODE_WORKER_TYPE: &str = "decode";

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
