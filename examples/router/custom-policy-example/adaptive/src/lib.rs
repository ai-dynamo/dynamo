// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Experimental, actor-local adaptive scorer budgets. No host lifecycle hooks are required.

pub mod controller;

use std::{sync::Arc, time::Instant};

use dynamo_kv_router::services::selection::{
    WorkerSelectionPolicyFactory, WorkerSelectionPolicyParameters,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistry,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::{
    RoutingPartitionId, WorkerInputView, WorkerInputs, WorkerPicker, WorkerSelectionContext,
    WorkerSelectionPolicy, WorkerSelectionPolicyError,
};

use controller::{AdaptiveRouter, Algorithm, Candidate, Parameters};

struct AdaptivePicker {
    router: AdaptiveRouter,
    started: Instant,
    partition: RoutingPartitionId,
    worker_type: &'static str,
    algorithm: Algorithm,
}

impl WorkerPicker for AdaptivePicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD
    }

    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let cache = input
            .cache()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("adaptive: missing cache input"))?;
        let load = input
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("adaptive: missing load input"))?;
        if cache.len() != load.len() || load.len() != input.candidates().len() {
            return Err(WorkerSelectionPolicyError::failed(
                "adaptive: misaligned candidate columns",
            ));
        }
        let blocks = context.request_blocks() as f64;
        let candidates = cache.iter().zip(load).map(|(cache, load)| Candidate {
            // Decode-only placement has no prefill saving to trade for load.
            affinity: if context.tracks_prefill_tokens() && blocks > 0.0 {
                cache.device_overlap_blocks() / blocks
            } else {
                0.0
            },
            active_requests: load.active_requests(),
        });
        let previous_updates = self.router.snapshot().updates;
        let selected = self
            .router
            .select(self.started.elapsed(), candidates)
            .ok_or_else(|| WorkerSelectionPolicyError::failed("adaptive: no eligible worker"))?;
        let state = self.router.snapshot();
        if state.updates != previous_updates {
            tracing::trace!(
                partition = %self.partition,
                worker_type = self.worker_type,
                algorithm = ?self.algorithm,
                distribution_weight = state.distribution_weight,
                imbalance = state.imbalance,
                pressure = state.pressure,
                updates = state.updates,
                "adaptive worker-selection budget"
            );
        }
        Ok(selected)
    }
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let parameters: Parameters = parameters.deserialize()?;
    parameters
        .validate()
        .map_err(WorkerSelectionPolicyProviderError::new)?;
    Ok(Arc::new(move |config, worker_type, partition| {
        WorkerSelectionPolicy::new(
            config.clone(),
            worker_type.as_str(),
            Vec::new(),
            Box::new(AdaptivePicker {
                router: AdaptiveRouter::from_validated(parameters.clone()),
                started: Instant::now(),
                partition: partition.into_owned(),
                worker_type: worker_type.as_str(),
                algorithm: parameters.algorithm,
            }),
        )
    }))
}

/// Register the opt-in `adaptive` type in a linked policy catalog.
pub fn register(
    registry: &mut WorkerSelectionPolicyRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register("adaptive", Arc::new(provider))
}
