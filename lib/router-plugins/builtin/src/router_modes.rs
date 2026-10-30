// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The non-KV routing modes as worker-selection policies, one module per mode.
//!
//! `--router-mode random`, `round-robin`, `least-loaded` and `power-of-two` resolve to these types
//! by name. Each picker is a thin adapter over `dynamo-routing-algorithms`, the same selection code
//! `PushRouter` runs, and declares at most [`WorkerInputs::ACTIVE_REQUESTS`].
//!
//! [`WorkerInputs::ACTIVE_REQUESTS`]: dynamo_kv_router::plugins::worker_selection::WorkerInputs::ACTIVE_REQUESTS

mod least_loaded;
mod power_of_two;
mod random;
mod round_robin;

use std::sync::Arc;

use dynamo_kv_router::KvRouterConfig;
use dynamo_kv_router::plugins::worker_selection::{
    WorkerInputView, WorkerPicker, WorkerSelectionPolicy, WorkerSelectionPolicyError,
    WorkerSelectionPolicyFactory,
};
use dynamo_kv_router::plugins::{
    RouterPluginRegistry, WorkerSelectionPolicyParameters, WorkerSelectionPolicyProviderError,
    WorkerSelectionPolicyRegistryError,
};

pub use least_loaded::LEAST_LOADED;
pub use power_of_two::POWER_OF_TWO;
pub use random::RANDOM;
pub use round_robin::ROUND_ROBIN;

fn no_candidates() -> WorkerSelectionPolicyError {
    WorkerSelectionPolicyError::failed("no eligible worker")
}

/// The row's active requests. The picker declared the input, so every in-range row reads `Some`.
fn active_requests(input: WorkerInputView<'_>) -> impl Fn(usize) -> u64 + '_ {
    move |row| input.active_requests(row).unwrap_or_default()
}

/// A provider whose factory builds a fresh picker per call, so every partition and selector
/// keeps its own state, such as its own rotation.
fn provider<P: WorkerPicker + 'static>(
    picker: fn() -> P,
) -> impl Fn(
    &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError>
+ Send
+ Sync
+ 'static {
    move |_parameters| {
        Ok(Arc::new(
            move |config: &KvRouterConfig, worker_type, _partition| {
                WorkerSelectionPolicy::new(
                    config.clone(),
                    worker_type.as_str(),
                    Vec::new(),
                    Box::new(picker()),
                )
            },
        ))
    }
}

/// Register the routing modes. Every host links them through `default_registry`.
pub(crate) fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register_worker_selection(RANDOM, Arc::new(provider(|| random::RandomPicker)))?;
    registry.register_worker_selection(
        ROUND_ROBIN,
        Arc::new(provider(round_robin::RoundRobinPicker::default)),
    )?;
    registry.register_worker_selection(
        LEAST_LOADED,
        Arc::new(provider(|| least_loaded::LeastLoadedPicker)),
    )?;
    registry.register_worker_selection(
        POWER_OF_TWO,
        Arc::new(provider(|| power_of_two::PowerOfTwoPicker)),
    )
}
