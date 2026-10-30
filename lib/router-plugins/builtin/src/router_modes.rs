// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The non-KV routing modes as worker-selection policies.
//!
//! `--router-mode random`, `round-robin`, `least-loaded` and `power-of-two` resolve to these types
//! by name. Each picker is a thin adapter over `dynamo-routing-algorithms`, the same selection code
//! `PushRouter` runs, and declares at most [`WorkerInputs::ACTIVE_REQUESTS`].

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use dynamo_kv_router::KvRouterConfig;
use dynamo_kv_router::plugins::worker_selection::{
    WorkerInputView, WorkerInputs, WorkerPicker, WorkerSelectionContext, WorkerSelectionPolicy,
    WorkerSelectionPolicyError, WorkerSelectionPolicyFactory,
};
use dynamo_kv_router::plugins::{
    RouterPluginRegistry, WorkerSelectionPolicyParameters, WorkerSelectionPolicyProviderError,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_routing_algorithms::{self as algorithms, ThreadRandom};

/// A uniform random worker.
pub const RANDOM: &str = "random";
/// One shared rotation over the workers.
pub const ROUND_ROBIN: &str = "round-robin";
/// The worker with the fewest active requests, ties broken at random.
pub const LEAST_LOADED: &str = "least-loaded";
/// The less loaded of two random workers.
pub const POWER_OF_TWO: &str = "power-of-two";

fn no_candidates() -> WorkerSelectionPolicyError {
    WorkerSelectionPolicyError::failed("no eligible worker")
}

/// The row's active requests. The picker declared the input, so every in-range row reads `Some`.
fn active_requests(input: WorkerInputView<'_>) -> impl Fn(usize) -> u64 + '_ {
    move |row| input.active_requests(row).unwrap_or_default()
}

struct RandomPicker;

impl WorkerPicker for RandomPicker {
    fn pick(
        &self,
        _context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        algorithms::random(input.len(), &mut ThreadRandom).ok_or_else(no_candidates)
    }
}

#[derive(Default)]
struct RoundRobinPicker {
    cursor: AtomicU64,
}

impl WorkerPicker for RoundRobinPicker {
    fn pick(
        &self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        if input.is_empty() {
            return Err(no_candidates());
        }
        // A preview reads the rotation without advancing it.
        let cursor = if context.is_preview() {
            self.cursor.load(Ordering::Relaxed)
        } else {
            self.cursor.fetch_add(1, Ordering::Relaxed)
        };
        algorithms::round_robin(cursor, input.len()).ok_or_else(no_candidates)
    }
}

struct LeastLoadedPicker;

impl WorkerPicker for LeastLoadedPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::ACTIVE_REQUESTS
    }

    fn pick(
        &self,
        _context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        algorithms::least_loaded(input.len(), active_requests(input), &mut ThreadRandom)
            .ok_or_else(no_candidates)
    }
}

struct PowerOfTwoPicker;

impl WorkerPicker for PowerOfTwoPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::ACTIVE_REQUESTS
    }

    fn pick(
        &self,
        _context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        algorithms::power_of_two(input.len(), active_requests(input), &mut ThreadRandom)
            .ok_or_else(no_candidates)
    }
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
    registry.register_worker_selection(RANDOM, Arc::new(provider(|| RandomPicker)))?;
    registry
        .register_worker_selection(ROUND_ROBIN, Arc::new(provider(RoundRobinPicker::default)))?;
    registry.register_worker_selection(LEAST_LOADED, Arc::new(provider(|| LeastLoadedPicker)))?;
    registry.register_worker_selection(POWER_OF_TWO, Arc::new(provider(|| PowerOfTwoPicker)))
}
