// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The default KV policy, composed from a scorer and a picker using the public plugin API.

mod parameters;
mod picker;
mod scorer;
mod selector;

use parameters::PolicyParameters;
pub(super) use parameters::register;
pub(crate) use picker::softmax_sample_index;
pub use selector::DefaultWorkerSelector;

use dynamo_kv_router::plugins::worker_selection::{
    WorkerScorer, WorkerSelectionPolicy, WorkerSelectionPolicyFactory,
};
use dynamo_kv_router::{KvRouterConfig, WorkerType};
use parking_lot::Mutex;
use std::sync::Arc;

/// The default cost scorer for `role`, with score weights resolved from `weights` and the role's
/// structure from the host's `config`, exactly as the default policy builds it. Policies that
/// build on the default logit compose this scorer, so its CACHE, LOAD and PREFERRED_TAINT handling
/// stays the default's own.
pub(crate) fn cost_scorer(
    config: &KvRouterConfig,
    weights: &KvRouterConfig,
    role: WorkerType,
) -> Box<dyn WorkerScorer> {
    let is_plain_decode = role == WorkerType::Decode && !config.conditional_disagg_enabled;
    scorer::build(
        &PolicyParameters::from(weights),
        role.default_selector_label(),
        is_plain_decode,
    )
}

/// The default picker's temperature for `config`, range-normalized as `softmax_sample_index` uses it.
pub(crate) fn router_temperature(config: &KvRouterConfig) -> f64 {
    PolicyParameters::from(config).router_temperature
}

/// Construct the builtin default from configured policy parameters.
/// Per-request score overrides are not used. Request load-tracking remains host-owned.
pub fn default_policy(config: KvRouterConfig, worker_label: &'static str) -> WorkerSelectionPolicy {
    let parameters = PolicyParameters::from(&config);
    policy_with_rng(config, parameters, worker_label, None, false)
}

fn policy_with_rng(
    config: KvRouterConfig,
    parameters: PolicyParameters,
    worker_label: &'static str,
    rng: Option<Arc<Mutex<fastrand::Rng>>>,
    is_plain_decode: bool,
) -> WorkerSelectionPolicy {
    let scorer = scorer::build(&parameters, worker_label, is_plain_decode);
    let picker = picker::DefaultPicker::new(parameters.router_temperature, rng);
    WorkerSelectionPolicy::new(config, worker_label, vec![scorer], Box::new(picker))
        .with_exclusive_affinity(true)
}

/// Factory installed by routing hosts, including hosts without a custom catalog.
pub fn default_factory() -> WorkerSelectionPolicyFactory {
    Arc::new(|config, role, _partition| {
        policy_for_role(config.clone(), role, PolicyParameters::from(config), None)
    })
}

fn policy_for_role(
    config: KvRouterConfig,
    role: dynamo_kv_router::WorkerType,
    parameters: PolicyParameters,
    rng: Option<Arc<Mutex<fastrand::Rng>>>,
) -> WorkerSelectionPolicy {
    let is_plain_decode =
        role == dynamo_kv_router::WorkerType::Decode && !config.conditional_disagg_enabled;
    policy_with_rng(
        config,
        parameters,
        role.default_selector_label(),
        rng,
        is_plain_decode,
    )
}
