// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The default KV policy, composed from a scorer and a picker using the public plugin API.

mod parameters;
mod picker;
mod scorer;
mod selector;

use parameters::PolicyParameters;
pub(super) use parameters::register;
pub use selector::DefaultWorkerSelector;

use dynamo_kv_router::KvRouterConfig;
use dynamo_kv_router::plugins::worker_selection::{
    WorkerSelectionPolicy, WorkerSelectionPolicyFactory,
};
use parking_lot::Mutex;
use std::sync::Arc;

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
        policy_for_role(config.clone(), role, PolicyParameters::from(config))
    })
}

fn policy_for_role(
    config: KvRouterConfig,
    role: dynamo_kv_router::WorkerType,
    parameters: PolicyParameters,
) -> WorkerSelectionPolicy {
    let is_plain_decode =
        role == dynamo_kv_router::WorkerType::Decode && !config.conditional_disagg_enabled;
    policy_with_rng(
        config,
        parameters,
        role.default_selector_label(),
        None,
        is_plain_decode,
    )
}
