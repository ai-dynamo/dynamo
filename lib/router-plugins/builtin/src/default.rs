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
    runtime_policy(config, parameters, worker_label)
}

fn runtime_policy(
    config: KvRouterConfig,
    parameters: PolicyParameters,
    worker_label: &'static str,
) -> WorkerSelectionPolicy {
    let initial = parameters.snapshot();
    let policy = policy_with_rng(config.clone(), parameters, worker_label, None);
    policy.with_runtime_config(
        initial,
        Arc::new(move |values| {
            let parameters = PolicyParameters::from_runtime(values, &config)?;
            Ok(policy_with_rng(
                config.clone(),
                parameters,
                worker_label,
                None,
            ))
        }),
    )
}

fn policy_with_rng(
    config: KvRouterConfig,
    parameters: PolicyParameters,
    worker_label: &'static str,
    rng: Option<Arc<Mutex<fastrand::Rng>>>,
) -> WorkerSelectionPolicy {
    let scorer = scorer::build(&parameters, worker_label);
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
    // Plain decode has no cache inputs; expose its effective startup values so runtime
    // attempts to enable any cache credit are rejected by the input-capability check.
    let parameters = if is_plain_decode {
        PolicyParameters {
            overlap_score_credit: 0.0,
            host_cache_hit_weight: 0.0,
            disk_cache_hit_weight: 0.0,
            shared_cache_multiplier: 0.0,
            prefix_affinity_weight: 0.0,
            ..parameters
        }
    } else {
        parameters
    };
    runtime_policy(config, parameters, role.default_selector_label())
}
