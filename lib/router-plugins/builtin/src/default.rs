// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The default KV policy, composed from a scorer and a picker using the public plugin API.

mod parameters;
mod picker;
mod scorer;
mod selector;

use parameters::PolicyParameters;
pub(super) use parameters::register;
pub use selector::{DefaultWorkerSelector, DefaultWorkerSelectorControl};

use dynamo_kv_router::plugins::worker_selection::{
    WorkerSelectionPolicy, WorkerSelectionPolicyFactory,
};
use dynamo_kv_router::{KvRouterConfig, WorkerType};
use parking_lot::Mutex;
use std::sync::Arc;

/// Construct the builtin default from configured policy parameters.
/// Per-request score overrides are not used. Request load-tracking remains host-owned.
pub fn default_policy(config: KvRouterConfig, worker_label: &'static str) -> WorkerSelectionPolicy {
    let parameters = PolicyParameters::from(&config);
    runtime_policy(config, parameters, worker_label, None)
}

/// Resolved runtime parameters a default policy starts with for `role`.
///
/// This is the map that `default_factory` exposes through `runtime_config()` at revision 0 and
/// that `PATCH /router_policy` patches: the startup values resolved from `config`, with cache
/// credits forced to zero on plain decode workers (decode without conditional disaggregation).
/// Keys are the policy parameter names, and every value is a JSON number. A selected policy
/// instance's own `parameters` are not applied: this describes the factory with no instance.
pub fn startup_policy_parameters(
    config: &KvRouterConfig,
    role: WorkerType,
) -> serde_json::Map<String, serde_json::Value> {
    role_parameters(config, role, PolicyParameters::from(config)).snapshot()
}

/// Build a runtime-configurable policy. A rebuild from patched parameters keeps `rng`, so a
/// seeded selector continues its random stream across updates.
fn runtime_policy(
    config: KvRouterConfig,
    parameters: PolicyParameters,
    worker_label: &'static str,
    rng: Option<Arc<Mutex<fastrand::Rng>>>,
) -> WorkerSelectionPolicy {
    let initial = parameters.snapshot();
    let policy = policy_with_rng(config.clone(), parameters, worker_label, rng.clone());
    policy.with_runtime_config(
        initial,
        Arc::new(move |values| {
            let parameters = PolicyParameters::from_runtime(values, &config)?;
            Ok(policy_with_rng(
                config.clone(),
                parameters,
                worker_label,
                rng.clone(),
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
    role: WorkerType,
    parameters: PolicyParameters,
) -> WorkerSelectionPolicy {
    let parameters = role_parameters(&config, role, parameters);
    runtime_policy(config, parameters, role.default_selector_label(), None)
}

fn role_parameters(
    config: &KvRouterConfig,
    role: WorkerType,
    parameters: PolicyParameters,
) -> PolicyParameters {
    let is_plain_decode = role == WorkerType::Decode && !config.conditional_disagg_enabled;
    // Plain decode has no cache inputs; expose its effective startup values so runtime
    // attempts to enable any cache credit are rejected by the input-capability check.
    if is_plain_decode {
        PolicyParameters {
            overlap_score_credit: 0.0,
            host_cache_hit_weight: 0.0,
            disk_cache_hit_weight: 0.0,
            shared_cache_multiplier: 0.0,
            ..parameters
        }
    } else {
        parameters
    }
}
