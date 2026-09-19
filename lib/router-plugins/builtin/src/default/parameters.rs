// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Startup parameters and provider registration for the default policy.

use dynamo_kv_router::KvRouterConfig;
use dynamo_kv_router::plugins::{
    RouterPluginRegistry, WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistryError,
};
use std::sync::Arc;

use super::policy_for_role;

#[derive(Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Parameters {
    overlap_score_credit: Option<f64>,
    overlap_score_credit_decay: Option<f64>,
    prefill_load_scale: Option<f64>,
    decode_active_request_weight: Option<f64>,
    host_cache_hit_weight: Option<f64>,
    disk_cache_hit_weight: Option<f64>,
    shared_cache_multiplier: Option<f64>,
    router_temperature: Option<f64>,
}

pub(crate) fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register_worker_selection(
        "dynamo-default-cost-fn",
        Arc::new(|parameters| {
            let parameters: Parameters = parameters.deserialize()?;
            for (name, value) in [
                ("overlap_score_credit", parameters.overlap_score_credit),
                (
                    "overlap_score_credit_decay",
                    parameters.overlap_score_credit_decay,
                ),
                ("prefill_load_scale", parameters.prefill_load_scale),
                (
                    "decode_active_request_weight",
                    parameters.decode_active_request_weight,
                ),
                ("host_cache_hit_weight", parameters.host_cache_hit_weight),
                ("disk_cache_hit_weight", parameters.disk_cache_hit_weight),
                (
                    "shared_cache_multiplier",
                    parameters.shared_cache_multiplier,
                ),
                ("router_temperature", parameters.router_temperature),
            ] {
                if value.is_some_and(|value| !value.is_finite() || value < 0.0) {
                    return Err(WorkerSelectionPolicyProviderError::new(format!(
                        "{name} must be finite and non-negative"
                    )));
                }
            }
            Ok(Arc::new(
                move |config: &KvRouterConfig, role, _partition| {
                    let mut config = config.clone();
                    if let Some(value) = parameters.overlap_score_credit {
                        config.overlap_score_credit = value;
                    }
                    if let Some(value) = parameters.overlap_score_credit_decay {
                        config.overlap_score_credit_decay = value;
                    }
                    if let Some(value) = parameters.prefill_load_scale {
                        config.prefill_load_scale = value;
                    }
                    if let Some(value) = parameters.decode_active_request_weight {
                        config.decode_active_request_weight = value;
                    }
                    if let Some(value) = parameters.host_cache_hit_weight {
                        config.host_cache_hit_weight = value;
                    }
                    if let Some(value) = parameters.disk_cache_hit_weight {
                        config.disk_cache_hit_weight = value;
                    }
                    if let Some(value) = parameters.shared_cache_multiplier {
                        config.shared_cache_multiplier = value;
                    }
                    if let Some(value) = parameters.router_temperature {
                        config.router_temperature = value;
                    }
                    policy_for_role(config, role)
                },
            ))
        }),
    )
}
