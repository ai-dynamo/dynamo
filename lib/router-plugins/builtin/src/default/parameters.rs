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

/// Only the startup values consumed by the default scorer and picker.
#[derive(Clone, Copy)]
pub(super) struct PolicyParameters {
    pub(super) overlap_score_credit: f64,
    pub(super) overlap_score_credit_decay: f64,
    pub(super) prefill_load_scale: f64,
    pub(super) decode_active_request_weight: f64,
    pub(super) host_cache_hit_weight: f64,
    pub(super) disk_cache_hit_weight: f64,
    pub(super) shared_cache_multiplier: f64,
    pub(super) router_temperature: f64,
}

impl From<&KvRouterConfig> for PolicyParameters {
    fn from(config: &KvRouterConfig) -> Self {
        Parameters::default().resolve(config)
    }
}

impl Parameters {
    fn resolve(&self, config: &KvRouterConfig) -> PolicyParameters {
        PolicyParameters {
            overlap_score_credit: self
                .overlap_score_credit
                .unwrap_or(config.overlap_score_credit),
            overlap_score_credit_decay: self
                .overlap_score_credit_decay
                .unwrap_or(config.overlap_score_credit_decay),
            prefill_load_scale: self.prefill_load_scale.unwrap_or(config.prefill_load_scale),
            decode_active_request_weight: self
                .decode_active_request_weight
                .unwrap_or(config.decode_active_request_weight),
            host_cache_hit_weight: self
                .host_cache_hit_weight
                .unwrap_or(config.host_cache_hit_weight),
            disk_cache_hit_weight: self
                .disk_cache_hit_weight
                .unwrap_or(config.disk_cache_hit_weight),
            shared_cache_multiplier: self
                .shared_cache_multiplier
                .unwrap_or(config.shared_cache_multiplier),
            router_temperature: self.router_temperature.unwrap_or(config.router_temperature),
        }
    }
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
                    policy_for_role(config.clone(), role, parameters.resolve(config))
                },
            ))
        }),
    )
}
