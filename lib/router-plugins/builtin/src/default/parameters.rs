// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Startup parameters and provider registration for the default policy.

use dynamo_kv_router::KvRouterConfig;
use dynamo_kv_router::config::SharedCacheType;
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
#[derive(Clone, Copy, serde::Serialize)]
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
    fn validate(&self) -> Result<(), WorkerSelectionPolicyProviderError> {
        for (name, value) in [
            ("overlap_score_credit", self.overlap_score_credit),
            (
                "overlap_score_credit_decay",
                self.overlap_score_credit_decay,
            ),
            ("prefill_load_scale", self.prefill_load_scale),
            (
                "decode_active_request_weight",
                self.decode_active_request_weight,
            ),
            ("host_cache_hit_weight", self.host_cache_hit_weight),
            ("disk_cache_hit_weight", self.disk_cache_hit_weight),
            ("shared_cache_multiplier", self.shared_cache_multiplier),
            ("router_temperature", self.router_temperature),
        ] {
            if value.is_some_and(|value| !value.is_finite() || value < 0.0) {
                return Err(WorkerSelectionPolicyProviderError::new(format!(
                    "{name} must be finite and non-negative"
                )));
            }
        }
        Ok(())
    }

    fn resolve(&self, config: &KvRouterConfig) -> PolicyParameters {
        let overlap_score_credit = self
            .overlap_score_credit
            .unwrap_or(config.overlap_score_credit);
        // Legacy load-only configs retain tier weights for host accounting. Only explicit
        // policy parameters enable tier scoring independently of device credit.
        let inherited_cache_weight = |weight| {
            if overlap_score_credit == 0.0 {
                0.0
            } else {
                weight
            }
        };
        PolicyParameters {
            overlap_score_credit,
            overlap_score_credit_decay: self
                .overlap_score_credit_decay
                .unwrap_or(config.overlap_score_credit_decay),
            prefill_load_scale: self.prefill_load_scale.unwrap_or(config.prefill_load_scale),
            decode_active_request_weight: self
                .decode_active_request_weight
                .unwrap_or(config.decode_active_request_weight),
            host_cache_hit_weight: self
                .host_cache_hit_weight
                .unwrap_or(inherited_cache_weight(config.host_cache_hit_weight)),
            disk_cache_hit_weight: self
                .disk_cache_hit_weight
                .unwrap_or(inherited_cache_weight(config.disk_cache_hit_weight)),
            shared_cache_multiplier: self.shared_cache_multiplier.unwrap_or_else(|| {
                inherited_cache_weight(config.shared_cache_multiplier.unwrap_or(
                    match config.shared_cache_type {
                        SharedCacheType::None => 0.0,
                        SharedCacheType::Hicache => 0.5,
                    },
                ))
            }),
            router_temperature: self.router_temperature.unwrap_or(config.router_temperature),
        }
    }
}

impl PolicyParameters {
    pub(super) fn snapshot(&self) -> serde_json::Map<String, serde_json::Value> {
        let serde_json::Value::Object(parameters) = serde_json::json!(self) else {
            unreachable!("PolicyParameters is a struct that serializes to a JSON object")
        };
        parameters
    }

    pub(super) fn from_runtime(
        values: serde_json::Map<String, serde_json::Value>,
        config: &KvRouterConfig,
    ) -> Result<Self, dynamo_kv_router::plugins::worker_selection::RuntimePolicyError> {
        use dynamo_kv_router::plugins::worker_selection::RuntimePolicyError;
        // Runtime maps are resolved; startup parameters may be absent or null.
        if values.values().any(serde_json::Value::is_null) {
            return Err(RuntimePolicyError::InvalidParameters(
                "resolved runtime parameters must be numbers".into(),
            ));
        }
        let parameters: Parameters = serde_json::from_value(serde_json::Value::Object(values))
            .map_err(|error| RuntimePolicyError::InvalidParameters(error.to_string()))?;
        parameters
            .validate()
            .map_err(|error| RuntimePolicyError::InvalidParameters(error.to_string()))?;
        Ok(parameters.resolve(config))
    }
}

pub(crate) fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register_worker_selection(
        "dynamo-default-cost-fn",
        Arc::new(|parameters| {
            let parameters: Parameters = parameters.deserialize()?;
            parameters.validate()?;
            Ok(Arc::new(
                move |config: &KvRouterConfig, role, _partition| {
                    policy_for_role(config.clone(), role, parameters.resolve(config))
                },
            ))
        }),
    )
}
