// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Startup parameters and provider registration for the default policy.

use std::collections::HashMap;
use std::sync::Arc;

use dynamo_kv_router::plugins::prelude::*;

use super::{Preset, policy_for_role};
use crate::kv_fetch::{self, KvFetchParameters};

/// Policy type of the `kvcr` preset, selected under a `worker_selection` role.
pub const KVCR_POLICY_TYPE: &str = "kvcr";

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
    /// The hints the policy sends with the worker it books, by name; replaces the preset's
    /// default list. `[]` sends none.
    enable_hints: Option<Vec<String>>,
    /// Each enabled hint's parameters, keyed by its name.
    hint_parameters: HashMap<String, serde_json::Value>,
}

/// The hints a policy instance sends, resolved from `enable_hints` and `hint_parameters` once at
/// startup. Unlike [`PolicyParameters`], not part of the runtime-patchable snapshot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct HintSettings {
    pub(super) kv_fetch: Option<KvFetchParameters>,
}

impl HintSettings {
    /// The hints `preset` sends when its parameters list none: `kvcr` fetches the copies it
    /// credits, the default cost sends nothing.
    pub(super) fn default_for(preset: Preset) -> Self {
        match preset {
            Preset::Default => Self::default(),
            Preset::Kvcr => Self {
                kv_fetch: Some(KvFetchParameters::default()),
            },
        }
    }

    /// The emitter a policy with these settings attaches; `None` sends no hints. `credited` is
    /// whether the policy's score assumed the copy a fetch would make.
    pub(super) fn emitter(
        &self,
        credited: bool,
    ) -> Result<Option<Arc<dyn HintEmitter>>, WorkerSelectionPolicyProviderError> {
        self.kv_fetch
            .map(|parameters| kv_fetch::KvFetch::emitter(parameters, credited))
            .transpose()
    }
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

impl From<&SelectionConfig> for PolicyParameters {
    fn from(config: &SelectionConfig) -> Self {
        Parameters::default().resolve(config)
    }
}

impl Parameters {
    /// Resolve `enable_hints` and `hint_parameters` for `preset`, rejecting names no default
    /// policy knows, parameters for a hint that is not enabled, and copy credit without the fetch
    /// that would send the copies.
    fn hints(&self, preset: Preset) -> Result<HintSettings, WorkerSelectionPolicyProviderError> {
        let Some(names) = &self.enable_hints else {
            if let Some(name) = self.hint_parameters.keys().next() {
                return Err(WorkerSelectionPolicyProviderError::new(format!(
                    "hint_parameters configures {name:?}, but enable_hints is not set; list the hints to send, such as enable_hints: [{}]",
                    kv_fetch::HINT_NAME
                )));
            }
            let settings = HintSettings::default_for(preset);
            return Ok(settings);
        };
        let mut settings = HintSettings::default();
        for name in names {
            match name.as_str() {
                kv_fetch::HINT_NAME => {
                    if settings.kv_fetch.is_some() {
                        return Err(WorkerSelectionPolicyProviderError::new(format!(
                            "enable_hints lists {name:?} more than once"
                        )));
                    }
                    let parameters = match self.hint_parameters.get(name) {
                        Some(value) => serde_json::from_value(value.clone()).map_err(|error| {
                            WorkerSelectionPolicyProviderError::new(format!(
                                "invalid hint_parameters.{name}: {error}"
                            ))
                        })?,
                        None => KvFetchParameters::default(),
                    };
                    kv_fetch::KvFetch::new(parameters, false)?;
                    settings.kv_fetch = Some(parameters);
                }
                _ => {
                    return Err(WorkerSelectionPolicyProviderError::new(format!(
                        "enable_hints names unknown hint {name:?}; available: {}",
                        kv_fetch::HINT_NAME
                    )));
                }
            }
        }
        if let Some(name) = self
            .hint_parameters
            .keys()
            .find(|name| !names.contains(name))
        {
            return Err(WorkerSelectionPolicyProviderError::new(format!(
                "hint_parameters configures {name:?}, which enable_hints does not enable"
            )));
        }
        if preset == Preset::Kvcr
            && self
                .shared_cache_multiplier
                .is_some_and(|multiplier| multiplier > 0.0)
            && settings.kv_fetch.is_none()
        {
            return Err(WorkerSelectionPolicyProviderError::new(format!(
                "shared_cache_multiplier credits the booked worker for KV it would copy, which only the {} hint tells it to fetch; enable it, or set shared_cache_multiplier to 0",
                kv_fetch::HINT_NAME
            )));
        }
        Ok(settings)
    }

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

    fn resolve(&self, config: &SelectionConfig) -> PolicyParameters {
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
        config: &SelectionConfig,
    ) -> Result<Self, dynamo_kv_router::plugins::prelude::RuntimePolicyError> {
        use dynamo_kv_router::plugins::prelude::*;
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

/// Register the named default and the `kvcr` preset, which take the same parameters.
pub(crate) fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    for (policy_type, preset) in [
        ("dynamo-default-cost-fn", Preset::Default),
        (KVCR_POLICY_TYPE, Preset::Kvcr),
    ] {
        registry.register_worker_selection(
            policy_type,
            Arc::new(move |parameters| {
                let parameters: Parameters = parameters.deserialize()?;
                parameters.validate()?;
                let hints = parameters.hints(preset)?;
                Ok(Arc::new(
                    move |config: &SelectionConfig, role, _partition| {
                        policy_for_role(
                            config.clone(),
                            role,
                            parameters.resolve(config),
                            hints,
                            preset,
                        )
                    },
                ))
            }),
        )?;
    }
    Ok(())
}
