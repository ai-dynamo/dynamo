// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::protocols::tensor;
use dynamo_kv_router::protocols::{KvTransferEnforcement, KvTransferPreferredWeight};

// Reserve a topology namespace so generated taints can be rebuilt without touching caller taints.
pub const TOPOLOGY_TAINT_PREFIX: &str = "dynamo.topology/";

/// Canonical worker-taint form for topology metadata.
///
/// A topology domain/value pair such as `zone=us-east-1a` becomes
/// `dynamo.topology/zone=us-east-1a`.
pub fn topology_taint(domain: &str, value: &str) -> String {
    format!("{TOPOLOGY_TAINT_PREFIX}{domain}={value}")
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct DisaggregatedEndpoint {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bootstrap_host: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bootstrap_port: Option<u16>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelRuntimeConfig {
    pub total_kv_blocks: Option<u64>,

    pub max_num_seqs: Option<u64>,

    pub max_num_batched_tokens: Option<u64>,

    pub tool_call_parser: Option<String>,

    pub reasoning_parser: Option<String>,

    /// When true, strip tool definitions from the chat template when tool_choice is "none".
    #[serde(default = "default_exclude_tools_when_tool_choice_none")]
    pub exclude_tools_when_tool_choice_none: bool,

    /// Starting rank of data parallel ranks for this worker (0 if DP not enabled)
    #[serde(default = "default_data_parallel_start_rank")]
    pub data_parallel_start_rank: u32,

    /// Total number of data parallel ranks for this worker (1 if DP not enabled)
    #[serde(default = "default_data_parallel_size")]
    pub data_parallel_size: u32,

    /// Enable worker-local KV indexer for tracking this worker's own KV cache state (default: true)
    #[serde(default = "default_local_indexer")]
    pub enable_local_indexer: bool,

    /// Mapping of engine-specific runtime configs
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub runtime_data: HashMap<String, serde_json::Value>,

    // Provide tensor model config in the case where the model type is Tensor.
    // Currently use JSON object for convinence, the programmatic way is to
    // define the model config struct as part of the tensor protocol and
    // import it here.
    // [gluo TODO] switch to ModelConfig if desired and workout a way to
    // prepare it in a convinent way, the protobuf library used by tonic
    // doesn't provide JSON parsing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tensor_model_config: Option<tensor::TensorModelConfig>,

    /// Bootstrap endpoint for disaggregated serving (prefill workers publish this)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disaggregated_endpoint: Option<DisaggregatedEndpoint>,

    #[serde(default = "default_eagle")]
    pub enable_eagle: bool,

    #[serde(default, skip_serializing_if = "HashSet::is_empty")]
    pub taints: HashSet<String>,

    /// Stable routing identity, set via the `DYN_STABLE_ROUTING_ID` env var. Used as
    /// the rendezvous-hash key so cache assignments survive a new ephemeral
    /// `worker_id`. `None` if unset.
    ///
    /// Recommended k8s wire-up (downward API on a StatefulSet pod):
    ///
    /// ```yaml
    /// env:
    ///   - name: DYN_STABLE_ROUTING_ID
    ///     valueFrom:
    ///       fieldRef:
    ///         fieldPath: metadata.name
    /// ```
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stable_routing_id: Option<String>,

    /// Topology domain labels for this worker (e.g. {"zone": "us-east-1a", "rack": "rack1"}).
    /// Workers publish these as metadata and as additive canonical taints with the
    /// `dynamo.topology/<domain>=<value>` format.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub topology_domains: HashMap<String, String>,

    /// Topology domain used for KV-cache transfer routing (e.g. "zone").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_transfer_domain: Option<String>,

    /// KV transfer topology enforcement mode selected by DGD (`required` or `preferred`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_transfer_enforcement: Option<KvTransferEnforcement>,

    /// Preferred-taint weight used when `kv_transfer_enforcement` is `preferred`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_transfer_preferred_weight: Option<KvTransferPreferredWeight>,
}

const fn default_data_parallel_start_rank() -> u32 {
    0
}

const fn default_data_parallel_size() -> u32 {
    1
}

const fn default_local_indexer() -> bool {
    true
}

const fn default_exclude_tools_when_tool_choice_none() -> bool {
    true
}

const fn default_eagle() -> bool {
    false
}

impl Default for ModelRuntimeConfig {
    fn default() -> Self {
        Self {
            total_kv_blocks: None,
            max_num_seqs: None,
            max_num_batched_tokens: None,
            tool_call_parser: None,
            reasoning_parser: None,
            exclude_tools_when_tool_choice_none: default_exclude_tools_when_tool_choice_none(),
            data_parallel_start_rank: default_data_parallel_start_rank(),
            data_parallel_size: default_data_parallel_size(),
            enable_local_indexer: true,
            runtime_data: HashMap::new(),
            tensor_model_config: None,
            disaggregated_endpoint: None,
            enable_eagle: false,
            taints: HashSet::new(),
            stable_routing_id: None,
            topology_domains: HashMap::new(),
            kv_transfer_domain: None,
            kv_transfer_enforcement: None,
            kv_transfer_preferred_weight: None,
        }
    }
}

impl dynamo_kv_router::WorkerConfigLike for ModelRuntimeConfig {
    fn data_parallel_start_rank(&self) -> u32 {
        self.data_parallel_start_rank
    }

    fn data_parallel_size(&self) -> u32 {
        self.data_parallel_size
    }

    fn max_num_batched_tokens(&self) -> Option<u64> {
        self.max_num_batched_tokens
    }

    fn total_kv_blocks(&self) -> Option<u64> {
        self.total_kv_blocks
    }

    fn taints(&self) -> &HashSet<String> {
        &self.taints
    }

    fn stable_routing_id(&self) -> Option<&str> {
        self.stable_routing_id.as_deref()
    }

    fn topology_domains(&self) -> Option<&HashMap<String, String>> {
        if self.topology_domains.is_empty() {
            None
        } else {
            Some(&self.topology_domains)
        }
    }

    fn kv_transfer_domain(&self) -> Option<&str> {
        self.kv_transfer_domain.as_deref()
    }

    fn kv_transfer_enforcement(&self) -> Option<KvTransferEnforcement> {
        self.kv_transfer_enforcement
    }

    fn kv_transfer_preferred_weight(&self) -> Option<KvTransferPreferredWeight> {
        self.kv_transfer_preferred_weight
    }
}

impl ModelRuntimeConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_engine_specific<T: Serialize>(&mut self, key: &str, value: T) -> anyhow::Result<()> {
        self.runtime_data
            .insert(key.to_string(), serde_json::to_value(value)?);
        Ok(())
    }

    pub fn get_engine_specific<T: DeserializeOwned>(&self, key: &str) -> anyhow::Result<Option<T>> {
        if let Some(value) = self.runtime_data.get(key) {
            Ok(Some(serde_json::from_value(value.clone())?))
        } else {
            Ok(None)
        }
    }

    /// Validate topology and KV-transfer fields before the runtime config is published.
    pub fn validate_topology_config(&self) -> anyhow::Result<()> {
        for (domain, value) in &self.topology_domains {
            let trimmed_domain = domain.trim();
            if trimmed_domain.is_empty() {
                anyhow::bail!("topology_domains keys must be non-empty");
            }
            if trimmed_domain != domain {
                anyhow::bail!(
                    "topology_domains key {domain:?} must not contain leading or trailing whitespace"
                );
            }

            let trimmed_value = value.trim();
            if trimmed_value.is_empty() {
                anyhow::bail!("topology_domains value for {domain:?} must be non-empty");
            }
            if trimmed_value != value {
                anyhow::bail!(
                    "topology_domains value for {domain:?} must not contain leading or trailing whitespace"
                );
            }
        }

        if let Some(domain) = &self.kv_transfer_domain {
            let trimmed_domain = domain.trim();
            if trimmed_domain.is_empty() {
                anyhow::bail!("kv_transfer_domain must be non-empty when set");
            }
            if trimmed_domain != domain {
                anyhow::bail!("kv_transfer_domain must not contain leading or trailing whitespace");
            }

            let Some(value) = self.topology_domains.get(domain) else {
                anyhow::bail!(
                    "kv_transfer_domain {domain:?} must reference a key in topology_domains"
                );
            };
            if value.is_empty() {
                anyhow::bail!(
                    "topology_domains entry for kv_transfer_domain {domain:?} must be non-empty"
                );
            }
        }

        if self.kv_transfer_enforcement.is_some() && self.kv_transfer_domain.is_none() {
            anyhow::bail!("kv_transfer_enforcement requires kv_transfer_domain");
        }

        if matches!(
            self.kv_transfer_enforcement,
            Some(KvTransferEnforcement::Preferred)
        ) && self.kv_transfer_preferred_weight.is_none()
        {
            anyhow::bail!(
                "kv_transfer_preferred_weight is required when kv_transfer_enforcement is preferred"
            );
        }

        if self.kv_transfer_preferred_weight.is_some()
            && !matches!(
                self.kv_transfer_enforcement,
                Some(KvTransferEnforcement::Preferred)
            )
        {
            anyhow::bail!(
                "kv_transfer_preferred_weight can only be set when kv_transfer_enforcement is preferred"
            );
        }

        Ok(())
    }

    /// Rebuild canonical topology taints derived from `topology_domains`.
    ///
    /// Existing caller-provided taints outside the reserved topology prefix are preserved; generated
    /// topology taints are refreshed in the same set so `RoutingConstraints` can match them through
    /// the standard worker taints path.
    pub fn add_topology_taints(&mut self) -> &mut Self {
        self.taints
            .retain(|taint| !taint.starts_with(TOPOLOGY_TAINT_PREFIX));
        self.taints
            .extend(self.topology_domains.iter().filter_map(|(domain, value)| {
                let domain = domain.trim();
                let value = value.trim();
                if domain.is_empty() || value.is_empty() {
                    None
                } else {
                    Some(topology_taint(domain, value))
                }
            }));
        self
    }

    /// Populate `stable_routing_id` from the `DYN_STABLE_ROUTING_ID` environment variable.
    ///
    /// Sets the field only if it is currently unset; returns `&mut self` for chaining. If
    /// `DYN_STABLE_ROUTING_ID` is unset or empty/whitespace-only, the field is left
    /// as `None`.
    ///
    /// See the doc on [`ModelRuntimeConfig::stable_routing_id`] for the recommended k8s
    /// downward-API recipe.
    pub fn populate_stable_routing_id_from_env(&mut self) -> &mut Self {
        if self.stable_routing_id.is_some() {
            return self;
        }
        let candidate = std::env::var("DYN_STABLE_ROUTING_ID")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        if let Some(value) = candidate {
            tracing::info!(stable_routing_id = %value, "populated stable_routing_id from DYN_STABLE_ROUTING_ID");
            self.stable_routing_id = Some(value);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Env-touching tests use `temp_env` (snapshot + restore around the closure) and
    // `#[serial_test::serial]` (serialize against every other env-touching test in the
    // binary, not just this module). A module-local mutex would be insufficient because
    // `std::env::{set_var, remove_var}` race against `getenv` in any other thread.

    #[test]
    #[serial_test::serial]
    fn populates_from_dyn_env() {
        temp_env::with_vars([("DYN_STABLE_ROUTING_ID", Some("worker-3"))], || {
            let mut cfg = ModelRuntimeConfig::default();
            cfg.populate_stable_routing_id_from_env();
            assert_eq!(cfg.stable_routing_id.as_deref(), Some("worker-3"));
        });
    }

    #[test]
    #[serial_test::serial]
    fn preserves_caller_supplied_value() {
        temp_env::with_vars([("DYN_STABLE_ROUTING_ID", Some("from-env"))], || {
            let mut cfg = ModelRuntimeConfig {
                stable_routing_id: Some("explicit".to_string()),
                ..Default::default()
            };
            cfg.populate_stable_routing_id_from_env();
            assert_eq!(cfg.stable_routing_id.as_deref(), Some("explicit"));
        });
    }

    #[test]
    #[serial_test::serial]
    fn no_meaningful_env_leaves_field_none() {
        // Whitespace-only is rejected…
        temp_env::with_vars([("DYN_STABLE_ROUTING_ID", Some("   "))], || {
            let mut cfg = ModelRuntimeConfig::default();
            cfg.populate_stable_routing_id_from_env();
            assert!(cfg.stable_routing_id.is_none());
        });
        // …as is having the var unset.
        temp_env::with_vars_unset(["DYN_STABLE_ROUTING_ID"], || {
            let mut cfg = ModelRuntimeConfig::default();
            cfg.populate_stable_routing_id_from_env();
            assert!(cfg.stable_routing_id.is_none());
        });
    }

    #[test]
    fn roundtrips_through_serde_json() {
        let cfg = ModelRuntimeConfig {
            stable_routing_id: Some("worker-7".to_string()),
            ..Default::default()
        };
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(json.contains("\"stable_routing_id\":\"worker-7\""));
        let parsed: ModelRuntimeConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.stable_routing_id.as_deref(), Some("worker-7"));
    }

    #[test]
    fn serde_skips_when_none() {
        let cfg = ModelRuntimeConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(!json.contains("stable_routing_id"));
    }

    #[test]
    fn test_serde_empty_topology_domains_omitted() {
        let config = ModelRuntimeConfig::default();
        let serialized = serde_json::to_string(&config).unwrap();

        // Empty topology_domains should not appear in serialized output
        assert!(
            !serialized.contains("topology_domains"),
            "empty topology_domains should be skipped during serialization, got: {serialized}"
        );
    }

    #[test]
    fn test_serde_backward_compat_deserialize_without_topology_domains() {
        // Simulate a config serialized before topology_domains existed
        let json = r#"{
            "total_kv_blocks": 100,
            "max_num_seqs": 32,
            "max_num_batched_tokens": null,
            "tool_call_parser": null,
            "reasoning_parser": null
        }"#;

        let config: ModelRuntimeConfig = serde_json::from_str(json).unwrap();
        assert!(config.topology_domains.is_empty());
        assert!(config.kv_transfer_domain.is_none());
        assert!(config.kv_transfer_enforcement.is_none());
        assert!(config.kv_transfer_preferred_weight.is_none());
    }

    #[test]
    fn test_serde_round_trip_preserves_topology_transfer_fields_and_taints() {
        let mut config = ModelRuntimeConfig {
            taints: HashSet::from(["caller/taint=value".to_string()]),
            topology_domains: HashMap::from([
                ("zone".to_string(), "us-west-2b".to_string()),
                ("rack".to_string(), "rack1".to_string()),
            ]),
            kv_transfer_domain: Some("zone".to_string()),
            kv_transfer_enforcement: Some(KvTransferEnforcement::Preferred),
            kv_transfer_preferred_weight: Some(KvTransferPreferredWeight::try_from(0.85).unwrap()),
            ..Default::default()
        };
        config.add_topology_taints();

        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: ModelRuntimeConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.topology_domains.len(), 2);
        assert_eq!(deserialized.topology_domains["zone"], "us-west-2b");
        assert_eq!(deserialized.topology_domains["rack"], "rack1");
        assert_eq!(deserialized.kv_transfer_domain.as_deref(), Some("zone"));
        assert_eq!(
            deserialized.kv_transfer_enforcement,
            Some(KvTransferEnforcement::Preferred)
        );
        assert_eq!(
            deserialized
                .kv_transfer_preferred_weight
                .map(KvTransferPreferredWeight::get),
            Some(0.85)
        );
        assert!(deserialized.taints.contains("caller/taint=value"));
        assert!(
            deserialized
                .taints
                .contains("dynamo.topology/zone=us-west-2b")
        );
        assert!(deserialized.taints.contains("dynamo.topology/rack=rack1"));
    }

    #[test]
    fn test_serde_rejects_invalid_kv_transfer_fields() {
        for json in [
            r#"{"kv_transfer_enforcement":"fallback"}"#,
            r#"{"kv_transfer_preferred_weight":1.1}"#,
        ] {
            assert!(
                serde_json::from_str::<ModelRuntimeConfig>(json).is_err(),
                "{json} should fail"
            );
        }
    }

    #[test]
    fn test_validate_topology_config_accepts_matching_transfer_domain() {
        let config = ModelRuntimeConfig {
            topology_domains: HashMap::from([("zone".to_string(), "us-east-1a".to_string())]),
            kv_transfer_domain: Some("zone".to_string()),
            kv_transfer_enforcement: Some(KvTransferEnforcement::Required),
            ..Default::default()
        };

        config.validate_topology_config().unwrap();
    }

    #[test]
    fn test_validate_topology_config_accepts_preferred_with_weight() {
        let config = ModelRuntimeConfig {
            topology_domains: HashMap::from([("zone".to_string(), "us-east-1a".to_string())]),
            kv_transfer_domain: Some("zone".to_string()),
            kv_transfer_enforcement: Some(KvTransferEnforcement::Preferred),
            kv_transfer_preferred_weight: Some(KvTransferPreferredWeight::try_from(0.5).unwrap()),
            ..Default::default()
        };

        config.validate_topology_config().unwrap();
    }

    #[test]
    fn test_validate_topology_config_rejects_missing_transfer_domain_key() {
        let config = ModelRuntimeConfig {
            topology_domains: HashMap::from([("zone".to_string(), "us-east-1a".to_string())]),
            kv_transfer_domain: Some("rack".to_string()),
            ..Default::default()
        };

        let error = config.validate_topology_config().unwrap_err().to_string();
        assert!(error.contains("must reference a key in topology_domains"));
    }

    #[test]
    fn test_validate_topology_config_rejects_empty_transfer_domain_value() {
        let config = ModelRuntimeConfig {
            topology_domains: HashMap::from([("zone".to_string(), " ".to_string())]),
            kv_transfer_domain: Some("zone".to_string()),
            ..Default::default()
        };

        let error = config.validate_topology_config().unwrap_err().to_string();
        assert!(error.contains("must be non-empty"));
    }

    #[test]
    fn test_validate_topology_config_rejects_padded_topology_value() {
        let config = ModelRuntimeConfig {
            topology_domains: HashMap::from([("zone".to_string(), " us-east-1a ".to_string())]),
            ..Default::default()
        };

        let error = config.validate_topology_config().unwrap_err().to_string();
        assert!(error.contains("must not contain leading or trailing whitespace"));
    }

    #[test]
    fn test_validate_topology_config_rejects_empty_transfer_domain() {
        let config = ModelRuntimeConfig {
            kv_transfer_domain: Some(" ".to_string()),
            ..Default::default()
        };

        let error = config.validate_topology_config().unwrap_err().to_string();
        assert_eq!(error, "kv_transfer_domain must be non-empty when set");
    }

    #[test]
    fn test_validate_topology_config_rejects_padded_transfer_domain() {
        let config = ModelRuntimeConfig {
            topology_domains: HashMap::from([("zone".to_string(), "us-east-1a".to_string())]),
            kv_transfer_domain: Some(" zone ".to_string()),
            ..Default::default()
        };

        let error = config.validate_topology_config().unwrap_err().to_string();
        assert_eq!(
            error,
            "kv_transfer_domain must not contain leading or trailing whitespace"
        );
    }

    #[test]
    fn test_validate_topology_config_rejects_enforcement_without_transfer_domain() {
        let config = ModelRuntimeConfig {
            kv_transfer_enforcement: Some(KvTransferEnforcement::Required),
            ..Default::default()
        };

        let error = config.validate_topology_config().unwrap_err().to_string();
        assert_eq!(error, "kv_transfer_enforcement requires kv_transfer_domain");
    }

    #[test]
    fn test_validate_topology_config_requires_weight_for_preferred() {
        let config = ModelRuntimeConfig {
            topology_domains: HashMap::from([("zone".to_string(), "us-east-1a".to_string())]),
            kv_transfer_domain: Some("zone".to_string()),
            kv_transfer_enforcement: Some(KvTransferEnforcement::Preferred),
            ..Default::default()
        };

        let error = config.validate_topology_config().unwrap_err().to_string();
        assert_eq!(
            error,
            "kv_transfer_preferred_weight is required when kv_transfer_enforcement is preferred"
        );
    }

    #[test]
    fn test_validate_topology_config_rejects_weight_without_preferred() {
        let config = ModelRuntimeConfig {
            topology_domains: HashMap::from([("zone".to_string(), "us-east-1a".to_string())]),
            kv_transfer_domain: Some("zone".to_string()),
            kv_transfer_enforcement: Some(KvTransferEnforcement::Required),
            kv_transfer_preferred_weight: Some(KvTransferPreferredWeight::try_from(0.5).unwrap()),
            ..Default::default()
        };

        let error = config.validate_topology_config().unwrap_err().to_string();
        assert_eq!(
            error,
            "kv_transfer_preferred_weight can only be set when kv_transfer_enforcement is preferred"
        );
    }

    #[test]
    fn test_add_topology_taints_preserves_caller_supplied_taints() {
        let mut config = ModelRuntimeConfig {
            taints: HashSet::from(["caller/taint=value".to_string()]),
            ..Default::default()
        };
        config
            .topology_domains
            .insert("zone".to_string(), "us-east-1a".to_string());

        config.add_topology_taints();

        assert!(config.taints.contains("caller/taint=value"));
        assert!(config.taints.contains("dynamo.topology/zone=us-east-1a"));
    }

    #[test]
    fn test_add_topology_taints_rebuilds_generated_taints() {
        let mut config = ModelRuntimeConfig {
            taints: HashSet::from([
                "caller/taint=value".to_string(),
                "dynamo.topology/zone=stale-zone".to_string(),
                "dynamo.topology/rack=stale-rack".to_string(),
            ]),
            ..Default::default()
        };
        config
            .topology_domains
            .insert("zone".to_string(), "us-east-1a".to_string());

        config.add_topology_taints();

        assert!(config.taints.contains("caller/taint=value"));
        assert!(config.taints.contains("dynamo.topology/zone=us-east-1a"));
        assert!(!config.taints.contains("dynamo.topology/zone=stale-zone"));
        assert!(!config.taints.contains("dynamo.topology/rack=stale-rack"));

        config.topology_domains.clear();
        config.add_topology_taints();

        assert!(config.taints.contains("caller/taint=value"));
        assert!(
            !config
                .taints
                .iter()
                .any(|taint| taint.starts_with(TOPOLOGY_TAINT_PREFIX))
        );
    }
}
