// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_router::protocols::{RoutingConstraints, WorkerId};
use std::collections::{HashMap, HashSet};
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const RUNTIME_LORA_PROTOCOL_VERSION: u16 = 2;
pub const RUNTIME_LORA_KEY_PREFIX: &str = "dyn-lora-";
pub const MAX_RUNTIME_MODEL_ID_BYTES: usize = 4096;
pub const MAX_RUNTIME_BASE_MODEL_BYTES: usize = 512;
pub const MAX_RUNTIME_SOURCE_URI_BYTES: usize = 3072;
// Adapter identity is versioned independently of the worker wire protocol.
const IDENTITY_DOMAIN: &[u8] = b"dynamo-runtime-lora-v1\0";
pub const RUNTIME_LORA_SCHEMES_RUNTIME_KEY: &str = "runtime_lora_schemes";
pub const RUNTIME_LORA_SUPPORT_RUNTIME_KEY: &str = "supports_runtime_lora_resolution";
pub const RUNTIME_LORA_VERSIONS_RUNTIME_KEY: &str = "runtime_lora_protocol_versions";
static RUNTIME_LORA_CAPABILITY_TAINT: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| capability_taint(RUNTIME_LORA_PROTOCOL_VERSION));

fn capability_taint(protocol_version: u16) -> String {
    format!("dynamo.runtime-lora/v{protocol_version}")
}

/// Only these exact worker status/symbol pairs are safe for client rendering.
pub fn public_error_code(status: u16, code: &str) -> Option<&'static str> {
    Some(match (status, code) {
        (400, "invalid_lora_model_id") => "invalid_lora_model_id",
        (400, "runtime_lora_identity_mismatch") => "runtime_lora_identity_mismatch",
        (400, "runtime_lora_unsupported") => "runtime_lora_unsupported",
        (400, "unsupported_lora_scheme") => "unsupported_lora_scheme",
        (404, "lora_not_found") => "lora_not_found",
        (422, "invalid_lora_adapter") => "invalid_lora_adapter",
        (429, "lora_capacity_exceeded") => "lora_capacity_exceeded",
        (500, "lora_load_failed") => "lora_load_failed",
        (500, "lora_plugin_error") => "lora_plugin_error",
        (500, "runtime_lora_identity_collision") => "runtime_lora_identity_collision",
        (500, "runtime_lora_source_changed") => "runtime_lora_source_changed",
        (503, "lora_resolver_unavailable") => "lora_resolver_unavailable",
        (504, "runtime_lora_resolve_timeout") => "runtime_lora_resolve_timeout",
        _ => return None,
    })
}

/// Borrowed discovery metadata, without a dependency on model cards.
pub trait RuntimeLoraWorkerFacts {
    fn runtime_value(&self, key: &str) -> Option<&serde_json::Value>;
}

pub fn supports_runtime_lora<
    C: RuntimeLoraWorkerFacts + dynamo_kv_router::protocols::WorkerConfigLike,
>(
    config: &C,
    is_base: bool,
    requires_preprocessing: bool,
    is_aggregated: bool,
) -> bool {
    is_base
        && requires_preprocessing
        && is_aggregated
        && config
            .taints()
            .contains(RUNTIME_LORA_CAPABILITY_TAINT.as_str())
        && runtime_lora_worker_schemes_view(config).is_some()
}

struct RuntimeLoraSchemes<'a>(&'a [serde_json::Value]);

impl PartialEq for RuntimeLoraSchemes<'_> {
    fn eq(&self, other: &Self) -> bool {
        // Scheme order and duplicate advertisements do not change resolver policy.
        self.0 == other.0
            || (self.0.iter().all(|scheme| other.0.contains(scheme))
                && other.0.iter().all(|scheme| self.0.contains(scheme)))
    }
}

fn runtime_lora_worker_schemes_view(
    config: &impl RuntimeLoraWorkerFacts,
) -> Option<RuntimeLoraSchemes<'_>> {
    if config
        .runtime_value(RUNTIME_LORA_SUPPORT_RUNTIME_KEY)
        .and_then(serde_json::Value::as_bool)
        != Some(true)
    {
        return None;
    }
    let versions = config
        .runtime_value(RUNTIME_LORA_VERSIONS_RUNTIME_KEY)?
        .as_array()?;
    let mut supported = false;
    for version in versions {
        let version = u32::try_from(version.as_u64()?).ok()?;
        supported |= version == u32::from(RUNTIME_LORA_PROTOCOL_VERSION);
    }
    if !supported {
        return None;
    }
    let schemes = config
        .runtime_value(RUNTIME_LORA_SCHEMES_RUNTIME_KEY)?
        .as_array()?;
    (!schemes.is_empty() && schemes.iter().all(serde_json::Value::is_string))
        .then_some(RuntimeLoraSchemes(schemes))
}

pub fn runtime_lora_worker_schemes(config: &impl RuntimeLoraWorkerFacts) -> Option<Vec<String>> {
    let mut schemes: Vec<_> = runtime_lora_worker_schemes_view(config)?
        .0
        .iter()
        .filter_map(serde_json::Value::as_str)
        .map(str::to_owned)
        .collect();
    schemes.sort_unstable();
    schemes.dedup();
    Some(schemes)
}

pub fn runtime_lora_workers_have_homogeneous_resolvers<C: RuntimeLoraWorkerFacts>(
    worker_ids: &[WorkerId],
    worker_configs: &HashMap<WorkerId, C>,
) -> bool {
    let mut expected = None;
    for worker_id in worker_ids {
        let Some(config) = worker_configs.get(worker_id) else {
            return false;
        };
        let Some(schemes) = runtime_lora_worker_schemes_view(config) else {
            return false;
        };
        match &expected {
            Some(expected) if expected != &schemes => return false,
            None => expected = Some(schemes),
            _ => {}
        }
    }
    expected.is_some()
}

pub fn constrain_runtime_lora_workers<
    C: RuntimeLoraWorkerFacts + dynamo_kv_router::WorkerConfigLike,
>(
    is_runtime_lora: bool,
    allowed_worker_ids: Option<HashSet<WorkerId>>,
    live_worker_ids: &HashSet<WorkerId>,
    worker_configs: &HashMap<WorkerId, C>,
    routing_constraints: &RoutingConstraints,
) -> Option<HashSet<WorkerId>> {
    if !is_runtime_lora {
        return allowed_worker_ids;
    }

    let mut eligible = allowed_worker_ids.unwrap_or_else(|| live_worker_ids.clone());
    let mut malformed_workers = 0;
    let mut expected_schemes = None;
    let mut homogeneous = true;
    eligible.retain(|worker_id| {
        live_worker_ids.contains(worker_id)
            && worker_configs.get(worker_id).is_some_and(|config| {
                if !config
                    .taints()
                    .contains(RUNTIME_LORA_CAPABILITY_TAINT.as_str())
                    || !routing_constraints.is_compatible_with_worker_taints(config.taints())
                {
                    return false;
                }
                let Some(schemes) = runtime_lora_worker_schemes_view(config) else {
                    malformed_workers += 1;
                    return false;
                };
                match &expected_schemes {
                    Some(expected) if expected != &schemes => homogeneous = false,
                    None => expected_schemes = Some(schemes),
                    _ => {}
                }
                true
            })
    });
    if malformed_workers > 0 {
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            tracing::warn!(
                reason = "malformed_capability",
                excluded_workers = malformed_workers,
                "Excluding malformed runtime LoRA workers"
            )
        });
    }
    if !homogeneous {
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            tracing::warn!(
                reason = "heterogeneous_resolvers",
                workers = eligible.len(),
                "Runtime LoRA workers advertise incompatible resolver policies"
            )
        });
        eligible.clear();
    }
    Some(eligible)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeLoraConfig {
    pub enabled: bool,
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RuntimeLoraSourceUri(String);

impl RuntimeLoraSourceUri {
    pub fn new(value: String) -> Self {
        Self(value)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for RuntimeLoraSourceUri {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("<redacted-runtime-lora-uri>")
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct RuntimeLoraSelection {
    pub requested_model: String,
    pub base_model_name: String,
    pub source_uri: RuntimeLoraSourceUri,
    pub adapter_key: String,
    pub protocol_version: u16,
}

impl fmt::Debug for RuntimeLoraSelection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RuntimeLoraSelection")
            .field("requested_model", &"<redacted-runtime-lora-model>")
            .field("base_model_name", &self.base_model_name)
            .field("source_uri", &self.source_uri)
            .field("adapter_key", &self.adapter_key)
            .field("protocol_version", &self.protocol_version)
            .finish()
    }
}

impl RuntimeLoraSelection {
    pub fn capability_taint(&self) -> String {
        capability_taint(self.protocol_version)
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RuntimeLoraError {
    #[error("{0}")]
    InvalidConfiguration(String),
    #[error("runtime LoRA model id is invalid")]
    InvalidModelId,
    #[error("runtime LoRA base model must be specified when it cannot be inferred")]
    BaseModelRequired,
    #[error("runtime LoRA base model was not found")]
    BaseModelNotFound,
    #[error("runtime LoRA is unsupported for base model '{0}'")]
    BaseModelUnsupported(String),
}

impl RuntimeLoraConfig {
    pub fn from_env() -> Result<Self, RuntimeLoraError> {
        use dynamo_runtime::config::environment_names::llm;

        let enabled = dynamo_runtime::config::env_is_truthy(llm::DYN_LORA_RUNTIME_LOAD_ENABLED);
        let lora_enabled = dynamo_runtime::config::env_is_truthy(llm::DYN_LORA_ENABLED);
        Self::from_flags(enabled, lora_enabled)
    }

    fn from_flags(enabled: bool, lora_enabled: bool) -> Result<Self, RuntimeLoraError> {
        if enabled && !lora_enabled {
            return Err(RuntimeLoraError::InvalidConfiguration(
                "DYN_LORA_RUNTIME_LOAD_ENABLED requires DYN_LORA_ENABLED=true".to_string(),
            ));
        }
        Ok(Self { enabled })
    }
}

pub fn identity_digest(base_model_name: &str, source_uri: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(IDENTITY_DOMAIN);
    hasher.update(base_model_name.as_bytes());
    hasher.update(b"\0");
    hasher.update(source_uri.as_bytes());
    hasher.finalize().into()
}

pub fn adapter_key(base_model_name: &str, source_uri: &str) -> String {
    let digest = identity_digest(base_model_name, source_uri);
    let mut key = String::with_capacity(RUNTIME_LORA_KEY_PREFIX.len() + 32);
    key.push_str(RUNTIME_LORA_KEY_PREFIX);
    for byte in &digest[..16] {
        use std::fmt::Write;
        write!(key, "{byte:02x}").expect("writing to String cannot fail");
    }
    key
}

fn contains_forbidden_character(value: &str) -> bool {
    value
        .bytes()
        .any(|byte| byte.is_ascii_control() || byte == 0x7f)
}

fn validate_opaque_source(source_uri: &str) -> Result<(), RuntimeLoraError> {
    if source_uri.is_empty()
        || source_uri.len() > MAX_RUNTIME_SOURCE_URI_BYTES
        || contains_forbidden_character(source_uri)
    {
        return Err(RuntimeLoraError::InvalidModelId);
    }

    Ok(())
}

pub fn validate_runtime_lora_identity(
    model: &str,
    base: &str,
    source: &str,
    key: &str,
) -> Result<(), RuntimeLoraError> {
    if base.is_empty()
        || base.len() > MAX_RUNTIME_BASE_MODEL_BYTES
        || contains_forbidden_character(base)
        || model != base
    {
        return Err(RuntimeLoraError::InvalidModelId);
    }
    validate_opaque_source(source)?;
    if adapter_key(base, source) != key {
        return Err(RuntimeLoraError::InvalidModelId);
    }
    Ok(())
}

pub fn resolve_runtime_lora_model<K, F, B>(
    requested_model: &str,
    config: &RuntimeLoraConfig,
    is_known_exact: K,
    resolve_base: F,
    infer_base: B,
) -> Result<Option<RuntimeLoraSelection>, RuntimeLoraError>
where
    K: Fn(&str) -> bool,
    F: Fn(&str) -> Option<String>,
    B: Fn() -> Result<String, RuntimeLoraError>,
{
    if !config.enabled {
        return Ok(None);
    }
    if is_known_exact(requested_model) {
        return Ok(None);
    }
    if requested_model.is_empty()
        || requested_model.len() > MAX_RUNTIME_MODEL_ID_BYTES
        || contains_forbidden_character(requested_model)
    {
        return Err(RuntimeLoraError::InvalidModelId);
    }

    let (canonical_base, source_uri) = if let Some((base, source)) = requested_model.split_once('|')
    {
        if base.is_empty() || source.is_empty() {
            return Err(RuntimeLoraError::InvalidModelId);
        }
        if base.len() > MAX_RUNTIME_BASE_MODEL_BYTES || contains_forbidden_character(base) {
            return Err(RuntimeLoraError::InvalidModelId);
        }
        let canonical = match resolve_base(base) {
            Some(canonical) => canonical,
            None if is_known_exact(base) => {
                return Err(RuntimeLoraError::BaseModelUnsupported(base.to_string()));
            }
            None => return Err(RuntimeLoraError::BaseModelNotFound),
        };
        (canonical, source)
    } else {
        (infer_base()?, requested_model)
    };

    if canonical_base.len() > MAX_RUNTIME_BASE_MODEL_BYTES
        || contains_forbidden_character(&canonical_base)
    {
        return Err(RuntimeLoraError::InvalidModelId);
    }
    validate_opaque_source(source_uri)?;

    Ok(Some(RuntimeLoraSelection {
        requested_model: requested_model.to_string(),
        adapter_key: adapter_key(&canonical_base, source_uri),
        base_model_name: canonical_base,
        source_uri: RuntimeLoraSourceUri::new(source_uri.to_string()),
        protocol_version: RUNTIME_LORA_PROTOCOL_VERSION,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestWorker(serde_json::Value, HashSet<String>, std::cell::Cell<usize>);

    impl RuntimeLoraWorkerFacts for TestWorker {
        fn runtime_value(&self, key: &str) -> Option<&serde_json::Value> {
            self.2.set(self.2.get() + 1);
            self.0.get(key)
        }
    }

    impl dynamo_kv_router::WorkerConfigLike for TestWorker {
        fn data_parallel_start_rank(&self) -> u32 {
            0
        }
        fn data_parallel_size(&self) -> u32 {
            1
        }
        fn max_num_batched_tokens(&self) -> Option<u64> {
            None
        }
        fn total_kv_blocks(&self) -> Option<u64> {
            None
        }
        fn taints(&self) -> &HashSet<String> {
            &self.1
        }
    }

    fn capable_worker(schemes: &[&str]) -> TestWorker {
        TestWorker(
            serde_json::json!({
                "supports_runtime_lora_resolution": true,
                "runtime_lora_protocol_versions": [2],
                "runtime_lora_schemes": schemes,
            }),
            HashSet::from([RUNTIME_LORA_CAPABILITY_TAINT.to_string()]),
            Default::default(),
        )
    }

    #[test]
    fn public_error_codes_require_exact_status_and_symbol() {
        for (status, symbol) in [
            (400, "invalid_lora_model_id"),
            (400, "runtime_lora_identity_mismatch"),
            (400, "runtime_lora_unsupported"),
            (400, "unsupported_lora_scheme"),
            (404, "lora_not_found"),
            (422, "invalid_lora_adapter"),
            (429, "lora_capacity_exceeded"),
            (500, "lora_load_failed"),
            (500, "lora_plugin_error"),
            (500, "runtime_lora_identity_collision"),
            (500, "runtime_lora_source_changed"),
            (503, "lora_resolver_unavailable"),
            (504, "runtime_lora_resolve_timeout"),
        ] {
            assert_eq!(public_error_code(status, symbol), Some(symbol));
            assert_eq!(public_error_code(status + 1, symbol), None);
            assert_eq!(
                public_error_code(status, &format!("{symbol}: token=secret")),
                None
            );
            assert_eq!(public_error_code(status, &format!("{symbol}\n")), None);
        }
        assert_eq!(public_error_code(500, "lora_unknown"), None);
        assert_eq!(public_error_code(500, "provider:///private/path"), None);
    }

    #[test]
    fn runtime_identity_validation_preserves_opaque_sources() {
        let source = "wandb-artifact:///entity/model|version";
        let key = adapter_key("base", source);
        assert!(validate_runtime_lora_identity("base", "base", source, &key).is_ok());
        assert!(validate_runtime_lora_identity("other", "base", source, &key).is_err());
        assert!(validate_runtime_lora_identity("base", "base", source, "wrong").is_err());
        for invalid in ["", "x\n", &"a".repeat(MAX_RUNTIME_SOURCE_URI_BYTES + 1)] {
            let matching_key = adapter_key("base", invalid);
            assert!(
                validate_runtime_lora_identity("base", "base", invalid, &matching_key).is_err()
            );
        }
        for invalid in ["", "x\0", &"a".repeat(MAX_RUNTIME_BASE_MODEL_BYTES + 1)] {
            let matching_key = adapter_key(invalid, source);
            assert!(
                validate_runtime_lora_identity(invalid, invalid, source, &matching_key).is_err()
            );
        }
        let max_base = "b".repeat(MAX_RUNTIME_BASE_MODEL_BYTES);
        let max_source = "s".repeat(MAX_RUNTIME_SOURCE_URI_BYTES);
        let max_key = adapter_key(&max_base, &max_source);
        assert!(
            validate_runtime_lora_identity(&max_base, &max_base, &max_source, &max_key).is_ok()
        );
    }

    #[test]
    fn capability_decoding_normalizes_sets_and_rejects_malformed_values() {
        assert_eq!(
            runtime_lora_worker_schemes(&capable_worker(&["b", "a", "a"])),
            Some(vec!["a".into(), "b".into()])
        );
        assert!(runtime_lora_worker_schemes(&capable_worker(&[])).is_none());
        for facts in [
            serde_json::json!({"supports_runtime_lora_resolution": "true"}),
            serde_json::json!({"supports_runtime_lora_resolution": true, "runtime_lora_protocol_versions": [65538], "runtime_lora_schemes": ["a"]}),
            serde_json::json!({"supports_runtime_lora_resolution": true, "runtime_lora_protocol_versions": [2, -1], "runtime_lora_schemes": ["a"]}),
            serde_json::json!({"supports_runtime_lora_resolution": true, "runtime_lora_protocol_versions": [2, 4294967296_u64], "runtime_lora_schemes": ["a"]}),
            serde_json::json!({"supports_runtime_lora_resolution": true, "runtime_lora_protocol_versions": [2, 2.0], "runtime_lora_schemes": ["a"]}),
            serde_json::json!({"supports_runtime_lora_resolution": true, "runtime_lora_protocol_versions": [2], "runtime_lora_schemes": [null]}),
        ] {
            assert!(
                runtime_lora_worker_schemes(&TestWorker(facts, HashSet::new(), Default::default()))
                    .is_none()
            );
        }
    }

    #[test]
    fn runtime_restriction_decodes_each_worker_capability_once() {
        let configs = HashMap::from([
            (1, capable_worker(&["a", "b"])),
            (2, capable_worker(&["b", "a", "a"])),
        ]);
        let live = HashSet::from([1, 2]);
        assert_eq!(
            constrain_runtime_lora_workers(
                true,
                None,
                &live,
                &configs,
                &RoutingConstraints::default()
            ),
            Some(live)
        );
        for worker in configs.values() {
            assert_eq!(worker.2.get(), 3, "read each capability field once");
        }
    }

    #[test]
    fn capability_view_borrows_current_discovery_values() {
        let mut worker = capable_worker(&["a", "b"]);
        let schemes = runtime_lora_worker_schemes_view(&worker).unwrap();
        assert!(std::ptr::eq(
            schemes.0,
            worker.0[RUNTIME_LORA_SCHEMES_RUNTIME_KEY]
                .as_array()
                .unwrap()
                .as_slice()
        ));
        assert_eq!(worker.2.get(), 3);

        worker.0[RUNTIME_LORA_SCHEMES_RUNTIME_KEY] = serde_json::json!(["updated"]);
        assert_eq!(
            runtime_lora_worker_schemes_view(&worker).unwrap().0,
            &[serde_json::json!("updated")]
        );
        worker.0[RUNTIME_LORA_SUPPORT_RUNTIME_KEY] = serde_json::json!(false);
        assert!(runtime_lora_worker_schemes_view(&worker).is_none());
    }

    #[test]
    fn runtime_restriction_requires_requested_capability_taint() {
        let capable = capable_worker(&["custom"]);
        let mut missing_taint = capable_worker(&["custom"]);
        missing_taint.1.clear();
        assert!(supports_runtime_lora(&capable, true, true, true));
        assert!(!supports_runtime_lora(&missing_taint, true, true, true));
        let configs = HashMap::from([(1, capable), (2, missing_taint)]);
        let constraints = RoutingConstraints {
            required_taints: HashSet::from(["dynamo.runtime-lora/v2".into()]),
            ..Default::default()
        };
        assert_eq!(
            constrain_runtime_lora_workers(
                true,
                None,
                &HashSet::from([1, 2]),
                &configs,
                &constraints
            ),
            Some(HashSet::from([1]))
        );
        assert_eq!(
            configs[&2].2.get(),
            0,
            "reject missing taint before capability decoding"
        );
    }

    #[test]
    fn runtime_restriction_preserves_authoritative_empty_and_rejects_heterogeneity() {
        let configs = HashMap::from([(1, capable_worker(&["a"])), (2, capable_worker(&["b"]))]);
        let live = HashSet::from([1, 2, 3]);
        let constraints = RoutingConstraints::default();
        assert_eq!(
            constrain_runtime_lora_workers(true, None, &live, &configs, &constraints),
            Some(HashSet::new())
        );
        assert_eq!(
            constrain_runtime_lora_workers(
                true,
                Some(HashSet::new()),
                &live,
                &configs,
                &constraints
            ),
            Some(HashSet::new())
        );
        assert_eq!(
            constrain_runtime_lora_workers(
                true,
                Some(HashSet::from([1, 3, 4])),
                &live,
                &configs,
                &constraints
            ),
            Some(HashSet::from([1]))
        );
        assert_eq!(
            constrain_runtime_lora_workers(false, None, &live, &configs, &constraints),
            None
        );
    }

    #[test]
    fn runtime_restriction_requires_capability_taint_without_caller_constraints() {
        let mut untainted = capable_worker(&["a"]);
        untainted.1.clear();
        let configs = HashMap::from([(1, capable_worker(&["a"])), (2, untainted)]);
        let live = HashSet::from([1, 2]);
        assert_eq!(
            constrain_runtime_lora_workers(
                true,
                None,
                &live,
                &configs,
                &RoutingConstraints::default()
            ),
            Some(HashSet::from([1]))
        );
        assert_eq!(
            constrain_runtime_lora_workers(
                false,
                None,
                &live,
                &configs,
                &RoutingConstraints::default()
            ),
            None
        );
    }

    fn config() -> RuntimeLoraConfig {
        RuntimeLoraConfig { enabled: true }
    }

    #[test]
    fn runtime_loading_requires_lora_serving() {
        assert_eq!(
            RuntimeLoraConfig::from_flags(true, false),
            Err(RuntimeLoraError::InvalidConfiguration(
                "DYN_LORA_RUNTIME_LOAD_ENABLED requires DYN_LORA_ENABLED=true".to_string()
            ))
        );
        assert_eq!(
            RuntimeLoraConfig::from_flags(true, true),
            Ok(RuntimeLoraConfig { enabled: true })
        );
    }

    fn resolve_base(name: &str) -> Option<String> {
        match name {
            "base" | "base-alias" => Some("base".to_string()),
            "preloaded" => Some("preloaded".to_string()),
            _ => None,
        }
    }

    #[test]
    fn adapter_key_matches_v1_golden_vectors() {
        assert_eq!(
            adapter_key(
                "meta-llama/Llama-3.1-8B-Instruct",
                "wandb-artifact:///team/project/adapter:v7",
            ),
            "dyn-lora-c7246b154263a336c006517e4bc6d7c8"
        );
        assert_eq!(
            adapter_key("base", "custom://adapter@sha256:abc"),
            "dyn-lora-e3cfcd02f2e62cb70f45999324e7d4de"
        );
        assert_eq!(
            adapter_key("base", "wandb-artifact:///entity/project/artifact:v1"),
            "dyn-lora-27056eb300024fbfc91334895bba1e26"
        );
        assert_eq!(
            adapter_key("base", "wandb-artifact:///a|b:v1"),
            "dyn-lora-74bbe88c562d9d177d098e3ba118851a"
        );
    }

    #[test]
    fn exact_known_model_takes_precedence_over_runtime_grammar() {
        let selection = resolve_runtime_lora_model(
            "preloaded",
            &config(),
            |name| resolve_base(name).is_some(),
            resolve_base,
            || Ok("base".to_string()),
        )
        .expect("known model should be accepted");
        assert!(selection.is_none());
    }

    #[test]
    fn registered_models_bypass_runtime_only_limits() {
        for name in [
            "a".repeat(MAX_RUNTIME_MODEL_ID_BYTES + 1),
            "registered\nname".into(),
        ] {
            assert_eq!(
                resolve_runtime_lora_model(
                    &name,
                    &config(),
                    |candidate| candidate == name,
                    |_| panic!("registered model must not resolve a runtime base"),
                    || panic!("registered model must not infer a runtime base"),
                ),
                Ok(None)
            );
            assert_eq!(
                resolve_runtime_lora_model(
                    &name,
                    &config(),
                    |_| false,
                    resolve_base,
                    || Ok("base".into())
                ),
                Err(RuntimeLoraError::InvalidModelId)
            );
        }
    }

    #[test]
    fn combined_form_binds_canonical_base_and_opaque_source() {
        let selection = resolve_runtime_lora_model(
            "base-alias|wandb-artifact:///team/project/adapter:v7|segment",
            &config(),
            |name| resolve_base(name).is_some(),
            resolve_base,
            || Err(RuntimeLoraError::BaseModelRequired),
        )
        .expect("runtime model should parse")
        .expect("runtime selection expected");

        assert_eq!(
            selection.requested_model,
            "base-alias|wandb-artifact:///team/project/adapter:v7|segment"
        );
        assert_eq!(selection.base_model_name, "base");
        assert_eq!(
            selection.source_uri.as_str(),
            "wandb-artifact:///team/project/adapter:v7|segment"
        );
        assert_eq!(selection.protocol_version, 2);
        assert_eq!(selection.capability_taint(), "dynamo.runtime-lora/v2");
        assert_eq!(
            selection.adapter_key,
            adapter_key("base", selection.source_uri.as_str())
        );
    }

    #[test]
    fn bare_opaque_source_requires_a_unique_inferred_base() {
        assert_eq!(
            resolve_runtime_lora_model(
                "wandb-artifact:///team/project/adapter:v7",
                &config(),
                |name| resolve_base(name).is_some(),
                resolve_base,
                || Err(RuntimeLoraError::BaseModelRequired),
            ),
            Err(RuntimeLoraError::BaseModelRequired)
        );

        let selection = resolve_runtime_lora_model(
            "wandb-artifact:///team/project/adapter:v7",
            &config(),
            |name| resolve_base(name).is_some(),
            resolve_base,
            || Ok("base".to_string()),
        )
        .expect("runtime model should parse")
        .expect("runtime selection expected");
        assert_eq!(selection.base_model_name, "base");
    }

    #[test]
    fn known_but_ineligible_base_is_unsupported_for_combined_and_bare_forms() {
        assert_eq!(
            resolve_runtime_lora_model(
                "unsupported|wandb-artifact:///team/project/adapter:v7",
                &config(),
                |name| name == "unsupported",
                |_| None,
                || Err(RuntimeLoraError::BaseModelRequired),
            ),
            Err(RuntimeLoraError::BaseModelUnsupported(
                "unsupported".to_string()
            ))
        );

        assert_eq!(
            resolve_runtime_lora_model(
                "wandb-artifact:///team/project/adapter:v7",
                &config(),
                |_| false,
                |_| None,
                || {
                    Err(RuntimeLoraError::BaseModelUnsupported(
                        "unsupported".to_string(),
                    ))
                },
            ),
            Err(RuntimeLoraError::BaseModelUnsupported(
                "unsupported".to_string()
            ))
        );
    }

    #[test]
    fn source_uri_debug_is_redacted_but_serde_preserves_wire_value() {
        let source = RuntimeLoraSourceUri::new(
            "wandb-artifact:///team/private/adapter:v7?harmless=value".to_string(),
        );
        assert_eq!(format!("{source:?}"), "<redacted-runtime-lora-uri>");
        assert_eq!(
            serde_json::to_string(&source).unwrap(),
            "\"wandb-artifact:///team/private/adapter:v7?harmless=value\""
        );
    }

    #[test]
    fn selection_debug_redacts_combined_model_and_source_uri() {
        let selection = resolve_runtime_lora_model(
            "base|wandb-artifact:///secret",
            &config(),
            |_| false,
            resolve_base,
            || Err(RuntimeLoraError::BaseModelRequired),
        )
        .unwrap()
        .unwrap();

        let debug = format!("{selection:?}");
        assert!(!debug.contains("wandb-artifact:///"));
        assert!(!debug.contains("secret"));
        assert!(debug.contains("<redacted-runtime-lora-model>"));
        assert!(debug.contains("<redacted-runtime-lora-uri>"));
    }

    #[test]
    fn rejects_unknown_base_and_unsafe_opaque_source() {
        assert_eq!(
            resolve_runtime_lora_model(
                "missing|wandb-artifact:///team/project/adapter:v7",
                &config(),
                |name| resolve_base(name).is_some(),
                resolve_base,
                || Err(RuntimeLoraError::BaseModelRequired),
            ),
            Err(RuntimeLoraError::BaseModelNotFound)
        );
        let opaque = "not-a-uri?token=opaque#provider-syntax";
        let selection = resolve_runtime_lora_model(
            &format!("base|{opaque}"),
            &config(),
            |name| resolve_base(name).is_some(),
            resolve_base,
            || Err(RuntimeLoraError::BaseModelRequired),
        )
        .unwrap()
        .unwrap();
        assert_eq!(selection.source_uri.as_str(), opaque);

        assert_eq!(
            resolve_runtime_lora_model(
                "base|opaque\nnext",
                &config(),
                |name| resolve_base(name).is_some(),
                resolve_base,
                || Err(RuntimeLoraError::BaseModelRequired),
            ),
            Err(RuntimeLoraError::InvalidModelId)
        );
    }

    #[test]
    fn disabled_runtime_loading_preserves_existing_behavior() {
        let mut disabled = config();
        disabled.enabled = false;
        assert_eq!(
            resolve_runtime_lora_model(
                "base|wandb-artifact:///team/project/adapter:v7",
                &disabled,
                |name| resolve_base(name).is_some(),
                resolve_base,
                || Ok("base".to_string()),
            ),
            Ok(None)
        );
    }
}
