// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const RUNTIME_LORA_PROTOCOL_VERSION: u16 = 2;
pub const RUNTIME_LORA_KEY_PREFIX: &str = "dyn-lora-";
pub const MAX_RUNTIME_MODEL_ID_BYTES: usize = 4096;
pub const MAX_RUNTIME_BASE_MODEL_BYTES: usize = 512;
pub const MAX_RUNTIME_SOURCE_URI_BYTES: usize = 3072;
const IDENTITY_DOMAIN: &[u8] = b"dynamo-runtime-lora-v1\0";

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
        format!("dynamo.runtime-lora/v{}", self.protocol_version)
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
    #[error("runtime LoRA base model '{0}' was not found")]
    BaseModelNotFound(String),
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

pub fn is_runtime_lora_key(name: &str) -> bool {
    name.strip_prefix(RUNTIME_LORA_KEY_PREFIX)
        .is_some_and(|suffix| {
            suffix.len() == 32 && suffix.bytes().all(|byte| byte.is_ascii_hexdigit())
        })
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
    B: Fn() -> Option<String>,
{
    if !config.enabled {
        return Ok(None);
    }
    if requested_model.is_empty()
        || requested_model.len() > MAX_RUNTIME_MODEL_ID_BYTES
        || contains_forbidden_character(requested_model)
    {
        return Err(RuntimeLoraError::InvalidModelId);
    }

    if is_known_exact(requested_model) {
        return Ok(None);
    }

    let (canonical_base, source_uri) = if let Some((base, source)) = requested_model.split_once('|')
    {
        if base.is_empty() || source.is_empty() {
            return Err(RuntimeLoraError::InvalidModelId);
        }
        if base.len() > MAX_RUNTIME_BASE_MODEL_BYTES || contains_forbidden_character(base) {
            return Err(RuntimeLoraError::InvalidModelId);
        }
        let canonical = resolve_base(base)
            .ok_or_else(|| RuntimeLoraError::BaseModelNotFound(base.to_string()))?;
        (canonical, source)
    } else {
        (
            infer_base().ok_or(RuntimeLoraError::BaseModelRequired)?,
            requested_model,
        )
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
        assert!(is_runtime_lora_key(
            "dyn-lora-74bbe88c562d9d177d098e3ba118851a"
        ));
        assert!(!is_runtime_lora_key("dyn-lora-not-a-digest"));
    }

    #[test]
    fn exact_known_model_takes_precedence_over_runtime_grammar() {
        let selection = resolve_runtime_lora_model(
            "preloaded",
            &config(),
            |name| resolve_base(name).is_some(),
            resolve_base,
            || Some("base".to_string()),
        )
        .expect("known model should be accepted");
        assert!(selection.is_none());
    }

    #[test]
    fn combined_form_binds_canonical_base_and_opaque_source() {
        let selection = resolve_runtime_lora_model(
            "base-alias|wandb-artifact:///team/project/adapter:v7|segment",
            &config(),
            |name| resolve_base(name).is_some(),
            resolve_base,
            || None,
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
                || None,
            ),
            Err(RuntimeLoraError::BaseModelRequired)
        );

        let selection = resolve_runtime_lora_model(
            "wandb-artifact:///team/project/adapter:v7",
            &config(),
            |name| resolve_base(name).is_some(),
            resolve_base,
            || Some("base".to_string()),
        )
        .expect("runtime model should parse")
        .expect("runtime selection expected");
        assert_eq!(selection.base_model_name, "base");
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
            || None,
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
                || None,
            ),
            Err(RuntimeLoraError::BaseModelNotFound("missing".to_string()))
        );
        let opaque = "not-a-uri?token=opaque#provider-syntax";
        let selection = resolve_runtime_lora_model(
            &format!("base|{opaque}"),
            &config(),
            |name| resolve_base(name).is_some(),
            resolve_base,
            || None,
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
                || None,
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
                || Some("base".to_string()),
            ),
            Ok(None)
        );
    }
}
