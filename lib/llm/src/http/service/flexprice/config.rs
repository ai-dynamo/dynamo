// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configuration for the JWT auth + FlexPrice usage-billing layer.
//!
//! The auth layer activates only when `DYN_AUTH_ENABLED=true`. FlexPrice usage
//! emission is optional on top of it — it requires `DYN_FLEXPRICE_ENABLED=true`,
//! which in turn requires `DYN_AUTH_ENABLED=true` because the org UUID is
//! sourced from the authenticated JWT.

use anyhow::{Result, bail};
use dynamo_runtime::config::environment_names::llm as env_llm;

/// JWT bearer-token authentication configuration for inference endpoints.
#[derive(Debug, Clone, Default)]
pub struct AuthConfig {
    pub enabled: bool,
    /// HMAC secret(s), tried in order to support key rotation.
    pub secret_keys: Vec<String>,
    /// Org UUID allowlist; empty means allow all authenticated orgs.
    pub valid_orgs: Vec<String>,
}

impl AuthConfig {
    pub fn from_env() -> Self {
        Self {
            enabled: dynamo_runtime::config::env_is_truthy(env_llm::DYN_AUTH_ENABLED),
            secret_keys: split_csv_env(env_llm::DYN_AUTH_SECRET_KEY),
            valid_orgs: split_csv_env(env_llm::DYN_AUTH_VALID_ORGS),
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.enabled && self.secret_keys.is_empty() {
            bail!(
                "{} is required when {}=true",
                env_llm::DYN_AUTH_SECRET_KEY,
                env_llm::DYN_AUTH_ENABLED
            );
        }
        Ok(())
    }
}

/// FlexPrice async usage-billing configuration.
#[derive(Debug, Clone, Default)]
pub struct FlexPriceConfig {
    pub enabled: bool,
    pub api_key: String,
    pub api_host: String,
    pub event_name: String,
    pub source_name: String,
}

impl FlexPriceConfig {
    pub fn from_env() -> Self {
        Self {
            enabled: dynamo_runtime::config::env_is_truthy(env_llm::DYN_FLEXPRICE_ENABLED),
            api_key: std::env::var(env_llm::DYN_FLEXPRICE_API_KEY).unwrap_or_default(),
            api_host: std::env::var(env_llm::DYN_FLEXPRICE_API_HOST)
                .unwrap_or_default()
                .trim_end_matches('/')
                .to_string(),
            event_name: std::env::var(env_llm::DYN_FLEXPRICE_EVENT_NAME).unwrap_or_default(),
            source_name: std::env::var(env_llm::DYN_FLEXPRICE_SOURCE_NAME).unwrap_or_default(),
        }
    }

    pub fn validate(&self, auth_enabled: bool) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        if !auth_enabled {
            bail!(
                "{}=true requires {}=true (org ID is sourced from the authenticated JWT)",
                env_llm::DYN_FLEXPRICE_ENABLED,
                env_llm::DYN_AUTH_ENABLED
            );
        }
        if self.api_key.is_empty() {
            bail!(
                "{} is required when {}=true",
                env_llm::DYN_FLEXPRICE_API_KEY,
                env_llm::DYN_FLEXPRICE_ENABLED
            );
        }
        if self.api_host.is_empty() {
            bail!(
                "{} is required when {}=true",
                env_llm::DYN_FLEXPRICE_API_HOST,
                env_llm::DYN_FLEXPRICE_ENABLED
            );
        }
        Ok(())
    }

    pub fn resolve_event_name(&self, model_name: &str) -> String {
        if !self.event_name.is_empty() {
            return self.event_name.clone();
        }
        if model_name.is_empty() {
            "dynamo-llm-usage".to_string()
        } else {
            format!("{model_name}-llm-usage")
        }
    }

    pub fn resolve_source_name(&self, model_name: &str) -> String {
        if !self.source_name.is_empty() {
            self.source_name.clone()
        } else if model_name.is_empty() {
            "dynamo".to_string()
        } else {
            model_name.to_string()
        }
    }
}

fn split_csv_env(name: &str) -> Vec<String> {
    std::env::var(name)
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_event_name_defaults_to_model_suffix() {
        let cfg = FlexPriceConfig::default();
        assert_eq!(cfg.resolve_event_name("llama-3"), "llama-3-llm-usage");
        assert_eq!(cfg.resolve_event_name(""), "dynamo-llm-usage");
    }

    #[test]
    fn resolve_event_name_prefers_override() {
        let cfg = FlexPriceConfig {
            event_name: "custom-event".to_string(),
            ..Default::default()
        };
        assert_eq!(cfg.resolve_event_name("llama-3"), "custom-event");
    }

    #[test]
    fn resolve_source_name_defaults_to_model() {
        let cfg = FlexPriceConfig::default();
        assert_eq!(cfg.resolve_source_name("llama-3"), "llama-3");
        assert_eq!(cfg.resolve_source_name(""), "dynamo");
    }

    #[test]
    fn flexprice_requires_auth() {
        let cfg = FlexPriceConfig {
            enabled: true,
            api_key: "k".to_string(),
            api_host: "api.flexprice.io".to_string(),
            ..Default::default()
        };
        assert!(cfg.validate(false).is_err());
        assert!(cfg.validate(true).is_ok());
    }

    #[test]
    fn flexprice_requires_api_key_and_host() {
        let cfg = FlexPriceConfig {
            enabled: true,
            ..Default::default()
        };
        assert!(cfg.validate(true).is_err());
    }

    #[test]
    fn auth_requires_secret_keys_when_enabled() {
        let cfg = AuthConfig {
            enabled: true,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());

        let cfg = AuthConfig {
            enabled: true,
            secret_keys: vec!["secret".to_string()],
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }
}
