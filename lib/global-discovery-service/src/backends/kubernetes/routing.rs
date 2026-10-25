// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Model identity from operator configuration; live frontend addresses remain MCS-owned.

use std::collections::HashMap;

use dynamo_global_router_shared::{PoolKey, PoolKeyError, PoolLocation};
use serde::Deserialize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum McsRoutingConfigError {
    #[error("invalid MCS routing configuration: {0}")]
    Invalid(String),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    PoolKey(#[from] PoolKeyError),
}

/// Supplies the model identity EndpointSlices do not carry. It contains no pod addresses.
#[derive(Clone, Debug)]
pub struct McsRoutingConfig {
    pub(super) frontend_service_suffix: String,
    pub(super) frontend_port_name: String,
    pools: HashMap<PoolKey, PoolRoutingMetadata>,
}

#[derive(Clone, Debug)]
pub(super) struct PoolRoutingMetadata {
    pub model: String,
    pub location: Option<PoolLocation>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigFile {
    #[serde(default = "frontend_suffix")]
    frontend_service_suffix: String,
    #[serde(default = "http_port")]
    frontend_port_name: String,
    pools: Vec<PoolBinding>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PoolBinding {
    site_id: String,
    namespace: String,
    dgd_name: String,
    model: String,
    location: Option<PoolLocation>,
}

fn frontend_suffix() -> String {
    "-frontend".to_owned()
}
fn http_port() -> String {
    "http".to_owned()
}

impl McsRoutingConfig {
    pub fn from_json(bytes: &[u8]) -> Result<Self, McsRoutingConfigError> {
        let file: ConfigFile = serde_json::from_slice(bytes)?;
        let invalid = |message: &str| McsRoutingConfigError::Invalid(message.to_owned());
        let suffix = &file.frontend_service_suffix;
        if !suffix.starts_with('-')
            || suffix.len() < 2
            || !suffix
                .bytes()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'-')
        {
            return Err(invalid(
                "frontend_service_suffix must be a nonempty DNS suffix starting with '-'",
            ));
        }
        let port = &file.frontend_port_name;
        if port.is_empty()
            || port.len() > 15
            || !port
                .bytes()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'-')
            || !port.bytes().any(|c| c.is_ascii_lowercase())
            || port.starts_with('-')
            || port.ends_with('-')
        {
            return Err(invalid(
                "frontend_port_name must be a Kubernetes named port",
            ));
        }
        if file.pools.is_empty() {
            return Err(invalid("pools must not be empty"));
        }
        let mut pools = HashMap::new();
        for binding in file.pools {
            let key = PoolKey::new(binding.site_id, binding.namespace, binding.dgd_name)?;
            if binding.model.is_empty()
                || binding.model.trim() != binding.model
                || binding.model.chars().any(char::is_control)
            {
                return Err(invalid(
                    "model must be nonempty without surrounding whitespace or control characters",
                ));
            }
            if binding
                .location
                .as_ref()
                .is_some_and(|l| l.region.is_empty() || l.region.trim() != l.region)
            {
                return Err(invalid(
                    "location.region must be nonempty without surrounding whitespace",
                ));
            }
            if pools
                .insert(
                    key,
                    PoolRoutingMetadata {
                        model: binding.model,
                        location: binding.location,
                    },
                )
                .is_some()
            {
                return Err(invalid("duplicate pool key"));
            }
        }
        Ok(Self {
            frontend_service_suffix: file.frontend_service_suffix,
            frontend_port_name: file.frontend_port_name,
            pools,
        })
    }

    pub(super) fn validate_relay_suffix(&self, suffix: &str) -> Result<(), McsRoutingConfigError> {
        if suffix.is_empty()
            || suffix.ends_with(&self.frontend_service_suffix)
            || self.frontend_service_suffix.ends_with(suffix)
        {
            return Err(McsRoutingConfigError::Invalid(
                "frontend and relay service suffixes must be distinct and unambiguous".to_owned(),
            ));
        }
        Ok(())
    }

    pub(super) fn get(&self, key: &PoolKey) -> Option<&PoolRoutingMetadata> {
        self.pools.get(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn configuration_requires_unique_valid_models_and_no_static_addresses() {
        let pool = json!({"site_id":"west", "namespace":"dynamo", "dgd_name":"blue", "model":"qwen", "location":{"region":"us-west-2"}});
        let config = |pools| json!({"pools":pools}).to_string();
        let valid = McsRoutingConfig::from_json(config(json!([pool.clone()])).as_bytes()).unwrap();
        assert_eq!(valid.frontend_port_name, "http");
        assert!(valid.validate_relay_suffix("-pool-relay").is_ok());
        assert!(valid.validate_relay_suffix("-frontend").is_err());
        assert!(
            McsRoutingConfig::from_json(config(json!([pool.clone(), pool.clone()])).as_bytes())
                .is_err()
        );
        assert!(McsRoutingConfig::from_json(config(json!([])).as_bytes()).is_err());
        let mut invalid = pool.clone();
        invalid["model"] = json!(" ");
        assert!(McsRoutingConfig::from_json(config(json!([invalid])).as_bytes()).is_err());
        let mut invalid = pool;
        invalid["frontends"] = json!(["http://stale:8000"]);
        assert!(McsRoutingConfig::from_json(config(json!([invalid])).as_bytes()).is_err());
    }
}
