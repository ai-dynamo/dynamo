// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-independent identities for pools in a Global View.
//!
//! A routing pool is one DGD in the initial deployment. Its identity is stable
//! across process and relay restarts, while relay and KV producer generations
//! are tracked separately by the source adapter.

pub mod source;
pub mod state;

use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// An opaque routing identity. Callers must not parse its representation.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PoolId(String);

impl PoolId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for PoolId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Stable deployment coordinates used by the POC derivation.
///
/// `namespace` is the Kubernetes namespace containing the DGD, not the
/// Dynamo runtime namespace derived by the operator.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PoolKey {
    site_id: String,
    namespace: String,
    dgd_name: String,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PoolKeyError {
    #[error("{field} must be non-empty without surrounding whitespace or control characters")]
    InvalidField { field: &'static str },
}

impl PoolKey {
    pub fn new(
        site_id: impl Into<String>,
        namespace: impl Into<String>,
        dgd_name: impl Into<String>,
    ) -> Result<Self, PoolKeyError> {
        let site_id = site_id.into();
        let namespace = namespace.into();
        let dgd_name = dgd_name.into();
        validate(&site_id, "site_id")?;
        validate(&namespace, "namespace")?;
        validate(&dgd_name, "dgd_name")?;
        Ok(Self {
            site_id,
            namespace,
            dgd_name,
        })
    }

    pub fn site_id(&self) -> &str {
        &self.site_id
    }

    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    pub fn dgd_name(&self) -> &str {
        &self.dgd_name
    }
}

fn validate(value: &str, field: &'static str) -> Result<(), PoolKeyError> {
    if value.is_empty() || value.trim() != value || value.chars().any(char::is_control) {
        return Err(PoolKeyError::InvalidField { field });
    }
    Ok(())
}

/// Global View owns assignment of routing pool IDs. Implementations can
/// change without leaking derivation details into catalog or routing code.
pub trait PoolIdDeriver: Send + Sync {
    fn derive(&self, key: &PoolKey) -> PoolId;
}

/// Initial deterministic derivation, shared by every Global Router replica.
///
/// Changing this algorithm changes pool IDs and requires a deployment
/// migration. The API continues treating IDs as opaque strings.
#[derive(Clone, Copy, Debug, Default)]
pub struct V1PoolIdDeriver;

impl PoolIdDeriver for V1PoolIdDeriver {
    fn derive(&self, key: &PoolKey) -> PoolId {
        let mut hasher = blake3::Hasher::new_derive_key("dynamo.global_view.pool_id.v1");
        for part in [&key.site_id, &key.namespace, &key.dgd_name] {
            let bytes = part.as_bytes();
            hasher.update(&(bytes.len() as u64).to_be_bytes());
            hasher.update(bytes);
        }
        PoolId(format!("pool-v1-{}", hasher.finalize().to_hex()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(site: &str, namespace: &str, dgd: &str) -> PoolKey {
        PoolKey::new(site, namespace, dgd).unwrap()
    }

    #[test]
    fn identity_survives_deriver_restart() {
        let pool = key("ohio", "dynamo", "mocker-1");
        assert_eq!(V1PoolIdDeriver.derive(&pool), V1PoolIdDeriver.derive(&pool));
    }

    #[test]
    fn moving_or_renaming_a_dgd_changes_identity() {
        let deriver = V1PoolIdDeriver;
        let original = deriver.derive(&key("ohio", "dynamo", "mocker-1"));
        for changed in [
            key("west", "dynamo", "mocker-1"),
            key("ohio", "other", "mocker-1"),
            key("ohio", "dynamo", "mocker-2"),
        ] {
            assert_ne!(original, deriver.derive(&changed));
        }
    }

    #[test]
    fn component_boundaries_cannot_alias() {
        let deriver = V1PoolIdDeriver;
        assert_ne!(
            deriver.derive(&key("ab", "c", "d")),
            deriver.derive(&key("a", "bc", "d")),
        );
    }

    #[test]
    fn rejects_invalid_coordinates() {
        assert_eq!(
            PoolKey::new("", "dynamo", "mocker-1"),
            Err(PoolKeyError::InvalidField { field: "site_id" })
        );
        assert_eq!(
            PoolKey::new("ohio", " dynamo", "mocker-1"),
            Err(PoolKeyError::InvalidField { field: "namespace" })
        );
        assert_eq!(
            PoolKey::new("ohio", "dynamo", "mocker\n1"),
            Err(PoolKeyError::InvalidField { field: "dgd_name" })
        );
    }

    #[test]
    fn api_representation_is_an_opaque_string() {
        let id = V1PoolIdDeriver.derive(&key("ohio", "dynamo", "mocker-1"));
        assert_eq!(serde_json::to_string(&id).unwrap(), format!("\"{id}\""));
    }
}
